"""chat/knowledge/db_store.py — pgvector-backed persistent knowledge store.

Architecture:
  1. Chunks are stored in the `knowledge_chunks` PostgreSQL table (migration e3a1c9f7b2d4).
  2. Embeddings (vector(768)) are computed via Ollama nomic-embed-text and persisted —
     no recomputation on restart (zero-cost recovery).
  3. Similarity search uses pgvector's HNSW index (cosine distance) for sub-millisecond
     retrieval even as the knowledge base grows to thousands of chunks.
  4. Full admin CRUD: add / update / delete chunks via REST endpoints.
  5. Falls back to in-memory TF-IDF KnowledgeStore if pgvector is unavailable.

Usage (called in api.py lifespan):
    from src.chat.knowledge.db_store import init_knowledge_db_store
    init_knowledge_db_store(engine)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    JSON,
    MetaData,
    String,
    Table,
    Text,
    delete,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine

from .policies import KNOWLEDGE_BASE, KnowledgeChunk
from src.metrics import (
    KNOWLEDGE_RETRIEVAL_TOTAL,
    KNOWLEDGE_RETRIEVAL_EMPTY,
    KNOWLEDGE_RETRIEVAL_HIT_COUNT,
    KNOWLEDGE_RETRIEVAL_HIT_RATIO,
    KNOWLEDGE_RETRIEVAL_QUALITY_TOTAL,
    KNOWLEDGE_SIMILARITY_SCORE,
)
from src.utils import get_logger

logger = get_logger("chat.knowledge.db_store")

EMBED_DIM = 768  # nomic-embed-text output dimension
_RETRIEVAL_WINDOW_SIZE = int(os.getenv("CHAT_KB_RETRIEVAL_WINDOW", "200"))
_LOW_SIMILARITY_THRESHOLD = float(os.getenv("CHAT_KB_SIMILARITY_LOW_THRESHOLD", "0.55"))
_HIGH_SIMILARITY_THRESHOLD = float(
    os.getenv("CHAT_KB_SIMILARITY_HIGH_THRESHOLD", "0.80")
)


class KnowledgeDbStore:
    """PostgreSQL + pgvector knowledge store.

    Provides the same retrieval interface as the in-memory ``KnowledgeStore``
    but persists embeddings in the DB (no recomputation on restart) and
    exposes admin CRUD methods for dynamic knowledge management.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.metadata = MetaData()
        self._table = self._define_table()
        self._embed_client: Any = None
        self._retrieval_windows: dict[str, deque[int]] = {
            "vector": deque(maxlen=_RETRIEVAL_WINDOW_SIZE),
            "fallback": deque(maxlen=_RETRIEVAL_WINDOW_SIZE),
        }
        self._seed_and_embed()

    # ── Table definition ───────────────────────────────────────────────────────

    def _define_table(self) -> Table:
        return Table(
            "knowledge_chunks",
            self.metadata,
            Column("id", Integer(), primary_key=True, autoincrement=True),
            Column("chunk_id", String(50), nullable=False),
            Column("category", String(50), nullable=False),
            Column("tags", JSON(), nullable=False),
            Column("title", String(200), nullable=False),
            Column("content", Text(), nullable=False),
            Column("priority", Integer(), nullable=False),
            Column("is_active", Boolean(), nullable=False),
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
            Column("updated_at", DateTime(timezone=True), server_default=func.now()),
            extend_existing=True,
        )

    # ── Embedding client ───────────────────────────────────────────────────────

    def _get_embed_client(self) -> Any:
        if self._embed_client is None:
            try:
                from ..ollama_client import get_embedding_client

                self._embed_client = get_embedding_client()
            except Exception as exc:
                logger.debug("Embed client unavailable: %s", exc)
        return self._embed_client

    def _embed_text(self, content: str) -> list[float] | None:
        client = self._get_embed_client()
        if client is None:
            return None
        return client.embed_sync(content)

    # ── Seeding & indexing ─────────────────────────────────────────────────────

    def _seed_and_embed(self) -> None:
        """Seed DB from policies.py if empty; embed any un-embedded chunks."""
        try:
            with self.engine.begin() as conn:
                count = (
                    conn.execute(select(func.count()).select_from(self._table)).scalar()
                    or 0
                )

                if count == 0:
                    logger.info(
                        "Knowledge DB empty — seeding %d chunks from policies.py",
                        len(KNOWLEDGE_BASE),
                    )
                    for chunk in KNOWLEDGE_BASE:
                        conn.execute(
                            insert(self._table).values(
                                chunk_id=chunk.chunk_id,
                                category=chunk.category,
                                tags=chunk.tags,
                                title=chunk.title,
                                content=chunk.content,
                                priority=chunk.priority,
                                is_active=True,
                            )
                        )
                else:
                    # Sync new chunks from policies.py not yet in DB
                    existing = {
                        r[0]
                        for r in conn.execute(select(self._table.c.chunk_id)).fetchall()
                    }
                    new_chunks = [
                        c for c in KNOWLEDGE_BASE if c.chunk_id not in existing
                    ]
                    for chunk in new_chunks:
                        conn.execute(
                            insert(self._table).values(
                                chunk_id=chunk.chunk_id,
                                category=chunk.category,
                                tags=chunk.tags,
                                title=chunk.title,
                                content=chunk.content,
                                priority=chunk.priority,
                                is_active=True,
                            )
                        )
                        logger.info(
                            "Knowledge DB: synced new chunk '%s'", chunk.chunk_id
                        )

            # Embed chunks that are missing embeddings
            self._embed_missing()

        except Exception as exc:
            logger.warning("Knowledge DB seed/embed error: %s", exc)

    def _embed_missing(self) -> None:
        """Compute and persist embeddings for all active chunks without one."""
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT id, title, content FROM knowledge_chunks "
                        "WHERE is_active = TRUE AND embedding IS NULL"
                    )
                ).fetchall()

            if not rows:
                logger.info("Knowledge DB: all chunks already embedded ✓")
                return

            logger.info("Knowledge DB: embedding %d chunk(s) via Ollama…", len(rows))
            client = self._get_embed_client()
            if client is None:
                logger.warning(
                    "Knowledge DB: embedding client unavailable — "
                    "chunks will be embedded on next startup when Ollama is ready."
                )
                return

            embedded = 0
            for row in rows:
                chunk_id, title, content = row[0], row[1], row[2]
                text_to_embed = f"{title}. {content}"
                emb = client.embed_sync(text_to_embed)
                if emb is None:
                    continue
                with self.engine.begin() as conn:
                    conn.execute(
                        text(
                            "UPDATE knowledge_chunks "
                            "SET embedding = CAST(:emb AS vector), updated_at = NOW() "
                            "WHERE id = :id"
                        ),
                        {"emb": str(emb), "id": chunk_id},
                    )
                embedded += 1

            logger.info("Knowledge DB: %d/%d chunks embedded ✓", embedded, len(rows))

        except Exception as exc:
            logger.warning("Knowledge DB embed_missing error: %s", exc)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve_by_text(self, *, query: str, top_k: int = 3) -> list[KnowledgeChunk]:
        """Cosine similarity search via pgvector HNSW index.

        Falls back to priority-ordered chunks if embedding unavailable.
        Records Prometheus metrics: retrieval count, hit count, empty rate.
        """
        client = self._get_embed_client()
        if client is not None:
            try:
                emb = client.embed_sync(query)
                if emb is not None:
                    results, similarities = self._pgvector_search(emb, top_k)
                    method = "vector"
                    KNOWLEDGE_RETRIEVAL_TOTAL.labels(method=method).inc()
                    KNOWLEDGE_RETRIEVAL_HIT_COUNT.labels(method=method).observe(
                        len(results)
                    )
                    if not results:
                        KNOWLEDGE_RETRIEVAL_EMPTY.labels(method=method).inc()
                    self._record_retrieval_observability(
                        method=method,
                        query=query,
                        result_count=len(results),
                        similarities=similarities,
                    )
                    return results
            except Exception as exc:
                logger.debug("pgvector search failed, falling back: %s", exc)
        results = self._priority_fallback(top_k)
        method = "fallback"
        KNOWLEDGE_RETRIEVAL_TOTAL.labels(method=method).inc()
        KNOWLEDGE_RETRIEVAL_HIT_COUNT.labels(method=method).observe(len(results))
        if not results:
            KNOWLEDGE_RETRIEVAL_EMPTY.labels(method=method).inc()
        self._record_retrieval_observability(
            method=method,
            query=query,
            result_count=len(results),
            similarities=[],
        )
        return results

    async def retrieve_by_text_async(
        self, *, query: str, top_k: int = 3
    ) -> list[KnowledgeChunk]:
        """Async wrapper — runs embedding + search in a thread executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.retrieve_by_text(query=query, top_k=top_k)
        )

    def retrieve_by_customer(
        self, *, customer_data: dict, risk_score: float, top_k: int = 3
    ) -> list[KnowledgeChunk]:
        """Build a semantic query from customer features and do vector search."""
        parts: list[str] = []
        if risk_score >= 0.65:
            parts.append("yüksek risk iptal acil önlem depozito")
        elif risk_score >= 0.35:
            parts.append("orta risk iptal hatırlatma teyit")
        else:
            parts.append("düşük risk upsell ek hizmet gelir")

        dep = str(customer_data.get("deposit_type", ""))
        if dep == "No Deposit":
            parts.append("depozitosuz rezervasyon finansal bağlılık")
        elif dep == "Non Refund":
            parts.append("iade edilmez depozito koruma")

        seg = str(customer_data.get("market_segment", ""))
        if "Online" in seg:
            parts.append("online acenta OTA kanal karşılaştırma")
        elif "Corporate" in seg:
            parts.append("kurumsal şirket seyahat politikası")

        if int(customer_data.get("lead_time", 0) or 0) > 180:
            parts.append("uzun lead time erken rezervasyon plan değişikliği")
        if int(customer_data.get("previous_cancellations", 0) or 0) > 0:
            parts.append("geçmiş iptal tekrar eden müşteri kişisel temas")

        query = " ".join(parts) if parts else "iptal risk otel müşteri"
        return self.retrieve_by_text(query=query, top_k=top_k)

    def retrieve(self, *, tags: list[str], top_k: int = 3) -> list[KnowledgeChunk]:
        """Tag-based retrieval — wraps vector search for backward compatibility."""
        return self.retrieve_by_text(query=" ".join(tags), top_k=top_k)

    def _pgvector_search(
        self, emb: list[float], top_k: int
    ) -> tuple[list[KnowledgeChunk], list[float]]:
        """Execute pgvector cosine similarity search via HNSW index."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT chunk_id, category, tags, title, content, priority, "
                    "       1 - (embedding <=> CAST(:emb AS vector)) AS similarity "
                    "FROM knowledge_chunks "
                    "WHERE is_active = TRUE AND embedding IS NOT NULL "
                    "ORDER BY embedding <=> CAST(:emb AS vector) "
                    "LIMIT :k"
                ),
                {"emb": str(emb), "k": top_k},
            ).fetchall()

        similarities: list[float] = []
        for row in rows:
            if row[6] is not None:
                similarities.append(float(row[6]))

        chunks = [
            KnowledgeChunk(
                chunk_id=r[0],
                category=r[1],
                tags=r[2] or [],
                title=r[3],
                content=r[4],
                priority=r[5],
            )
            for r in rows
        ]
        return chunks, similarities

    def _record_retrieval_observability(
        self,
        *,
        method: str,
        query: str,
        result_count: int,
        similarities: list[float],
    ) -> None:
        """Emit metrics/logs for retrieval quality and threshold tuning."""
        if similarities:
            for score in similarities:
                KNOWLEDGE_SIMILARITY_SCORE.observe(score)

        hit_window = self._retrieval_windows.setdefault(
            method, deque(maxlen=_RETRIEVAL_WINDOW_SIZE)
        )
        hit_window.append(1 if result_count > 0 else 0)
        hit_ratio = sum(hit_window) / len(hit_window) if hit_window else 0.0
        KNOWLEDGE_RETRIEVAL_HIT_RATIO.labels(method=method).set(hit_ratio)

        if not similarities:
            quality_bucket = "no_similarity"
            top_similarity = 0.0
            mean_similarity = 0.0
        else:
            top_similarity = similarities[0]
            mean_similarity = sum(similarities) / len(similarities)
            if top_similarity < _LOW_SIMILARITY_THRESHOLD:
                quality_bucket = "low"
            elif top_similarity >= _HIGH_SIMILARITY_THRESHOLD:
                quality_bucket = "high"
            else:
                quality_bucket = "medium"

        KNOWLEDGE_RETRIEVAL_QUALITY_TOTAL.labels(
            method=method, bucket=quality_bucket
        ).inc()

        # Log query fingerprints (not raw text) to protect payload content.
        query_hash = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
        logger.info(
            "knowledge_retrieval_observed %s",
            json.dumps(
                {
                    "method": method,
                    "query_hash": query_hash,
                    "result_count": result_count,
                    "top_similarity": round(top_similarity, 6),
                    "mean_similarity": round(mean_similarity, 6),
                    "quality_bucket": quality_bucket,
                    "rolling_hit_ratio": round(hit_ratio, 6),
                    "low_threshold": _LOW_SIMILARITY_THRESHOLD,
                    "high_threshold": _HIGH_SIMILARITY_THRESHOLD,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
        )

    def _priority_fallback(self, top_k: int) -> list[KnowledgeChunk]:
        """Return highest-priority active chunks when vector search is unavailable."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(self._table)
                .where(self._table.c.is_active == True)  # noqa: E712
                .order_by(self._table.c.priority)
                .limit(top_k)
            ).fetchall()
        return [
            KnowledgeChunk(
                chunk_id=r.chunk_id,
                category=r.category,
                tags=r.tags or [],
                title=r.title,
                content=r.content,
                priority=r.priority,
            )
            for r in rows
        ]

    def evaluate_retrieval_dataset(
        self,
        *,
        dataset_path: str,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Run offline retrieval evaluation from JSON/JSONL samples.

        Supported sample format:
        - {"query": "...", "expected_chunk_ids": ["id1", "id2"]}
        - {"query": "...", "expected_chunk_id": "id1"}
        """
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Retrieval dataset not found: {dataset_path}")

        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return {
                "dataset_path": dataset_path,
                "sample_count": 0,
                "top_k": top_k,
                "hit_rate_at_k": 0.0,
                "mrr_at_k": 0.0,
            }

        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                records = parsed
            else:
                records = list(parsed.get("samples", []))

        total = 0
        hits = 0
        reciprocal_rank_sum = 0.0

        for item in records:
            if not isinstance(item, dict):
                continue
            query = str(item.get("query") or "").strip()
            if not query:
                continue

            expected_ids: set[str] = set()
            expected_many = item.get("expected_chunk_ids")
            if isinstance(expected_many, list):
                expected_ids.update(str(v) for v in expected_many if str(v).strip())
            expected_one = item.get("expected_chunk_id")
            if expected_one:
                expected_ids.add(str(expected_one))
            if not expected_ids:
                continue

            total += 1
            retrieved = self.retrieve_by_text(query=query, top_k=top_k)
            retrieved_ids = [chunk.chunk_id for chunk in retrieved]

            first_hit_rank = next(
                (
                    idx
                    for idx, chunk_id in enumerate(retrieved_ids, start=1)
                    if chunk_id in expected_ids
                ),
                None,
            )
            if first_hit_rank is not None:
                hits += 1
                reciprocal_rank_sum += 1.0 / float(first_hit_rank)

        hit_rate = (hits / total) if total else 0.0
        mrr_at_k = (reciprocal_rank_sum / total) if total else 0.0
        return {
            "dataset_path": dataset_path,
            "sample_count": total,
            "top_k": top_k,
            "hit_rate_at_k": round(hit_rate, 6),
            "mrr_at_k": round(mrr_at_k, 6),
            "evaluated_at": datetime.now(tz=timezone.utc).isoformat(),
            "thresholds": {
                "low_similarity": _LOW_SIMILARITY_THRESHOLD,
                "high_similarity": _HIGH_SIMILARITY_THRESHOLD,
            },
        }

    # ── Admin CRUD ─────────────────────────────────────────────────────────────

    def list_chunks(self, *, include_inactive: bool = False) -> list[dict]:
        """List all knowledge chunks with metadata (no embedding bytes)."""
        with self.engine.connect() as conn:
            stmt = select(
                self._table.c.id,
                self._table.c.chunk_id,
                self._table.c.category,
                self._table.c.tags,
                self._table.c.title,
                self._table.c.content,
                self._table.c.priority,
                self._table.c.is_active,
                self._table.c.created_at,
                self._table.c.updated_at,
                text("(embedding IS NOT NULL) AS has_embedding"),
            ).order_by(self._table.c.priority, self._table.c.id)
            if not include_inactive:
                stmt = stmt.where(self._table.c.is_active == True)  # noqa: E712
            rows = conn.execute(stmt).fetchall()
        return [dict(r._mapping) for r in rows]

    def create_chunk(
        self,
        *,
        chunk_id: str,
        category: str,
        tags: list[str],
        title: str,
        content: str,
        priority: int = 5,
    ) -> dict:
        """Create a new knowledge chunk and immediately embed it."""
        with self.engine.begin() as conn:
            result = conn.execute(
                insert(self._table)
                .values(
                    chunk_id=chunk_id,
                    category=category,
                    tags=tags,
                    title=title,
                    content=content,
                    priority=priority,
                    is_active=True,
                )
                .returning(self._table.c.id)
            )
            new_id = result.scalar()

        # Embed the new chunk
        embedded = False
        try:
            client = self._get_embed_client()
            if client:
                emb = client.embed_sync(f"{title}. {content}")
                if emb:
                    with self.engine.begin() as conn:
                        conn.execute(
                            text(
                                "UPDATE knowledge_chunks "
                                "SET embedding = CAST(:emb AS vector) "
                                "WHERE id = :id"
                            ),
                            {"emb": str(emb), "id": new_id},
                        )
                    embedded = True
        except Exception as exc:
            logger.warning("Failed to embed new chunk '%s': %s", chunk_id, exc)

        return {
            "id": new_id,
            "chunk_id": chunk_id,
            "embedded": embedded,
            "message": "Chunk oluşturuldu"
            + (" ve gömüldü." if embedded else " (embedding bekliyor)."),
        }

    def update_chunk(self, *, chunk_id: str, **fields: Any) -> bool:
        """Update chunk fields. Re-embeds automatically if title or content changed."""
        allowed = {"category", "tags", "title", "content", "priority", "is_active"}
        vals = {k: v for k, v in fields.items() if k in allowed}
        if not vals:
            return False

        with self.engine.begin() as conn:
            res = conn.execute(
                update(self._table)
                .where(self._table.c.chunk_id == chunk_id)
                .values(**vals, updated_at=func.now())
                .returning(self._table.c.id)
            )
            row = res.fetchone()

        if row is None:
            return False

        # Re-embed if text content changed
        if "title" in fields or "content" in fields:
            try:
                with self.engine.connect() as conn:
                    r = conn.execute(
                        select(self._table.c.title, self._table.c.content).where(
                            self._table.c.chunk_id == chunk_id
                        )
                    ).fetchone()
                if r:
                    client = self._get_embed_client()
                    if client:
                        emb = client.embed_sync(f"{r[0]}. {r[1]}")
                        if emb:
                            with self.engine.begin() as conn:
                                conn.execute(
                                    text(
                                        "UPDATE knowledge_chunks "
                                        "SET embedding = CAST(:emb AS vector), updated_at = NOW() "
                                        "WHERE chunk_id = :cid"
                                    ),
                                    {"emb": str(emb), "cid": chunk_id},
                                )
            except Exception as exc:
                logger.warning("Re-embed failed for '%s': %s", chunk_id, exc)

        return True

    def delete_chunk(self, *, chunk_id: str, hard_delete: bool = False) -> bool:
        """Soft-delete (deactivate) or hard-delete a chunk."""
        with self.engine.begin() as conn:
            if hard_delete:
                res = conn.execute(
                    delete(self._table).where(self._table.c.chunk_id == chunk_id)
                )
            else:
                res = conn.execute(
                    update(self._table)
                    .where(self._table.c.chunk_id == chunk_id)
                    .values(is_active=False, updated_at=func.now())
                )
        return (res.rowcount or 0) > 0

    def rebuild_embeddings(self) -> dict:
        """Force re-embed all active chunks (useful after model change)."""
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        "UPDATE knowledge_chunks SET embedding = NULL "
                        "WHERE is_active = TRUE"
                    )
                )
            self._embed_missing()
            with self.engine.connect() as conn:
                embedded_count = conn.execute(
                    text(
                        "SELECT COUNT(*) FROM knowledge_chunks "
                        "WHERE embedding IS NOT NULL"
                    )
                ).scalar()
            return {
                "embedded": embedded_count,
                "message": f"{embedded_count} chunk yeniden gömüldü.",
            }
        except Exception as exc:
            return {"error": str(exc)}


# ── Singleton factory ──────────────────────────────────────────────────────────

_db_store: KnowledgeDbStore | None = None


def init_knowledge_db_store(engine: Engine) -> KnowledgeDbStore:
    """Initialize the pgvector knowledge store. Call once in app lifespan."""
    global _db_store
    _db_store = KnowledgeDbStore(engine)
    logger.info("KnowledgeDbStore initialized (pgvector, table=knowledge_chunks)")
    return _db_store


def get_knowledge_db_store() -> KnowledgeDbStore | None:
    """Return the initialized DB store, or None if not yet initialized."""
    return _db_store
