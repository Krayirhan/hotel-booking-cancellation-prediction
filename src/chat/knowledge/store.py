from __future__ import annotations

import asyncio
from typing import Any

from .policies import KNOWLEDGE_BASE, KnowledgeChunk
from ...utils import get_logger

logger = get_logger("chat.knowledge.store")


class KnowledgeStore:
    """Retrieves policy chunks via tag overlap and TF-IDF cosine similarity (#27).

    At startup the store builds a TF-IDF index over chunk title + content + tags.
    ``retrieve_by_text`` uses cosine similarity for semantic-ish matching.
    ``retrieve`` falls back to tag-overlap scoring when no query text is given.
    """

    def __init__(self) -> None:
        self._chunks = KNOWLEDGE_BASE
        self._vectorizer: Any = None
        self._tfidf_matrix: Any = None
        self._embeddings: list[list[float]] | None = None
        self._embed_client: Any = None  # OllamaEmbeddingClient when available
        self._build_tfidf_index()
        self._try_build_embedding_index()  # best-effort; falls back to TF-IDF

    # ── TF-IDF index ──────────────────────────────────────────────────────────

    def _build_tfidf_index(self) -> None:
        """Build TF-IDF index over chunk text. Silently falls back if sklearn absent."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]

            texts = [f"{c.title} {c.content} {' '.join(c.tags)}" for c in self._chunks]
            self._vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            )
            self._tfidf_matrix = self._vectorizer.fit_transform(texts)
            logger.debug(
                "Knowledge store: TF-IDF index built (%d chunks)", len(self._chunks)
            )
        except Exception as exc:
            logger.warning(
                "Knowledge store: TF-IDF index unavailable, using tag-only retrieval. reason=%s",
                exc,
            )
            self._vectorizer = None
            self._tfidf_matrix = None

    # ── Retrieval methods ──────────────────────────────────────────────────────

    def retrieve_by_text(self, *, query: str, top_k: int = 3) -> list[KnowledgeChunk]:
        """Semantic retrieval via TF-IDF cosine similarity (#27).

        Falls back to priority-ordered chunks when the TF-IDF index is unavailable.
        """
        if self._vectorizer is None or self._tfidf_matrix is None:
            return sorted(self._chunks, key=lambda c: c.priority)[:top_k]

        try:
            from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import]
            import numpy as np  # type: ignore[import]

            query_vec = self._vectorizer.transform([query])
            sims = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
            top_indices = np.argsort(sims)[::-1][:top_k]
            return [self._chunks[int(i)] for i in top_indices if sims[int(i)] > 0]
        except Exception:
            return sorted(self._chunks, key=lambda c: c.priority)[:top_k]

    def retrieve(self, *, tags: list[str], top_k: int = 3) -> list[KnowledgeChunk]:
        """Tag-overlap scoring (used when no free-text query is available)."""
        query_tags = {t.lower() for t in tags}
        scored: list[tuple[float, KnowledgeChunk]] = []
        for chunk in self._chunks:
            chunk_tags = {t.lower() for t in chunk.tags}
            overlap = len(query_tags & chunk_tags)
            if overlap == 0:
                continue
            score = overlap * (6 - max(1, min(5, chunk.priority)))
            scored.append((float(score), chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def retrieve_by_customer(
        self, *, customer_data: dict, risk_score: float, top_k: int = 3
    ) -> list[KnowledgeChunk]:
        """Hybrid: TF-IDF query built from customer features, with tag-based fallback."""
        tags = self._extract_tags(customer_data=customer_data, risk_score=risk_score)

        if self._vectorizer is not None:
            # Build a short query from the customer's key features
            query_parts = list(tags)
            lead = customer_data.get("lead_time", 0)
            segment = customer_data.get("market_segment", "")
            deposit = customer_data.get("deposit_type", "")
            if lead:
                query_parts.append(f"lead time {lead}")
            if segment:
                query_parts.append(segment.lower())
            if deposit:
                query_parts.append(deposit.lower())
            return self.retrieve_by_text(query=" ".join(query_parts), top_k=top_k)

        return self.retrieve(tags=tags, top_k=top_k)

    def _extract_tags(self, *, customer_data: dict, risk_score: float) -> list[str]:
        tags: list[str] = []
        if risk_score >= 0.6:
            tags.append("yüksek_risk")
        elif risk_score < 0.35:
            tags.extend(["düşük_risk", "upsell"])

        dep = str(customer_data.get("deposit_type", ""))
        if dep == "No Deposit":
            tags.extend(["depozito", "no_deposit"])
        elif dep == "Non Refund":
            tags.extend(["depozito", "non_refund"])

        lead_time = int(customer_data.get("lead_time", 0) or 0)
        if lead_time > 180:
            tags.extend(["lead_time", "uzun", "erken_rezervasyon"])

        prev_cancel = int(customer_data.get("previous_cancellations", 0) or 0)
        if prev_cancel > 0:
            tags.extend(["geçmiş_iptal", "previous_cancellations"])

        segment = str(customer_data.get("market_segment", ""))
        if "Online" in segment:
            tags.extend(["online", "ota", "online_ta"])

        return tags

    # ── Ollama embedding index (optional — requires OLLAMA_EMBED_MODEL) ───────────

    def _try_build_embedding_index(self) -> None:
        """Pre-compute Ollama embeddings for all chunks at startup (best-effort).

        On success, ``retrieve_by_text_async`` uses vector cosine similarity
        instead of TF-IDF, giving true semantic matching across languages.
        Silently falls back to TF-IDF when the embedding model is unavailable.
        """
        try:
            from ..ollama_client import get_embedding_client

            embed_client = get_embedding_client()
            texts = [f"{c.title}. {c.content} {' '.join(c.tags)}" for c in self._chunks]
            embeddings = embed_client.embed_batch_sync(texts)
            if embeddings and len(embeddings) == len(self._chunks):
                self._embeddings = embeddings
                self._embed_client = embed_client
                logger.info(
                    "Knowledge store: embedding index ready (%d chunks, model=%s)",
                    len(self._chunks),
                    embed_client.model,
                )
            else:
                logger.debug("Knowledge store: embedding index skipped — TF-IDF only")
        except Exception as exc:
            logger.debug("Knowledge store: embedding init skipped: %s", exc)

    async def retrieve_by_text_async(
        self, *, query: str, top_k: int = 3
    ) -> list[KnowledgeChunk]:
        """Async semantic retrieval: Ollama embeddings → TF-IDF → priority order.

        When an Ollama embedding model is configured and available, uses vector
        cosine similarity for true semantic matching.  Falls back to TF-IDF
        (always pre-built) when embeddings are unavailable.
        """
        if self._embeddings is not None and self._embed_client is not None:
            try:
                loop = asyncio.get_running_loop()
                query_emb = await loop.run_in_executor(
                    None, self._embed_client.embed_sync, query
                )
                if query_emb is not None:
                    sims = [_cosine_sim(query_emb, emb) for emb in self._embeddings]
                    top_indices = sorted(
                        range(len(sims)), key=lambda i: sims[i], reverse=True
                    )[:top_k]
                    results = [self._chunks[i] for i in top_indices if sims[i] > 0.05]
                    if results:
                        return results
            except Exception as exc:
                logger.debug(
                    "Async embedding retrieval failed, falling back to TF-IDF: %s", exc
                )
        return self.retrieve_by_text(query=query, top_k=top_k)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two float vectors."""
    import math

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b + 1e-10)


_store: Any | None = None  # KnowledgeStore (TF-IDF) or KnowledgeDbStore (pgvector)


def get_knowledge_store() -> Any:
    """Return the best available knowledge store.

    Priority:
      1. pgvector KnowledgeDbStore (persistent, semantic, scalable) — if initialized
      2. In-memory KnowledgeStore (TF-IDF) — always available as fallback
    """
    global _store
    if _store is not None:
        return _store

    # Prefer pgvector DB store if it has been initialized in the app lifespan
    try:
        from .db_store import get_knowledge_db_store

        db = get_knowledge_db_store()
        if db is not None:
            from sqlalchemy import text

            # Ensure underlying schema exists; otherwise fall back to in-memory store.
            with db.engine.connect() as conn:
                conn.execute(text("SELECT 1 FROM knowledge_chunks LIMIT 1"))
            _store = db
            logger.info("Knowledge store: using pgvector KnowledgeDbStore")
            return _store
    except Exception as exc:
        logger.debug("KnowledgeDbStore unavailable: %s — using TF-IDF fallback", exc)

    _store = KnowledgeStore()
    logger.info("Knowledge store: using in-memory TF-IDF KnowledgeStore")
    return _store
