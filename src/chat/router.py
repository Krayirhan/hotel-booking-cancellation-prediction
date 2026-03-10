from __future__ import annotations

import base64
import binascii
import datetime
import hashlib
import hmac
import io
import json
import logging
import os
import re
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError

from .ollama_client import get_ollama_client
from .orchestrator import get_orchestrator

logger = logging.getLogger(__name__)

router_chat = APIRouter(prefix="/chat", tags=["chat"])


class StartSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_data: dict[str, Any] = Field(default_factory=dict)
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_label: str = "unknown"


class StartSessionResponse(BaseModel):
    session_id: str
    bot_message: str
    quick_actions: list[dict[str, str]]
    risk_score: float
    risk_label: str


class ChatMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    message: str = Field(min_length=1, max_length=2000)


class ChatMessageResponse(BaseModel):
    session_id: str
    bot_message: str
    quick_actions: list[dict[str, str]]


class SessionSummaryResponse(BaseModel):
    session_id: str
    risk_score: float
    risk_label: str
    message_count: int
    created_at: float
    last_active: float


@router_chat.get("/health")
async def chat_health() -> dict[str, str]:
    client = get_ollama_client()
    alive = await client.health()
    return {
        "status": "ok" if alive else "degraded",
        "ollama": "ok" if alive else "unreachable",
        "model": client.model,
    }


@router_chat.post("/session", response_model=StartSessionResponse)
async def start_session(body: StartSessionRequest) -> StartSessionResponse:
    orchestrator = get_orchestrator()
    try:
        session_id, bot_message = await orchestrator.start_session(
            customer_data=body.customer_data,
            risk_score=body.risk_score,
            risk_label=body.risk_label,
        )
        actions = await orchestrator.quick_actions(session_id=session_id)
        return StartSessionResponse(
            session_id=session_id,
            bot_message=bot_message,
            quick_actions=actions,
            risk_score=body.risk_score,
            risk_label=body.risk_label,
        )
    except Exception as exc:
        if isinstance(exc, SQLAlchemyError):
            raise HTTPException(
                status_code=503,
                detail="Chat knowledge/session store is unavailable. Check DB migrations.",
            ) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router_chat.post("/message", response_model=ChatMessageResponse)
async def message(body: ChatMessageRequest) -> ChatMessageResponse:
    orchestrator = get_orchestrator()
    try:
        bot_message = await orchestrator.send_message(
            session_id=body.session_id,
            user_message=body.message,
        )
        actions = await orchestrator.quick_actions(session_id=body.session_id)
        return ChatMessageResponse(
            session_id=body.session_id,
            bot_message=bot_message,
            quick_actions=actions,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        if isinstance(exc, SQLAlchemyError):
            raise HTTPException(
                status_code=503,
                detail="Chat knowledge/session store is unavailable. Check DB migrations.",
            ) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router_chat.post("/message/stream")
async def message_stream(body: ChatMessageRequest) -> StreamingResponse:
    """SSE streaming endpoint — streams LLM tokens as Server-Sent Events.

    Each event is a JSON object:
    - ``{"token": "..."}`` — partial content token
    - ``{"done": true, "quick_actions": [...]}`` — stream complete
    - ``{"error": "..."}`` — error occurred
    """
    orchestrator = get_orchestrator()

    async def event_stream():
        try:
            async for token in orchestrator.stream_message(
                session_id=body.session_id,
                user_message=body.message,
            ):
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            actions = await orchestrator.quick_actions(session_id=body.session_id)
            yield f"data: {json.dumps({'done': True, 'quick_actions': actions}, ensure_ascii=False)}\n\n"
        except ValueError as exc:
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.error("SSE stream error: %s", exc)
            yield f"data: {json.dumps({'error': 'Sunucu hatası oluştu'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router_chat.get("/session/{session_id}/summary", response_model=SessionSummaryResponse)
async def summary(session_id: str) -> SessionSummaryResponse:
    orchestrator = get_orchestrator()
    try:
        payload = await orchestrator.summary(session_id=session_id)
        return SessionSummaryResponse(**payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ── Risk skoru otomatik hesaplama ─────────────────────────────────────────────


class PredictRiskRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hotel: str = "City Hotel"
    lead_time: int = 0
    deposit_type: str = "No Deposit"
    market_segment: str = "Online TA"
    adults: int = 2
    children: int = 0
    babies: int = 0
    stays_in_week_nights: int = 0
    stays_in_weekend_nights: int = 1
    previous_cancellations: int = 0
    is_repeated_guest: int = 0
    adr: float = 100.0
    model_name: str | None = None  # None → aktif şampiyon model kullanılır


class PredictRiskResponse(BaseModel):
    risk_score: float
    risk_label: str
    risk_percent: float
    model_used: str = "champion"


@router_chat.get("/models")
async def list_available_models(request: Request) -> dict:
    """Mevcut run'daki tüm modelleri döner (model seçimi için)."""
    from pathlib import Path

    root = Path(".")
    latest_path = root / "models" / "latest.json"
    if not latest_path.exists():
        return {"models": [], "active_model": None, "run_id": None}

    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    run_id = latest.get("run_id", "")

    registry_path = root / "reports" / "metrics" / run_id / "model_registry.json"
    if not registry_path.exists():
        return {"models": [], "active_model": None, "run_id": run_id}

    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    serving = getattr(request.app.state, "serving", None)
    active_model = None
    if serving and serving.policy:
        active_model = serving.policy.selected_model

    models = [
        {"name": name, "artifact": artifact, "is_active": name == active_model}
        for name, artifact in registry.items()
    ]
    return {"run_id": run_id, "models": models, "active_model": active_model}


@router_chat.post("/predict-risk", response_model=PredictRiskResponse)
async def predict_risk(
    body: PredictRiskRequest, request: Request
) -> PredictRiskResponse:
    """Form verisinden model tahminini çalıştırarak iptal riski üretir."""
    import joblib as _joblib
    import pandas as pd
    from pathlib import Path
    from ..predict import validate_and_prepare_features

    serving = getattr(request.app.state, "serving", None)
    if serving is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")

    # ── Model seçimi ──────────────────────────────────────────────────────────
    model_to_use = serving.model
    model_name_used = getattr(serving.policy, "selected_model", "champion")

    if body.model_name:
        try:
            root = Path(".")
            latest_path = root / "models" / "latest.json"
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            run_id = latest.get("run_id", "")
            registry_path = (
                root / "reports" / "metrics" / run_id / "model_registry.json"
            )
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            if body.model_name not in registry:
                raise HTTPException(
                    status_code=400, detail=f"Model bulunamadı: {body.model_name}"
                )
            artifact = registry[body.model_name]
            model_path = root / artifact
            if not model_path.exists():
                raise HTTPException(
                    status_code=404, detail=f"Model dosyası bulunamadı: {artifact}"
                )
            model_to_use = _joblib.load(model_path)
            model_name_used = body.model_name
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Model yükleme hatası: {exc}"
            ) from exc

    # Varış tarihini lead_time'dan hesapla
    arrival = datetime.date.today() + datetime.timedelta(days=int(body.lead_time))

    record: dict[str, Any] = {
        # Form alanları
        "hotel": body.hotel,
        "lead_time": body.lead_time,
        "deposit_type": body.deposit_type,
        "market_segment": body.market_segment,
        "adults": body.adults,
        "children": body.children,
        "babies": body.babies,
        "stays_in_week_nights": body.stays_in_week_nights,
        "stays_in_weekend_nights": body.stays_in_weekend_nights,
        "previous_cancellations": body.previous_cancellations,
        "is_repeated_guest": body.is_repeated_guest,
        # Tarih alanları
        "arrival_date_year": arrival.year,
        "arrival_date_month": arrival.strftime("%B"),
        "arrival_date_week_number": arrival.isocalendar()[1],
        "arrival_date_day_of_month": arrival.day,
        # Varsayılan nümerik alanlar
        "previous_bookings_not_canceled": 0,
        "booking_changes": 0,
        "agent": 0,
        "company": 0,
        "days_in_waiting_list": 0,
        "adr": body.adr,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0,
        # Varsayılan kategorik alanlar
        "meal": "BB",
        "country": "PRT",
        "distribution_channel": "TA/TO",
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "customer_type": "Transient",
    }

    try:
        df = pd.DataFrame([record])
        X, _ = validate_and_prepare_features(
            df, serving.feature_spec, fail_on_missing=False
        )
        proba = float(model_to_use.predict_proba(X)[0, 1])
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Model tahmin hatası: {exc}"
        ) from exc

    if proba >= 0.65:
        label = "high"
    elif proba >= 0.35:
        label = "medium"
    else:
        label = "low"

    return PredictRiskResponse(
        risk_score=round(proba, 4),
        risk_label=label,
        risk_percent=round(proba * 100, 1),
        model_used=model_name_used,
    )


# ── Admin: Knowledge Base Management ─────────────────────────────────────────
# Endpoints for dynamic management of the RAG knowledge base.
# Requires pgvector KnowledgeDbStore to be initialized (api.py lifespan).


class KnowledgeChunkCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=3, max_length=50)
    category: str = Field(default="general", max_length=50)
    tags: list[str] = Field(default_factory=list)
    title: str = Field(min_length=3, max_length=200)
    content: str = Field(min_length=10)
    priority: int = Field(default=5, ge=1, le=10)


class KnowledgeChunkUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str | None = None
    tags: list[str] | None = None
    title: str | None = None
    content: str | None = None
    priority: int | None = Field(default=None, ge=1, le=10)
    is_active: bool | None = None


class KnowledgeIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str = Field(default="manual-entry", min_length=3, max_length=80)
    source_type: str = Field(default="text", pattern="^(text|pdf)$")
    content: str | None = Field(default=None)
    content_base64: str | None = Field(default=None)
    category: str = Field(default="general", max_length=50)
    tags: list[str] = Field(default_factory=list)
    priority: int = Field(default=5, ge=1, le=10)
    chunk_size: int = Field(default=900, ge=200, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)


def _require_db_store():
    from .knowledge.db_store import get_knowledge_db_store

    db = get_knowledge_db_store()
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="pgvector knowledge store henüz başlatılmadı. DATABASE_URL ayarını kontrol edin.",
        )
    return db


def _require_admin_key(request: Request) -> None:
    expected_admin = os.getenv("DS_ADMIN_KEY")
    if expected_admin and not hmac.compare_digest(
        request.headers.get("x-admin-key") or "", expected_admin
    ):
        raise HTTPException(status_code=403, detail="x-admin-key header gereklidir.")


def _chunk_text(content: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = re.sub(r"\r\n?", "\n", content).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    if not normalized:
        return []

    step = max(1, chunk_size - chunk_overlap)
    chunks: list[str] = []
    start = 0
    text_len = len(normalized)

    while start < text_len:
        end = min(text_len, start + chunk_size)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start += step

    return chunks


def _extract_pdf_text(content_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "PDF parse icin pypdf gerekli. pypdf yoksa source_type='text' "
                "ile cikarilmis metni gonderin."
            ),
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(content_bytes))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        text = "\n\n".join(p for p in pages if p)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"PDF parse hatasi: {exc}") from exc

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="PDF iceriginden metin cikarilamadi.",
        )
    return text


def _make_chunk_id(source_name: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", source_name.lower()).strip("-")[:24] or "kb"
    salt = hashlib.sha1(uuid.uuid4().hex.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{index:03d}-{salt}"


@router_chat.post("/knowledge", status_code=201, summary="Knowledge ingest endpoint")
async def ingest_knowledge(body: KnowledgeIngestRequest, request: Request) -> dict:
    """Ingest text/PDF content, chunk it, embed, and persist to pgvector store."""
    _require_admin_key(request)
    db = _require_db_store()

    if body.chunk_overlap >= body.chunk_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap, chunk_size degerinden kucuk olmalidir.",
        )

    if body.source_type == "pdf":
        if not body.content_base64:
            raise HTTPException(
                status_code=400,
                detail="PDF ingest icin content_base64 zorunludur.",
            )
        try:
            raw_pdf = base64.b64decode(body.content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail="content_base64 gecersiz.") from exc
        source_text = _extract_pdf_text(raw_pdf)
    else:
        source_text = (body.content or "").strip()

    if len(source_text) < 20:
        raise HTTPException(
            status_code=400,
            detail="Ingest edilecek metin en az 20 karakter olmalidir.",
        )

    chunks = _chunk_text(
        source_text,
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="Chunk olusturulamadi.")

    created_ids: list[str] = []
    failures: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        chunk_id = _make_chunk_id(body.source_name, idx)
        title = f"{body.source_name} [{idx}/{len(chunks)}]"
        try:
            db.create_chunk(
                chunk_id=chunk_id,
                category=body.category,
                tags=body.tags,
                title=title,
                content=chunk,
                priority=body.priority,
            )
            created_ids.append(chunk_id)
        except Exception as exc:
            failures.append(str(exc))

    if not created_ids:
        detail = failures[0] if failures else "Knowledge chunk kaydi basarisiz."
        raise HTTPException(status_code=400, detail=detail)

    return {
        "source_name": body.source_name,
        "source_type": body.source_type,
        "chunks_created": len(created_ids),
        "chunks_failed": len(failures),
        "chunk_ids": created_ids[:25],
    }


@router_chat.get("/knowledge/stats", summary="Knowledge base stats")
async def knowledge_stats(request: Request) -> dict:
    _require_admin_key(request)
    rows = _require_db_store().list_chunks(include_inactive=True)
    total = len(rows)
    active = sum(1 for row in rows if bool(row.get("is_active")))
    embedded = sum(1 for row in rows if bool(row.get("has_embedding")))

    categories: dict[str, int] = {}
    for row in rows:
        category = str(row.get("category") or "uncategorized")
        categories[category] = categories.get(category, 0) + 1

    return {
        "total_chunks": total,
        "active_chunks": active,
        "inactive_chunks": total - active,
        "embedded_chunks": embedded,
        "categories": categories,
    }


@router_chat.get("/admin/knowledge", summary="Tüm knowledge chunk'larını listele")
async def list_knowledge_chunks(
    request: Request, include_inactive: bool = False
) -> list[dict]:
    """Knowledge tabanındaki tüm chunk'ları döner (admin)."""
    _require_admin_key(request)
    return _require_db_store().list_chunks(include_inactive=include_inactive)


@router_chat.post(
    "/admin/knowledge", status_code=201, summary="Yeni knowledge chunk ekle"
)
async def create_knowledge_chunk(
    body: KnowledgeChunkCreateRequest, request: Request
) -> dict:
    """Yeni bir knowledge chunk oluşturur ve otomatik olarak embed eder."""
    _require_admin_key(request)
    db = _require_db_store()
    try:
        return db.create_chunk(
            chunk_id=body.chunk_id,
            category=body.category,
            tags=body.tags,
            title=body.title,
            content=body.content,
            priority=body.priority,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router_chat.put("/admin/knowledge/{chunk_id}", summary="Knowledge chunk güncelle")
async def update_knowledge_chunk(
    chunk_id: str, body: KnowledgeChunkUpdateRequest, request: Request
) -> dict:
    """Bir chunk'ı günceller. İçerik değişirse otomatik re-embed yapar."""
    _require_admin_key(request)
    db = _require_db_store()
    updates = body.model_dump(exclude_none=True)
    ok = db.update_chunk(chunk_id=chunk_id, **updates)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Chunk bulunamadı: {chunk_id}")
    return {"chunk_id": chunk_id, "message": "Güncellendi."}


@router_chat.delete("/admin/knowledge/{chunk_id}", summary="Knowledge chunk sil")
async def delete_knowledge_chunk(
    chunk_id: str, request: Request, hard: bool = False
) -> dict:
    """Chunk'ı devre dışı bırakır (soft delete) veya kalıcı siler (hard=true)."""
    _require_admin_key(request)
    db = _require_db_store()
    ok = db.delete_chunk(chunk_id=chunk_id, hard_delete=hard)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Chunk bulunamadı: {chunk_id}")
    action = "Kalıcı silindi." if hard else "Devre dışı bırakıldı."
    return {"chunk_id": chunk_id, "message": action}


@router_chat.post(
    "/admin/knowledge/rebuild-embeddings",
    summary="Tüm chunk embedding'lerini yeniden oluştur",
)
async def rebuild_knowledge_embeddings(request: Request) -> dict:
    """Tüm aktif chunk'ları sıfırdan embed eder. Model değişikliği sonrası kullanın."""
    _require_admin_key(request)
    db = _require_db_store()
    loop = __import__("asyncio").get_running_loop()
    result = await loop.run_in_executor(None, db.rebuild_embeddings)
    return result
