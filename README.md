# Hotel Booking Cancellation Prediction

> **Production-grade MLOps platform** — end-to-end ML pipeline with cost-sensitive decisioning, real-time API serving, RAG-powered chat, full observability, and Kubernetes-ready deployment.

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%2B-4169E1?logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-7%2B-DC382D?logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose%20v2-2496ED?logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-326CE5?logo=kubernetes&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-Pipeline-945DD6?logo=dvc&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-1.2.0-blue)

---

## Table of Contents

- [Overview](#overview)
- [Key Capabilities](#key-capabilities)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [ML Pipeline](#ml-pipeline)
- [API Reference](#api-reference)
- [Dashboard & Frontend](#dashboard--frontend)
- [Monitoring & Alerting](#monitoring--alerting)
- [Policy Management — Rollout & Rollback](#policy-management--rollout--rollback)
- [Testing](#testing)
- [Deployment](#deployment)
- [CI/CD](#cicd)
- [Performance & Load Testing](#performance--load-testing)
- [Data Lineage & Versioning](#data-lineage--versioning)
- [Environment Variables](#environment-variables)
- [Operational Docs](#operational-docs)

---

## Overview

`hotel_booking_cancellation_prediction` is a **full-stack MLOps platform** built on the [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) dataset. It goes well beyond a Jupyter notebook proof-of-concept by integrating every production concern:

| Concern | Implementation |
|---|---|
| **Reproducibility** | DVC-tracked pipeline with parameter versioning |
| **Cost-aware decisions** | Configurable cost matrix (TP/FP/FN/TN values) |
| **Online serving** | Versioned FastAPI endpoints (`/v1`, `/v2`) |
| **Calibration** | Two-stage sigmoid → isotonic calibration |
| **Observability** | Prometheus metrics, Jaeger tracing, Grafana dashboards |
| **Drift detection** | PSI + KS-based feature and prediction drift |
| **Safe deployments** | Blue/green slots, canary ingress, automated rollback |
| **RAG assistant** | Ollama-powered LLM with pgvector knowledge store |
| **Security** | API key auth, bcrypt hashed passwords, Lua rate limiting |

---

## Key Capabilities

- **End-to-end ML lifecycle** — `preprocess → split → train → calibrate → evaluate → predict`
- **Versioned REST API** — probability scoring and policy-based binary decisions
- **Cost-sensitive optimization** — configurable business cost matrix drives threshold selection
- **Two-stage model calibration** — sigmoid and isotonic regression calibration
- **RAG chat assistant** — Server-Sent Events streaming, pgvector HNSW retrieval, Ollama LLM
- **Real-time observability** — Prometheus counters/histograms, Grafana dashboards, Jaeger distributed tracing
- **Drift monitoring** — PSI/KS drift detection with webhook alerting and dead-letter queue retry
- **Policy lifecycle** — promote, rollback, and blue/green slot management via CLI
- **Canary deployments** — NGINX-weighted traffic splitting in Kubernetes
- **Data lineage** — per-run lineage artefacts automatically recorded by DVC
- **> 80% test coverage** enforced by CI

---

## Architecture

```
┌──────────────────────────── Client ──────────────────────────────────┐
│  React 18 SPA  (Vite · react-router-dom · Chart.js)                 │
│  useAuth · useRuns · useChat · useTheme · useSystem                  │
└────────────────────────┬─────────────────────────────────────────────┘
                         │ HTTP / SSE
┌────────────────────────▼─────────────────────────────────────────────┐
│  FastAPI  (src/api.py + api_lifespan.py)                             │
│                                                                      │
│  ├─ /v1 · /v2      →  ML inference + policy decision                │
│  ├─ /dashboard     →  DashboardStore (PostgreSQL)                   │
│  │                    └─ Redis cache (TTL 45 s)                      │
│  ├─ /chat          →  RAG pipeline (Ollama + pgvector HNSW)         │
│  ├─ /guests        →  CRM CRUD (PostgreSQL)                         │
│  ├─ /auth          →  Session management                            │
│  ├─ /health · /ready · /metrics  (liveness / readiness / Prometheus)│
│  └─ /reload        →  Zero-downtime model hot-reload                │
└──────┬──────────────────┬───────────────────────┬────────────────────┘
       │                  │                       │
  PostgreSQL           Redis                  Ollama
  + pgvector        (cache + sessions)       (qwen2.5:7b)
  (data / model                              (LLM + embeddings)
   artifacts)
       │
  Prometheus ──► Grafana
  Jaeger (OTLP tracing)
```

### Architecture Decision Records

Significant design decisions are documented in [`docs/adr/`](docs/adr/):

| ADR | Decision |
|---|---|
| [ADR-001](docs/adr/001-ml-pipeline-architecture.md) | ML pipeline architecture |
| [ADR-002](docs/adr/002-cost-sensitive-decisioning.md) | Cost-sensitive decisioning |
| [ADR-003](docs/adr/003-api-serving-design.md) | API serving design |
| [ADR-004](docs/adr/004-helm-gitops-deployment.md) | Helm + GitOps deployment |
| [ADR-005](docs/adr/005-distributed-tracing.md) | Distributed tracing |
| [ADR-006](docs/adr/006-data-validation-framework.md) | Data validation framework |
| [ADR-007](docs/adr/007-api-versioning.md) | API versioning strategy |
| [ADR-008](docs/adr/008-lua-rate-limiting.md) | Lua-based rate limiting |
| [ADR-009](docs/adr/009-model-health-metrics.md) | Model health metrics |
| [ADR-010](docs/adr/010-versioned-api-v1-v2.md) | Versioned API (v1 / v2) |
| [ADR-011](docs/adr/011-ollama-self-hosted.md) | Self-hosted LLM (Ollama) |
| [ADR-012](docs/adr/012-postgres-plus-redis.md) | Storage: PostgreSQL + Redis |
| [ADR-013](docs/adr/013-calibration-sigmoid-isotonic.md) | Two-stage calibration |

---

## Tech Stack

### Backend & ML

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| API framework | FastAPI + Uvicorn |
| ML / modeling | scikit-learn, pandas, numpy |
| Calibration | `CalibratedClassifierCV` — sigmoid + isotonic |
| Data validation | Pandera (schema + drift + anomaly) |
| Pipeline orchestration | DVC |
| Database ORM | SQLAlchemy + Alembic |
| Vector store | pgvector (HNSW index) |
| LLM inference | Ollama (`qwen2.5:7b`) |
| Observability | Prometheus, Grafana, Jaeger (OTLP) |
| Rate limiting | Redis Lua scripts |
| Testing | pytest, pytest-cov (≥ 80% branch coverage) |

### Frontend

| Layer | Technology |
|---|---|
| Framework | React 18 |
| Build tool | Vite |
| Routing | react-router-dom |
| Charts | Chart.js |
| Theming | CSS custom properties — light / dark mode |

### Infrastructure

| Layer | Technology |
|---|---|
| Containerisation | Docker + Docker Compose v2 |
| Orchestration | Kubernetes (manifests + Helm) |
| GitOps | Flux / Argo (deploy/gitops) |
| CI/CD | GitHub Actions |
| Load testing | Locust + k6 |

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `src/api.py` | FastAPI application factory — middleware, CORS, router registration |
| `src/api_lifespan.py` | Lifespan context — DB init, store bootstrap, startup / shutdown |
| `src/api_v1.py` / `api_v2.py` | Versioned prediction endpoints (`/v1`, `/v2`) |
| `src/dashboard.py` | Dashboard API router + Redis cache layer (TTL 45 s) |
| `src/dashboard_store.py` | SQLAlchemy `DashboardStore` — experiment / run records |
| `src/chat/router.py` | SSE streaming chat endpoint — RAG pipeline orchestration |
| `src/chat/knowledge/db_store.py` | pgvector HNSW knowledge store — Prometheus-instrumented |
| `src/data_validation.py` | Pandera schema validation, drift detection, anomaly detection |
| `src/guests.py` | Guest CRUD router |
| `src/metrics.py` | Prometheus counter/histogram definitions |
| `src/monitoring.py` | PSI / data drift monitoring |
| `src/policy.py` | Policy promotion, rollback, slot management |
| `src/calibration.py` | Two-stage model calibration |
| `src/cost_matrix.py` | Business cost matrix and threshold selection |
| `apps/frontend/src/` | React 18 SPA — dashboard pages, hooks, theming |

---

## Quick Start

### Prerequisites

| Tool | Minimum Version | Check |
|---|---|---|
| Docker Desktop | 24+ | `docker --version` |
| Docker Compose v2 | built-in | `docker compose version` |
| Python | 3.12+ | (for `check_setup.py`) |
| Git | any | `git --version` |

### 1 — Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/hotel-booking-cancellation-prediction.git
cd hotel-booking-cancellation-prediction
```

### 2 — Verify your environment

```bash
python scripts/check_setup.py
```

The script performs **10 automated checks** and tells you exactly what to fix:

| # | Check | Description |
|---|---|---|
| 1 | Python version | ≥ 3.12 required |
| 2 | Python packages | All packages from `requirements.txt` |
| 3 | Docker | Docker Engine + Compose v2 |
| 4 | Node.js / npm | ≥ v18 (frontend dev) |
| 5 | **Ollama** | CLI installed? Service running? **Model downloaded?** |
| 6 | `.env` file | Present? No `replace-me` placeholders? |
| 7 | Environment variables | All required / optional env vars |
| 8 | Service connectivity | Can reach PostgreSQL and Redis? |
| 9 | Project files | Model artefacts, raw data, frontend directory |
| 10 | Port availability | No Docker Compose port conflicts |

Exit code `0` = all good, `1` = critical issues found.

### 3 — Create your `.env` file

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Edit the four required values:

```dotenv
DS_API_KEY=<strong-random-string>

# Generate bcrypt hash: python -c "import bcrypt; print(bcrypt.hashpw(b'yourpassword', bcrypt.gensalt()).decode())"
DASHBOARD_ADMIN_PASSWORD_ADMIN=<bcrypt-hash-or-plaintext>

POSTGRES_PASSWORD=<strong-db-password>
GF_ADMIN_PASSWORD=<strong-grafana-password>
```

> ⚠️ Never commit `.env` — it is already listed in `.gitignore`.

### 4 — Train the ML model (first run only)

The `models/` directory is not tracked by Git. Train your first model:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

python main.py preprocess
python main.py train
python main.py evaluate
```

> **Dataset:** Download [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) from Kaggle and place `hotel_bookings.csv` inside `data/raw/`.

### 5 — Start the full stack with Docker Compose

```bash
docker compose -f docker-compose.dev.yml up --build
```

Initial image build takes ~2–3 minutes. Subsequent starts are much faster.

| Service | URL |
|---|---|
| **API** | http://localhost:8000 |
| **API Health** | http://localhost:8000/health |
| **Frontend Dashboard** | http://localhost:5173 |
| **Grafana** | http://localhost:3000 |
| **Jaeger (Tracing)** | http://localhost:16686 |
| **Prometheus** | http://localhost:9090 |

### Stop

```bash
# Stop services (data preserved)
docker compose -f docker-compose.dev.yml down

# Stop services and remove all volumes
docker compose -f docker-compose.dev.yml down -v
```

---

## ML Pipeline

The pipeline is defined in [`dvc.yaml`](dvc.yaml) and parameterised via [`params.yaml`](params.yaml).

```
data/raw/hotel_bookings.csv
       │
       ▼  preprocess
data/processed/dataset.parquet
       │
       ▼  split  (test_size=0.20, seed=42)
train.parquet · cal.parquet · test.parquet
       │
       ▼  train  (cv_folds=5)
models/<run_id>/model.joblib
       │
       ▼  calibrate  (sigmoid → isotonic)
models/<run_id>/model_calibrated.joblib
       │
       ▼  evaluate
reports/metrics/<run_id>/
```

### Cost matrix (configurable in `params.yaml`)

| | Predicted: Cancel | Predicted: No Cancel |
|---|---|---|
| **Actual: Cancel** | TP = +$180 | FN = −$200 |
| **Actual: No Cancel** | FP = −$20 | TN = $0 |

### CLI commands

```bash
python main.py preprocess    # Feature engineering & validation
python main.py split         # Train / calibration / test split
python main.py train         # Model training with cross-validation
python main.py evaluate      # AUC, Brier score, realized profit
python main.py predict       # Batch inference
```

Reproduce the entire pipeline:

```bash
dvc repro
```

---

## API Reference

### Start the server

```bash
# Environment variables (see .env.example for full list)
export DS_API_KEY=your-secret-key
export RATE_LIMIT_BACKEND=redis          # or "memory" for single-pod dev
export REDIS_URL=redis://localhost:6379/0

python main.py serve-api --host 0.0.0.0 --port 8000
```

### Authentication

All prediction and admin endpoints require:

```
x-api-key: <DS_API_KEY>
```

### Endpoints

#### Operational

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe — model & policy loaded |
| `GET` | `/metrics` | Prometheus metrics scrape target |

#### Inference

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/predict_proba` | Cancellation probability (v1 model) |
| `POST` | `/v2/predict_proba` | Cancellation probability (v2 model) |
| `POST` | `/v1/decide` | Policy-based binary decision (v1) |
| `POST` | `/v2/decide` | Policy-based binary decision (v2) |
| `POST` | `/reload` | Hot-reload model & policy without restart |

#### Dashboard

| Method | Path | Description |
|---|---|---|
| `GET` | `/dashboard/api/overview` | Train/test metrics summary |
| `GET` | `/dashboard/api/runs` | Experiment run history |
| `GET` | `/dashboard/api/db-status` | Database connectivity status |

#### Auth

| Method | Path | Description |
|---|---|---|
| `POST` | `/auth/login` | Create authenticated session |
| `POST` | `/auth/logout` | Invalidate session |
| `GET` | `/auth/me` | Current session info |

#### Rate limiting

| Backend | Use case |
|---|---|
| `memory` | Single-pod / development |
| `redis` | Distributed / multi-replica production |

---

## Dashboard & Frontend

### Run the frontend locally

```bash
cd apps/frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000

### Dashboard pages

| Page | Description |
|---|---|
| Overview | Real-time model metrics summary |
| Model Analysis | ROC curves, calibration plots, feature importance |
| Run History | All experiment runs with metrics comparison |
| System & Database | Service health, DB connectivity status |

**Default dev credentials:** username `admin` / password `ChangeMe123!`

---

## Monitoring & Alerting

```bash
python main.py monitor
```

Outputs:
- `reports/monitoring/<run_id>/monitoring_report.json`
- `reports/monitoring/latest_monitoring_report.json`

### Checks performed

| Check | Method |
|---|---|
| Feature drift | PSI (Population Stability Index) |
| Prediction drift | PSI + KS test |
| Model performance | AUC, Brier score, realized profit |
| Alert flags | Threshold-based flags with webhook delivery |

### Webhook alerting

Set `ALERT_WEBHOOK_URL` in your environment to receive alert payloads.

**Dead-letter queue retry** (for failed webhook deliveries):

```bash
python main.py retry-webhook-dlq --url https://example.com/webhook
```

> CI/CD: [`monitor.yml`](.github/workflows/monitor.yml) automatically triggers a policy rollback when `any_alert=true` appears in the monitoring report.

---

## Policy Management — Rollout & Rollback

### Promote a run's policy

```bash
python main.py promote-policy --run-id 20260217_220731
```

### Blue/Green promotion

```bash
python main.py promote-policy --run-id 20260217_220731 --slot blue
python main.py promote-policy --run-id 20260217_220731 --slot green
```

### Rollback

```bash
# Roll back to previous policy
python main.py rollback-policy

# Roll back a specific slot
python main.py rollback-policy --slot blue
```

---

## Testing

```bash
pytest
```

Branch coverage is enforced at **≥ 80%** (configured in `pyproject.toml`).

### Test scope

| Category | Examples |
|---|---|
| Unit — policy | Threshold selection, promotion logic |
| Unit — validation | Pandera schema, drift detectors, anomaly flags |
| Integration | API endpoints, database stores |
| Smoke | End-to-end pipeline pass-through |

---

## Deployment

### Docker (single container)

```bash
docker build -t hotel-booking-cancellation-prediction:latest .
docker run \
  -e DS_API_KEY=your-secret-key \
  -p 8000:8000 \
  hotel-booking-cancellation-prediction:latest
```

### Docker Compose (full stack)

```bash
docker compose -f docker-compose.dev.yml up --build
```

Includes: API · Frontend · PostgreSQL + pgvector · Redis · Prometheus · Grafana · Jaeger

### Kubernetes

Manifests are in [`deploy/k8s/`](deploy/k8s/):

```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/secrets.example.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/hpa.yaml
kubectl apply -f deploy/k8s/network-policy.yaml
kubectl apply -f deploy/k8s/pdb.yaml
kubectl apply -f deploy/k8s/ingress.yaml
kubectl apply -f deploy/k8s/canary-deployment.yaml
kubectl apply -f deploy/k8s/canary-ingress.yaml
```

Canary traffic weight is controlled via NGINX annotations in [`deploy/k8s/canary-ingress.yaml`](deploy/k8s/canary-ingress.yaml) (default: **10%**).

### Helm + GitOps

Helm chart and GitOps configuration are available in [`deploy/helm/`](deploy/helm/) and [`deploy/gitops/`](deploy/gitops/).

---

## CI/CD

| Workflow | File | Trigger |
|---|---|---|
| Continuous Integration | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Push / PR |
| Deployment | [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) | Push to `main` |
| Drift Monitoring | [`.github/workflows/monitor.yml`](.github/workflows/monitor.yml) | Scheduled / Push |
| Security Scan | [`.github/workflows/security.yml`](.github/workflows/security.yml) | Push / PR |

The monitoring workflow automatically triggers a **policy rollback** when `any_alert=true` is detected in the latest monitoring report.

---

## Performance & Load Testing

### Locust (interactive UI)

```bash
locust -f perf/locustfile.py --host http://127.0.0.1:8000
```

### k6 (scripted SLO validation)

```bash
k6 run perf/k6_smoke.js
```

SLO thresholds defined in the k6 script:

| Metric | Threshold |
|---|---|
| p95 latency | < 300 ms |
| p99 latency | < 800 ms |
| Error rate | < 1% |

---

## Data Lineage & Versioning

| Artefact | Location |
|---|---|
| Pipeline definition | [`dvc.yaml`](dvc.yaml) |
| Hyperparameters | [`params.yaml`](params.yaml) |
| Preprocess lineage | `reports/metrics/data_lineage_preprocess.json` |
| Per-run lineage | `reports/metrics/<run_id>/data_lineage.json` |

---

## Environment Variables

See [`.env.example`](.env.example) for the complete, annotated list.

| Variable | Required | Description |
|---|---|---|
| `DS_API_KEY` | ✅ | API authentication key |
| `POSTGRES_PASSWORD` | ✅ | PostgreSQL password |
| `GF_ADMIN_PASSWORD` | ✅ | Grafana admin password |
| `DASHBOARD_ADMIN_PASSWORD_ADMIN` | ✅ | Dashboard admin password (bcrypt hash) |
| `REDIS_URL` | ✅ (prod) | Redis connection string |
| `RATE_LIMIT_BACKEND` | — | `memory` (default) or `redis` |
| `ALERT_WEBHOOK_URL` | — | Webhook URL for monitoring alerts |
| `DATABASE_URL` | — | PostgreSQL DSN (overrides compose default) |
| `OLLAMA_BASE_URL` | — | Ollama API URL (default: `http://localhost:11434`) |

---

## Operational Docs

| Document | Description |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Full system architecture reference |
| [docs/runbook.md](docs/runbook.md) | On-call runbook — alerts, remediation steps |
| [docs/slo.md](docs/slo.md) | Service Level Objectives |
| [docs/adr/](docs/adr/) | Architecture Decision Records (ADR-001 → ADR-013) |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## Project Structure

```
.
├── apps/
│   ├── backend/          # Production entrypoint (python -m apps.backend.main)
│   └── frontend/         # React 18 SPA (Vite)
├── data/
│   ├── raw/              # Raw CSV (not tracked by Git — download from Kaggle)
│   ├── interim/          # Intermediate processing artefacts
│   └── processed/        # DVC-tracked Parquet files
├── deploy/
│   ├── k8s/              # Kubernetes manifests
│   ├── helm/             # Helm chart
│   ├── gitops/           # GitOps configuration
│   ├── dev/              # Local dev overrides
│   └── monitoring/       # Grafana dashboards, Prometheus rules
├── docs/
│   └── adr/              # Architecture Decision Records
├── models/               # Trained model artefacts (not tracked by Git)
├── notebooks/            # Exploratory analysis
├── perf/                 # Locust + k6 load test scripts
├── reports/              # Metrics, monitoring reports, evaluation outputs
├── scripts/              # Utility scripts (check_setup.py, etc.)
├── src/                  # Core ML + API source code
├── tests/                # pytest test suite
├── dvc.yaml              # DVC pipeline definition
├── params.yaml           # Hyperparameters + cost matrix
├── docker-compose.dev.yml
├── docker-compose.prod.yml
└── pyproject.toml
```

---

<p align="center">
  Built with FastAPI · React · PostgreSQL · Redis · Ollama · Prometheus · Kubernetes
</p>
