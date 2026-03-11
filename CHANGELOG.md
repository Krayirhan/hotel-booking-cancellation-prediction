# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.3.0] - 2026-02-19

### Added
- **Test coverage**: 241 Python tests (all passing); frontend 56 Vitest/RTL tests
  - CLI tests: 8 files, 47 tests (`train`, `evaluate`, `monitor`, `predict`, `preprocess`, `hpo`, `explain`, `policy`)
  - Module tests: `experiment_tracking` (17), `tracing` (17), `hpo` (13), `explain` (15)
  - Frontend RTL: `LoginPage`, `ErrorBoundary`, `AppStatusBar`, `useSystemStatus`
- **Prometheus model-health gauges** (`src/metrics.py`): `ds_model_roc_auc`, `ds_feature_psi`, `ds_model_action_rate`, `ds_label_drift_rate`
- **PrometheusRule alerts** (`deploy/monitoring/prometheus-rule.yaml`): `DSProjectHighPSI`, `DSProjectLowAUC`, `DSProjectActionRateAnomaly`
- **Helm K8s hardening** (`deploy/helm/`):
  - Container `securityContext`: `allowPrivilegeEscalation: false`, `readOnlyRootFilesystem: true`, `runAsNonRoot: true`, `capabilities.drop: [ALL]`
  - Pod `securityContext`: `runAsNonRoot: true`, `seccompProfile: RuntimeDefault`
  - `startupProbe`: `failureThreshold: 30`, `periodSeconds: 5` (covers slow model load)
  - `/tmp` `emptyDir` volume for writable scratch space
  - `image.tag` defaulted to `""` — must be set explicitly (no `latest` in prod)
- **Notebooks** (`notebooks/`): 6 end-to-end notebooks
  - `01_eda.ipynb`: Exploratory Data Analysis with target distribution, monthly trend, country cardinality
  - `02_feature_engineering.ipynb`: Cyclic encoding verification, frequency encoding, pipeline output shape
  - `03_training.ipynb`: 5-fold CV model comparison, HPO integration, artifact save
  - `04_evaluation.ipynb`: ROC/PR curves, confusion matrix, cost-matrix threshold sweep
  - `05_calibration.ipynb`: Isotonic vs Sigmoid ECE comparison, reliability diagrams
  - `06_explainability.ipynb`: Permutation importance, SHAP summary/waterfall/dependence
- **ADR-008**: Lua-Based Distributed Rate Limiting
- **ADR-009**: Prometheus Gauges for Model Health Monitoring
- **CI improvements** (`.github/workflows/ci.yml`):
  - `frontend` job: Node.js 20, `npm ci`, lint, build
  - Python `test` job matrix: `['3.10', '3.11']`
  - `bandit` security scan step
  - Coverage gate extended: `experiment_tracking`, `explain`, `hpo`
  - `docker-build` depends on both `test` and `frontend`

### Changed
- `src/features.py`: `FeatureEngineer.get_feature_names_out()` now returns proper `np.ndarray` from `_feature_names_out` (fixes sklearn `set_output` API compatibility)
- `src/rate_limit.py`: Replaced pipeline-based counter with **atomic Lua script** (sliding window, no race condition)
- `deploy/helm/hotel-booking-cancellation-prediction/values.yaml`: `image.tag` changed from `latest` to `""` with mandatory override comment

### Security
- Dashboard auth: production guard prevents `DASHBOARD_ADMIN_*` env fallback in non-dev environments
- CSP headers added to dashboard responses
- Gitleaks secret scanning step added to CI
- Lua atomic rate limiting closes race condition in previous pipeline-based implementation
- Container runs as non-root with dropped Linux capabilities

## [Unreleased — pre-1.3.0]

## [1.2.0] - 2026-02-18

### Added
- **Helm Chart**: Parameterized Kubernetes deployment via Helm chart (`deploy/helm/hotel-booking-cancellation-prediction/`)
  - Environment-specific values: `values-staging.yaml`, `values-production.yaml`
  - Templates: Deployment, Service, Ingress, HPA, PDB, NetworkPolicy, Canary, PrometheusRule
- **GitOps (ArgoCD + Flux)**: Declarative deployment sync (`deploy/gitops/`)
  - ArgoCD: Project + staging (auto-sync) + production (manual approve)
  - Flux: GitRepository + HelmRelease for both environments
- **Distributed Tracing**: OpenTelemetry instrumentation (`src/tracing.py`)
  - OTLP gRPC export to Jaeger/Tempo
  - Automatic FastAPI span creation
  - Custom ML inference spans with `ml.*` attributes
  - Configurable via `OTEL_ENABLED`, `OTEL_EXPORTER_OTLP_ENDPOINT`
- **docker-compose dev stack**: Single-command local environment (`docker-compose.dev.yml`)
  - API + Redis + Prometheus + Grafana + Jaeger
  - Pre-configured datasources and dashboard provisioning
- **Staging deploy workflow**: CI/CD pipeline with approval gate (`.github/workflows/deploy.yml`)
  - Build → Deploy staging → Smoke tests → Manual approve → Deploy production
- **Data validation framework**: Pandera-based schema + distribution assertions (`src/data_validation.py`)
  - Raw data schema (hotel bookings contract)
  - Processed data schema (post-preprocessing)
  - Inference payload validation
  - Distribution drift checks with configurable tolerance
  - Reference stats generation for monitoring
- **API versioning**: `/v1` and `/v2` prefix routing (`src/api_v1.py`, `src/api_v2.py`)
  - V1: Backward-compatible original endpoints
  - V2: Enhanced responses with `meta` block (api_version, model_used, latency_ms, request_id)
  - Root endpoints preserved for backward compatibility
- **Architecture Decision Records**: ADR documentation (`docs/adr/`)
- **CHANGELOG**: Structured change tracking (this file)

### Changed
- API app description updated to reference versioned endpoints
- Tracing integrated into predict_proba and decide endpoints

## [1.1.0] - 2026-02-17

### Added
- Blue/Green deployment slots with promote/rollback CLI commands
- Canary deployment support (K8s manifests)
- Rate limiting with Redis backend support
- API key authentication middleware
- Prometheus metrics (request count, latency histogram, inference counters)
- Grafana dashboard JSON
- PrometheusRule alerts (p95 latency, 5xx rate, inference errors)
- Alertmanager config (Slack + PagerDuty routing)
- Network policy for pod traffic isolation
- PodDisruptionBudget for safe rollouts
- HPA (CPU-based autoscaling)
- Webhook DLQ retry mechanism
- HPO with Optuna
- SHAP + permutation importance explainability
- MLflow experiment tracking integration
- Cost-sensitive decision framework
- Pipeline smoke tests in CI

## [1.0.0] - 2026-02-16

### Added
- Initial ML pipeline: preprocess → train → evaluate → predict → monitor
- Baseline LogisticRegression + challenger models (XGBoost, LightGBM, CatBoost, HistGB)
- Calibration (Isotonic + Sigmoid)
- FastAPI serving endpoint (`/predict_proba`, `/decide`, `/reload`)
- Feature spec contract (JSON schema)
- Decision policy engine with profit-optimal threshold selection
- DVC pipeline definition
- Docker multi-stage build
- CI pipeline (lint, type check, security, tests, coverage gate)
- Basic data validation (schema checks, target label validation)

[Unreleased]: https://github.com/your-org/hotel-booking-cancellation-prediction/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/your-org/hotel-booking-cancellation-prediction/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/your-org/hotel-booking-cancellation-prediction/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/your-org/hotel-booking-cancellation-prediction/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/your-org/hotel-booking-cancellation-prediction/releases/tag/v1.0.0
