.PHONY: setup setup-env check hooks lint test test-cov train evaluate predict monitor serve hpo explain load-locust load-k6 clean-artifacts clean-artifacts-dry dev-up dev-down helm-lint helm-template

# ── İlk kurulum (yeni PC) ───────────────────────────────────────────────────

## 1) Hazırlık kontrolü — eksik ne var?
check:
	python scripts/check_setup.py

## 2) .env oluştur + otomatik düzelt
setup-env:
	python scripts/check_setup.py --fix

## 3) Sanal ortam kur
setup:
	python -m venv .venv
	.venv\\Scripts\\python.exe -m pip install --upgrade pip
	.venv\\Scripts\\python.exe -m pip install -r requirements.txt

hooks:
	.venv\\Scripts\\python.exe -m pre_commit install --hook-type pre-commit --hook-type pre-push

lint:
	.venv\\Scripts\\python.exe -m pre_commit run --all-files

test:
	.venv\\Scripts\\python.exe -m pytest -q

test-cov:
	.venv\\Scripts\\python.exe -m pytest --cov=src --cov-report=term-missing

train:
	.venv\\Scripts\\python.exe main.py train

evaluate:
	.venv\\Scripts\\python.exe main.py evaluate

predict:
	.venv\\Scripts\\python.exe main.py predict

monitor:
	.venv\\Scripts\\python.exe main.py monitor

serve:
	.venv\\Scripts\\python.exe main.py serve-api --host 0.0.0.0 --port 8000

load-locust:
	.venv\\Scripts\\python.exe -m locust -f perf/locustfile.py --host http://127.0.0.1:8000

## k6 smoke load test (requires k6 installed: https://k6.io)
load-k6:
	k6 run perf/k6_smoke.js

# Artifact retention policy:
# - Keep newest N run dirs
# - Always keep run in models/latest.json
# - Delete only older-than-threshold dirs outside keep set
clean-artifacts:
	.venv\\Scripts\\python.exe scripts/clean_artifacts.py --models-dir models --latest-json models/latest.json --keep-runs 20 --max-age-days 30 --apply

clean-artifacts-dry:
	.venv\\Scripts\\python.exe scripts/clean_artifacts.py --models-dir models --latest-json models/latest.json --keep-runs 20 --max-age-days 30 --dry-run

hpo:
	.venv\\Scripts\\python.exe main.py hpo --n-trials 50

explain:
	.venv\\Scripts\\python.exe main.py explain

# ── Dev Stack (docker-compose) ──────────────────────────────────────
dev-up:
	docker compose -f docker-compose.dev.yml up --build -d
	@echo "API:        http://localhost:8000"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana:    http://localhost:3000  (admin/admin)"
	@echo "Jaeger UI:  http://localhost:16686"
	@echo "MinIO UI:   http://localhost:9001  (minioadmin/minioadmin)"

dev-down:
	docker compose -f docker-compose.dev.yml down -v

# ── DVC — data versioning & pipeline ────────────────────────────────
dvc-init:
	dvc init

dvc-pull:
	dvc pull

dvc-push:
	dvc push

dvc-repro:
	dvc repro

dvc-status:
	dvc status

dvc-dag:
	dvc dag

# ── Helm ────────────────────────────────────────────────────────────
helm-lint:
	helm lint deploy/helm/hotel-booking-cancellation-prediction

helm-template:
	helm template hotel-booking-cancellation-prediction deploy/helm/hotel-booking-cancellation-prediction --values deploy/helm/hotel-booking-cancellation-prediction/values.yaml
