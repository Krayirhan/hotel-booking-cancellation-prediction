#!/usr/bin/env bash
# restore_db.sh — Restore a PostgreSQL dump created by backup_db.sh
#
# Usage:
#   DATABASE_URL=postgresql://user:pass@host:5432/db \
#   RESTORE_FILE=/backups/db_20260218_120000.sql.gz \
#   ./scripts/restore_db.sh
#
# Safety notes:
#   - The target database is NOT dropped/re-created automatically.
#     Run `psql … -c "DROP DATABASE … ; CREATE DATABASE …"` first if a
#     clean restore is required.
#   - Use RESTORE_DRY_RUN=1 to print what *would* be executed without
#     touching the database.
#   - Restore is piped through `gunzip` so the .sql.gz file is never
#     fully decompressed to disk — safe for large dumps.

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────
: "${DATABASE_URL:?DATABASE_URL env var is required}"
: "${RESTORE_FILE:?RESTORE_FILE env var is required (path to .sql.gz)}"
DRY_RUN="${RESTORE_DRY_RUN:-0}"

if [[ ! -f "${RESTORE_FILE}" ]]; then
  echo "[restore_db] ERROR: File not found: ${RESTORE_FILE}" >&2
  exit 1
fi

# ── Parse connection parts from DATABASE_URL ──────────────────────────
# Supports:  postgresql+psycopg://user:pass@host:5432/dbname
#            postgresql://user:pass@host:5432/dbname
_url="${DATABASE_URL#postgresql+psycopg://}"
_url="${_url#postgresql://}"

PG_USER="${_url%%:*}"
_rest="${_url#*:}"
PG_PASS="${_rest%%@*}"
_rest="${_rest#*@}"
PG_HOST="${_rest%%:*}"
_rest="${_rest#*:}"
PG_PORT="${_rest%%/*}"
PG_DB="${_rest#*/}"
PG_DB="${PG_DB%%\?*}"

# ── Dry-run guard ──────────────────────────────────────────────────────
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[restore_db] DRY RUN — would restore:"
  echo "  Source : ${RESTORE_FILE}"
  echo "  Target : postgresql://${PG_USER}@${PG_HOST}:${PG_PORT}/${PG_DB}"
  echo "[restore_db] Set RESTORE_DRY_RUN=0 to execute."
  exit 0
fi

# ── Connectivity check ────────────────────────────────────────────────
echo "[restore_db] Checking database connectivity…"
PGPASSWORD="${PG_PASS}" psql \
  --host="${PG_HOST}" \
  --port="${PG_PORT}" \
  --username="${PG_USER}" \
  --dbname="${PG_DB}" \
  --command="SELECT 1;" \
  --quiet \
  > /dev/null

echo "[restore_db] Connected. Starting restore: ${RESTORE_FILE} → ${PG_DB}"

# ── Restore ───────────────────────────────────────────────────────────
START_TS="$(date -u +%Y%m%d_%H%M%S)"

gunzip -c "${RESTORE_FILE}" \
  | PGPASSWORD="${PG_PASS}" psql \
      --host="${PG_HOST}" \
      --port="${PG_PORT}" \
      --username="${PG_USER}" \
      --dbname="${PG_DB}" \
      --single-transaction \
      --set ON_ERROR_STOP=1

END_TS="$(date -u +%Y%m%d_%H%M%S)"
echo "[restore_db] Restore complete (started ${START_TS}, finished ${END_TS})."
echo "[restore_db] IMPORTANT: Run Alembic migrations if needed:"
echo "  alembic upgrade head"
