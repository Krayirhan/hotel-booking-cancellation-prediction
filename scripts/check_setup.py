#!/usr/bin/env python3
"""
DS-Project — Tam Sistem Hazırlık Kontrolü
==========================================
GitHub'dan klonladıktan sonra, Docker ile açmadan ÖNCE çalıştırın.
Eksik olan her şeyi tek tek listeler ve nasıl düzeleceğini söyler.

Kullanım:
    python scripts/check_setup.py           # Kontrol
    python scripts/check_setup.py --fix     # Kontrol + .env otomatik oluştur
    python scripts/check_setup.py --json    # Makine-okunabilir JSON çıktı
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIX = "--fix" in sys.argv
JSON_OUT = "--json" in sys.argv

# ── Windows: force UTF-8 console + stdout to prevent UnicodeEncodeError ──────
# Windows cmd / PowerShell default code page (cp1252) cannot encode emoji
# (✅ ⚠️ ❌ ─) used throughout this script.
# Step 1: Set Windows console code pages to UTF-8 (chcp 65001).
# Step 2: Reconfigure Python's stdout/stderr to UTF-8 so str→bytes is correct.
# errors='replace' keeps the script alive even if a surrogate slips through.
if sys.platform == "win32":
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleCP(65001)  # type: ignore[attr-defined]
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass  # Python < 3.7 — best-effort

# ─── Renk desteği ────────────────────────────────────────────────────────────
_RESET = "\033[0m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_DIM = "\033[2m"

_USE_COLOR = (not JSON_OUT) and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"{code}{text}{_RESET}" if _USE_COLOR else text


def ok(msg: str) -> str:
    return _c(_GREEN, f"  ✅  {msg}")


def warn(msg: str) -> str:
    return _c(_YELLOW, f"  ⚠️   {msg}")


def fail(msg: str) -> str:
    return _c(_RED, f"  ❌  {msg}")


def info(msg: str) -> str:
    return _c(_DIM, f"       {msg}")


def hdr(title: str, n: int, total: int) -> str:
    bar = f"{n} / {total}"
    return _c(_BOLD + _CYAN, f"\n{'─' * 60}\n  {bar}  {title}\n{'─' * 60}")


# ─── Durum takibi ─────────────────────────────────────────────────────────────
_issues: list[dict[str, str]] = []  # kritik — bunlar olmadan uygulama başlamaz
_warnings: list[dict[str, str]] = []  # opsiyonel — devam edilebilir


def _issue(key: str, msg: str, fix: str = "") -> None:
    _issues.append({"key": key, "msg": msg, "fix": fix})


def _warning(key: str, msg: str, fix: str = "") -> None:
    _warnings.append({"key": key, "msg": msg, "fix": fix})


# ─── Minimalist .env yükleyici ───────────────────────────────────────────────
def _load_dotenv(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# ─── Yardımcılar ─────────────────────────────────────────────────────────────
def _run(cmd: list[str], timeout: int = 6) -> tuple[bool, str]:
    """Komutu çalıştır; (başarı, çıktı) döndür."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return True, (r.stdout or r.stderr).strip()
    except FileNotFoundError:
        return False, "komut bulunamadı"
    except subprocess.TimeoutExpired:
        return False, "zaman aşımı"
    except Exception as e:
        return False, str(e)


def _http_get(url: str, timeout: int = 4) -> tuple[bool, Any]:
    """Basit HTTP GET; (başarı, parsed_json_veya_str) döndür."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode()
            try:
                return True, json.loads(body)
            except Exception:
                return True, body
    except Exception as e:
        return False, str(e)


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  KONTROL FONKSİYONLARI
# ═══════════════════════════════════════════════════════════════════════════════

TOTAL_CHECKS = 10


def check_python() -> None:
    """1 — Python >= 3.12"""
    print(hdr("Python Sürümü", 1, TOTAL_CHECKS))
    major, minor = sys.version_info[:2]
    ver = f"{major}.{minor}.{sys.version_info[2]}"
    if (major, minor) >= (3, 12):
        print(ok(f"Python {ver}  (>= 3.12)"))
    else:
        print(fail(f"Python {ver} — en az 3.12 gerekli"))
        print(info("https://www.python.org/downloads/"))
        _issue(
            "python_version",
            f"Python {ver} — 3.12+ gerekli",
            "https://www.python.org/downloads/",
        )


def check_python_packages() -> None:
    """2 — Gerekli Python paketleri"""
    print(hdr("Python Paketleri", 2, TOTAL_CHECKS))

    # (import_adı, pip_adı, zorunlu_mu)
    PACKAGES: list[tuple[str, str, bool]] = [
        ("fastapi", "fastapi", True),
        ("uvicorn", "uvicorn", True),
        ("pydantic", "pydantic", True),
        ("sklearn", "scikit-learn", True),
        ("pandas", "pandas", True),
        ("numpy", "numpy", True),
        ("joblib", "joblib", True),
        ("xgboost", "xgboost", True),
        ("lightgbm", "lightgbm", True),
        ("catboost", "catboost", True),
        ("httpx", "httpx", True),
        ("bcrypt", "bcrypt", True),
        ("redis", "redis", True),
        ("sqlalchemy", "sqlalchemy", True),
        ("psycopg", "psycopg[binary]", True),
        ("prometheus_client", "prometheus-client", True),
        ("pandera", "pandera", True),
        ("yaml", "pyyaml", True),
        ("opentelemetry", "opentelemetry-api", False),
        ("mlflow", "mlflow", False),
        ("optuna", "optuna", False),
        ("shap", "shap", False),
        ("pytest", "pytest", False),
        ("dvc", "dvc", False),
    ]

    missing_required: list[str] = []
    missing_optional: list[str] = []

    for import_name, pip_name, required in PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            print(ok(f"{pip_name:<32} {ver}"))
        except ImportError:
            if required:
                print(fail(f"{pip_name:<32} EKSIK"))
                missing_required.append(pip_name)
            else:
                print(warn(f"{pip_name:<32} yüklü değil (opsiyonel)"))
                missing_optional.append(pip_name)

    if missing_required:
        cmd = f"pip install {' '.join(missing_required)}"
        print(info(f"Zorunlu eksikler icin: {cmd}"))
        _issue(
            "python_packages",
            f"Eksik zorunlu paketler: {', '.join(missing_required)}",
            cmd,
        )
    if missing_optional:
        cmd = f"pip install {' '.join(missing_optional)}"
        print(info(f"Opsiyonel icin: {cmd}"))
        _warning(
            "python_packages_optional",
            f"Eksik opsiyonel paketler: {', '.join(missing_optional)}",
            cmd,
        )


def check_docker() -> None:
    """3 — Docker Engine + Compose v2"""
    print(hdr("Docker", 3, TOTAL_CHECKS))

    ok_docker, out_docker = _run(["docker", "--version"])
    if ok_docker:
        print(ok(f"Docker Engine: {out_docker.splitlines()[0]}"))
    else:
        print(fail("Docker Engine bulunamadı"))
        print(info("https://docs.docker.com/get-docker/"))
        _issue("docker", "Docker kurulu degil", "https://docs.docker.com/get-docker/")

    ok_compose, out_compose = _run(["docker", "compose", "version"])
    if ok_compose:
        print(ok(f"Docker Compose v2: {out_compose.splitlines()[0]}"))
    else:
        print(fail("Docker Compose v2 bulunamadı"))
        print(info("https://docs.docker.com/compose/install/"))
        _issue(
            "docker_compose",
            "Docker Compose v2 eksik",
            "https://docs.docker.com/compose/install/",
        )


def check_nodejs() -> None:
    """4 — Node.js >= 18 + npm"""
    print(hdr("Node.js / npm  (Frontend gelistirme)", 4, TOTAL_CHECKS))

    ok_node, out_node = _run(["node", "--version"])
    if ok_node:
        ver_str = out_node.strip().lstrip("v")
        try:
            major = int(ver_str.split(".")[0])
            if major >= 18:
                print(ok(f"Node.js v{ver_str}  (>= 18)"))
            else:
                print(fail(f"Node.js v{ver_str} — en az v18 gerekli"))
                _issue(
                    "nodejs",
                    f"Node.js v{ver_str} — 18+ gerekli",
                    "https://nodejs.org/en/download/",
                )
        except ValueError:
            print(ok(f"Node.js {out_node}"))
    else:
        print(
            warn("Node.js bulunamadı — Docker Compose ile frontend acilacaksa gerekmez")
        )
        print(info("Lokal dev icin: https://nodejs.org/en/download/"))
        _warning("nodejs", "Node.js yuklu degil", "https://nodejs.org/en/download/")

    ok_npm, out_npm = _run(["npm", "--version"])
    if ok_npm:
        print(ok(f"npm v{out_npm.strip()}"))
    else:
        print(warn("npm bulunamadı — Node.js ile birlikte gelir"))


def check_ollama() -> None:
    """5 — Ollama: kurulu mu? calisiyor mu? model indirilmis mi?"""
    print(hdr("Ollama  (AI Chat)", 5, TOTAL_CHECKS))

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    # Docker icindeki adresi lokale ceviriyoruz
    ollama_url_local = ollama_url.replace("host.docker.internal", "localhost")
    target_model = os.getenv("OLLAMA_MODEL", "")

    # ── A) Ollama CLI ──────────────────────────────────────────────────────────
    ok_cli, out_cli = _run(["ollama", "--version"])
    if ok_cli:
        print(ok(f"Ollama CLI: {out_cli.splitlines()[0]}"))
    else:
        print(warn("Ollama CLI bulunamadı — PATH'de olmayabilir veya kurulmamis"))
        print(info("Kurulum: https://ollama.com/download"))
        print(
            info("Olmadan AI chat calismaz; diger tum ozellikler calismaya devam eder.")
        )
        _warning("ollama_cli", "Ollama CLI yuklu degil", "https://ollama.com/download")

    # ── B) Ollama servisi erisilebilir mi? ────────────────────────────────────
    tags_url = f"{ollama_url_local}/api/tags"
    reachable, tags_data = _http_get(tags_url, timeout=3)

    if reachable:
        print(ok(f"Ollama servisi: {ollama_url_local}  → yanit veriyor"))
    else:
        msg = str(tags_data)[:60]
        print(fail(f"Ollama servisi erisilemdei: {ollama_url_local}"))
        print(info(f"Hata: {msg}"))
        print(info("Baslatmak icin:  ollama serve"))
        print(info("Sadece AI chat etkilenir — diger API'lar calismaya devam eder."))
        _warning(
            "ollama_service",
            f"Ollama servisi calismiyor ({ollama_url_local})",
            "ollama serve",
        )
        # Servis yoksa model kontrolü anlamsiz
        if not target_model:
            print(warn("OLLAMA_MODEL de tanimsiz — .env dosyasina ekleyin"))
            _warning(
                "ollama_model_env",
                "OLLAMA_MODEL tanimsiz",
                ".env: OLLAMA_MODEL=llama3.2:3b",
            )
        else:
            print(
                info(
                    f"Model cekme komutu (servis ayaga kalkinca): ollama pull {target_model}"
                )
            )
        return

    # ── C) Indirilen modeller listesi ─────────────────────────────────────────
    if isinstance(tags_data, dict):
        models: list[str] = [m.get("name", "") for m in tags_data.get("models", [])]
    else:
        models = []

    if models:
        print(ok(f"Yuklu modeller ({len(models)} adet):"))
        for m in models:
            size_info = ""
            if isinstance(tags_data, dict):
                for entry in tags_data.get("models", []):
                    if entry.get("name") == m:
                        size = entry.get("size", 0)
                        if size:
                            size_info = f"  ({size / 1e9:.1f} GB)"
                        break
            print(info(f"  • {m}{size_info}"))
    else:
        print(warn("Hic model indirilmemis"))

    # ── D) Hedef model ────────────────────────────────────────────────────────
    if not target_model:
        print(warn("OLLAMA_MODEL env var tanimsiz"))
        print(info(".env dosyasina ekleyin: OLLAMA_MODEL=llama3.2:3b"))
        _warning(
            "ollama_model_env",
            "OLLAMA_MODEL tanimsiz",
            ".env: OLLAMA_MODEL=llama3.2:3b",
        )
        return

    # Tam eslesme VEYA name prefix eslesmesi (tag olmadan)
    model_found = any(
        m == target_model or m.startswith(target_model.split(":")[0]) for m in models
    )

    if model_found:
        print(ok(f"Hedef model mevcut: {target_model}"))
    else:
        print(fail(f"Hedef model eksik: {target_model}"))
        print(info(f"Cekme komutu:  ollama pull {target_model}"))
        print(info("Kucuk alternatif (~2 GB):  ollama pull llama3.2:3b"))
        print(info("Cok kucuk alternatif (~637 MB):  ollama pull qwen2.5:0.5b"))
        _issue(
            "ollama_model",
            f"Ollama modeli '{target_model}' indirilmemis",
            f"ollama pull {target_model}",
        )


def check_env_file() -> None:
    """6 — .env dosyasi"""
    print(hdr(".env Dosyasi", 6, TOTAL_CHECKS))
    env_path = ROOT / ".env"
    example_path = ROOT / ".env.example"

    if env_path.exists():
        print(ok(".env mevcut"))
        _load_dotenv(env_path)
        raw = env_path.read_text(encoding="utf-8")
        if "replace-me" in raw:
            print(warn(".env icinde 'replace-me' degerleri var — duzenleyin"))
            _warning(
                "env_replaceme",
                ".env icinde duzenlenmemis 'replace-me' degerleri var",
                ".env dosyasini editor ile acip doldurun",
            )
    else:
        if example_path.exists():
            if FIX:
                shutil.copy(example_path, env_path)
                _load_dotenv(env_path)
                print(ok(".env.example -> .env kopyalandi (--fix)"))
                print(warn("'replace-me' degerlerini .env dosyasinda duzenleyin!"))
                _warning(
                    "env_replaceme",
                    ".env icinde duzenlenmemis 'replace-me' degerleri var",
                    ".env dosyasini editor ile acip doldurun",
                )
            else:
                print(warn(".env bulunamadi"))
                print(info("Windows : copy .env.example .env"))
                print(info("Linux/Mac: cp .env.example .env"))
                print(info("Otomatik : python scripts/check_setup.py --fix"))
                _warning(
                    "env_missing",
                    ".env dosyasi yok",
                    "cp .env.example .env  veya  python scripts/check_setup.py --fix",
                )
        else:
            print(fail(".env ve .env.example her ikisi de eksik"))
            _issue("env_files", ".env ve .env.example yok")


def check_env_vars() -> None:
    """7 — Ortam degiskenleri"""
    print(hdr("Ortam Degiskenleri", 7, TOTAL_CHECKS))

    REQUIRED: dict[str, str] = {
        "DS_API_KEY": (
            "API kimlik dogrulama anahtari (x-api-key header'i).\n"
            "       Herhangi guclu bir string olabilir."
        ),
        "DASHBOARD_ADMIN_PASSWORD_ADMIN": (
            "Admin dashboard sifresi.\n"
            "       Bcrypt hash onerilir:\n"
            "         python -c \"import bcrypt; print(bcrypt.hashpw(b'sifre', bcrypt.gensalt()).decode())\"\n"
            "       Duz metin da calisir ama env sizintisinda tehlikelidir."
        ),
        "POSTGRES_PASSWORD": (
            "PostgreSQL sifresi — docker-compose.dev.yml icin zorunlu.\n"
            "       Bos birakilirsa PostgreSQL ve API veritabani baglantisi basarisiz olur."
        ),
        "GF_ADMIN_PASSWORD": (
            "Grafana yonetici sifresi — docker-compose.dev.yml icin zorunlu.\n"
            "       Bos birakilirsa Grafana containeri baslamaz."
        ),
    }

    OPTIONAL: dict[str, str] = {
        "DATABASE_URL": "PostgreSQL — yoksa SQLite (reports/dashboard.db)",
        "REDIS_URL": "Redis — yoksa in-memory rate limiter",
        "RATE_LIMIT_BACKEND": "redis | memory",
        "CORS_ALLOW_ORIGINS": "Frontend originleri",
        "DASHBOARD_URL": "Frontend URL (redirect icin)",
        "DASHBOARD_AUTH_ENABLED": "true | false",
        "DASHBOARD_TOKEN_TTL_MINUTES": "Token suresi dk (varsayilan: 480)",
        "OLLAMA_BASE_URL": "Ollama API — AI chat icin",
        "OLLAMA_MODEL": "Ollama model adi (orn: llama3.2:3b)",
        "CHAT_SESSION_TTL_SECONDS": "Chat oturum TTL sn (varsayilan: 3600)",
        "OTEL_ENABLED": "OpenTelemetry: true | false",
        "LOG_FORMAT": "json | text",
        "ALERT_WEBHOOK_URL": "Alarm webhook URL",
    }

    print("  -- Zorunlu --")
    for key, desc in REQUIRED.items():
        val = os.getenv(key, "")
        if val and val != "replace-me":
            masked = val[:4] + "***" if len(val) > 4 else "***"
            print(ok(f"{key} = {masked}"))
        else:
            print(fail(f"{key}  <- EKSIK"))
            for line in desc.splitlines():
                print(info(line))
            _issue(f"env_{key}", f"Zorunlu env var eksik: {key}")

    print("\n  -- Opsiyonel --")
    for key, desc in OPTIONAL.items():
        val = os.getenv(key, "")
        if val and val not in ("replace-me", ""):
            masked = (val[:12] + "...") if len(val) > 12 else val
            print(ok(f"{key} = {masked}"))
        else:
            print(warn(f"{key}  <- tanimsiz"))
            print(info(desc))
            _warning(f"env_{key}", f"Opsiyonel env var tanimsiz: {key}")


def check_connectivity() -> None:
    """8 — Veritabani ve Redis baglantisi"""
    print(hdr("Servis Baglantilari  (DB / Redis)", 8, TOTAL_CHECKS))

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    db_url = os.getenv("DATABASE_URL", "")
    if db_url and "postgresql" in db_url:
        try:
            from sqlalchemy import create_engine, text as sa_text

            engine = create_engine(
                db_url, pool_pre_ping=True, connect_args={"connect_timeout": 4}
            )
            with engine.connect() as conn:
                conn.execute(sa_text("SELECT 1"))
            print(ok("PostgreSQL baglantisi basarili"))
        except Exception as e:
            msg = str(e).splitlines()[0][:80]
            print(fail(f"PostgreSQL baglantisi basarisiz: {msg}"))
            print(
                info(
                    "Docker Compose ile: docker compose -f docker-compose.dev.yml up -d postgres"
                )
            )
            print(info("Yerel: PostgreSQL servisi calistigini dogrulayin"))
            _warning(
                "postgres",
                f"PostgreSQL baglanti hatasi: {msg}",
                "docker compose -f docker-compose.dev.yml up -d postgres",
            )
    elif db_url:
        print(ok(f"DATABASE_URL tanimli (SQLite/diger): {db_url[:50]}"))
    else:
        print(warn("DATABASE_URL tanimsiz — SQLite fallback kullanilir"))
        _warning("postgres", "DATABASE_URL yok — SQLite kullanilir")

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url = os.getenv("REDIS_URL", "")
    if redis_url:
        try:
            import redis as _redis  # type: ignore

            r = _redis.from_url(redis_url, socket_connect_timeout=3)
            r.ping()
            print(ok(f"Redis baglantisi basarili: {redis_url}"))
        except Exception as e:
            msg = str(e).splitlines()[0][:80]
            print(fail(f"Redis baglantisi basarisiz: {msg}"))
            print(info("docker compose -f docker-compose.dev.yml up -d redis"))
            _warning(
                "redis",
                f"Redis baglanti hatasi: {msg}",
                "docker compose -f docker-compose.dev.yml up -d redis",
            )
    else:
        print(warn("REDIS_URL tanimsiz — in-memory rate limiter kullanilir"))
        _warning("redis", "REDIS_URL yok — in-memory fallback")


def check_files() -> None:
    """9 — Proje dosyalari"""
    print(hdr("Proje Dosyalari", 9, TOTAL_CHECKS))

    REQUIRED_FILES: dict[str, str] = {
        "models/latest.json": (
            "Egitimli model yok — once pipeline'i calistirin:\n"
            "         1) pip install -r requirements.txt\n"
            "         2) python main.py preprocess\n"
            "         3) python main.py train\n"
            "         4) python main.py evaluate\n"
            "         Ham veri yoksa once data/raw/ altina yukleyin."
        ),
    }

    OPTIONAL_ITEMS: dict[str, str] = {
        "data/raw": "Ham veri (ML pipeline icin) — Kaggle'dan indirin",
        ".env": ".env ortam degiskenleri dosyasi",
        "apps/frontend": "Frontend React uygulamasi",
        "params.yaml": "Pipeline parametreleri",
    }

    print("  -- Zorunlu --")
    for rel, desc in REQUIRED_FILES.items():
        path = ROOT / rel
        if path.exists():
            if rel == "models/latest.json":
                try:
                    data = json.loads(path.read_text())
                    run_id = data.get("run_id", "?")
                    model_dir = ROOT / "models" / run_id
                    if model_dir.exists():
                        print(
                            ok(f"{rel}  (aktif run: {run_id} — model klasoru mevcut)")
                        )
                    else:
                        print(
                            warn(
                                f"{rel}  (run_id: {run_id} ama models/{run_id}/ EKSIK!)"
                            )
                        )
                        print(info("python main.py train"))
                        _warning(
                            "model_dir",
                            f"models/{run_id}/ klasoru yok",
                            "python main.py train",
                        )
                except Exception:
                    print(ok(f"{rel}"))
            else:
                print(ok(f"{rel}"))
        else:
            print(fail(f"{rel}  <- EKSIK"))
            for line in desc.splitlines():
                print(info(line))
            _issue(f"file_{rel.replace('/', '_')}", f"Dosya eksik: {rel}")

    print("\n  -- Opsiyonel --")
    for rel, desc in OPTIONAL_ITEMS.items():
        path = ROOT / rel
        if path.exists():
            if path.is_dir():
                entries = [p for p in path.iterdir() if not p.name.startswith(".")]
                if entries:
                    print(ok(f"{rel}/  ({len(entries)} oge)"))
                else:
                    print(warn(f"{rel}/  -> BOS KLASOR"))
                    _warning(f"dir_{rel.replace('/', '_')}", f"Bos klasor: {rel}")
            else:
                print(ok(f"{rel}"))
        else:
            print(warn(f"{rel}  <- yok"))
            print(info(desc))
            _warning(f"file_{rel.replace('/', '_')}", f"Opsiyonel eksik: {rel}")


def check_ports() -> None:
    """10 — Port cakismalari (Docker Compose oncesi)"""
    print(hdr("Port Durumu  (Docker Compose)", 10, TOTAL_CHECKS))
    print(
        info("Asagidaki portlar Docker Compose tarafindan kullanilacak — bos olmali\n")
    )

    COMPOSE_PORTS: dict[int, str] = {
        8000: "API (FastAPI)",
        5173: "Frontend (React/Vite)",
        5432: "PostgreSQL",
        6379: "Redis",
        9090: "Prometheus",
        3000: "Grafana",
        16686: "Jaeger UI",
        4317: "Jaeger OTLP gRPC",
    }

    for port, service in COMPOSE_PORTS.items():
        if _port_open("127.0.0.1", port):
            print(warn(f":{port:<6} MESGUL  -> {service}"))
            _warning(
                f"port_{port}",
                f"Port {port} mesgul ({service})",
                "docker compose down  veya  ilgili servisi durdurun",
            )
        else:
            print(ok(f":{port:<6} serbest -> {service}"))


# ═══════════════════════════════════════════════════════════════════════════════
#  OZET
# ═══════════════════════════════════════════════════════════════════════════════


def summary() -> int:
    print(f"\n{'=' * 60}")

    if JSON_OUT:
        out = {
            "errors": _issues,
            "warnings": _warnings,
            "ready": len(_issues) == 0,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if not _issues else 1

    if not _issues and not _warnings:
        print(ok("Her sey hazir!"))
        print(_c(_GREEN, "\n  Baslatmak icin:"))
        print("    docker compose -f docker-compose.dev.yml up --build\n")
        print("  URL'ler:")
        print("    API         -> http://localhost:8000")
        print("    Frontend    -> http://localhost:5173")
        print("    Grafana     -> http://localhost:3000  (admin / admin)")
        print("    Jaeger      -> http://localhost:16686")
        return 0

    if _issues:
        n = len(_issues)
        print(
            _c(
                _RED + _BOLD,
                f"\n  KRITIK HATA ({n}) — bunlar giderilmeden uygulama baslamaz:\n",
            )
        )
        for i, err in enumerate(_issues, 1):
            print(f"  {i}. {err['msg']}")
            if err.get("fix"):
                print(info(f"Duzelt: {err['fix']}"))

    if _warnings:
        n = len(_warnings)
        print(_c(_YELLOW, f"\n  UYARI ({n}) — opsiyonel ozellikler etkilenebilir:\n"))
        for i, w in enumerate(_warnings, 1):
            print(f"  {i}. {w['msg']}")
            if w.get("fix"):
                print(info(f"Duzelt: {w['fix']}"))

    print()
    if not _issues:
        print(_c(_GREEN, "  Uyarilari kabul ediyorsaniz baslatabilirsiniz:"))
        print("    docker compose -f docker-compose.dev.yml up --build")
    else:
        print(_c(_RED, "  Kritik hatalari giderin, sonra tekrar calistirin:"))
        print("    python scripts/check_setup.py")

    return 1 if _issues else 0


# ═══════════════════════════════════════════════════════════════════════════════
#  ANA GIRIS
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    if not JSON_OUT:
        print(_c(_BOLD, "\n  DS-Project — Tam Sistem Hazirlik Kontrolu"))
        print(f"  Proje koku : {ROOT}")
        print(f"  Sistem     : {sys.platform}  |  Python {sys.version.split()[0]}")
        if FIX:
            print(_c(_YELLOW, "  Mod        : --fix (otomatik duzeltme acik)"))

    # .env'i mumkun olan en erken yukle
    env_path = ROOT / ".env"
    if env_path.exists():
        _load_dotenv(env_path)

    check_python()
    check_python_packages()
    check_docker()
    check_nodejs()
    check_ollama()
    check_env_file()
    check_env_vars()
    check_connectivity()
    check_files()
    check_ports()

    sys.exit(summary())


if __name__ == "__main__":
    main()
