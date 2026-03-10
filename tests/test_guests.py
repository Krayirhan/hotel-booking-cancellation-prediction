"""test_guests.py — Tests for hotel guest CRUD endpoints (POST/GET/PATCH/DELETE /guests).

Uses FastAPI TestClient + SQLite in-memory DB (set by conftest._isolate_test_db_and_threads).
The ML model is not required — risk defaults to (0.5, 'medium') when serving=None.
"""

from __future__ import annotations

import os
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("DS_API_KEY", "test-key")
os.environ.setdefault("DS_ENV", "development")

from src.api import app  # noqa: E402  (after env setup)

API_KEY = "test-key"
_HEADERS = {"x-api-key": API_KEY}


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def client():
    """TestClient with lifespan — initialises DB + guest store via SQLite."""
    with TestClient(app) as c:
        yield c


def _minimal_guest(**overrides) -> dict:
    """Return a minimal valid GuestCreate payload."""
    base = {
        "first_name": "Ali",
        "last_name": "Yılmaz",
        "email": "ali.yilmaz@example.com",
        "hotel": "City Hotel",
        "lead_time": 14,
        "deposit_type": "No Deposit",
        "market_segment": "Online TA",
        "adults": 2,
        "children": 0,
        "babies": 0,
        "stays_in_week_nights": 3,
        "stays_in_weekend_nights": 1,
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
    }
    base.update(overrides)
    return base


# ─── POST /guests ──────────────────────────────────────────────────────────────


def test_create_guest_returns_201(client):
    payload = _minimal_guest()
    r = client.post("/guests", json=payload, headers=_HEADERS)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["first_name"] == "Ali"
    assert body["last_name"] == "Yılmaz"
    assert body["email"] == "ali.yilmaz@example.com"
    assert "id" in body
    assert isinstance(body["id"], int)


def test_create_guest_includes_risk_fields(client):
    r = client.post("/guests", json=_minimal_guest(), headers=_HEADERS)
    assert r.status_code == 201, r.text
    body = r.json()
    # risk_score is either float (if model loaded) or 0.5 (fallback)
    assert body["risk_score"] is not None
    assert body["risk_label"] in {"high", "medium", "low"}


def test_create_guest_requires_api_key(client):
    r = client.post("/guests", json=_minimal_guest())
    assert r.status_code == 401


def test_create_guest_validates_required_fields(client):
    # Missing first_name → 422
    bad = _minimal_guest()
    del bad["first_name"]
    r = client.post("/guests", json=bad, headers=_HEADERS)
    assert r.status_code == 422


def test_create_guest_minimal_fields_only(client):
    """Only first_name and last_name required; everything else has defaults."""
    payload = {"first_name": "Elif", "last_name": "Kaya"}
    r = client.post("/guests", json=payload, headers=_HEADERS)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["first_name"] == "Elif"
    assert body["hotel"] == "City Hotel"  # default


# ─── GET /guests ───────────────────────────────────────────────────────────────


def test_list_guests_returns_list(client):
    # Create two guests first
    client.post("/guests", json=_minimal_guest(first_name="Ahmet"), headers=_HEADERS)
    client.post("/guests", json=_minimal_guest(first_name="Fatma"), headers=_HEADERS)

    r = client.get("/guests", headers=_HEADERS)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "total" in body
    assert "items" in body
    assert isinstance(body["items"], list)
    assert body["total"] >= 2


def test_list_guests_search_filter(client):
    client.post("/guests", json=_minimal_guest(first_name="Mehmet", last_name="Demir", email="mehmet@x.com"), headers=_HEADERS)
    client.post("/guests", json=_minimal_guest(first_name="Zeynep", last_name="Çelik"), headers=_HEADERS)

    r = client.get("/guests?search=Mehmet", headers=_HEADERS)
    assert r.status_code == 200
    body = r.json()
    names = [item["first_name"] for item in body["items"]]
    assert "Mehmet" in names


def test_list_guests_pagination(client):
    # Create 5 guests
    for i in range(5):
        client.post("/guests", json=_minimal_guest(first_name=f"G{i}", last_name="Test"), headers=_HEADERS)

    r = client.get("/guests?limit=2&offset=0", headers=_HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) <= 2


def test_list_guests_requires_api_key(client):
    r = client.get("/guests")
    assert r.status_code == 401


# ─── GET /guests/{id} ─────────────────────────────────────────────────────────


def test_get_guest_by_id(client):
    r_create = client.post("/guests", json=_minimal_guest(), headers=_HEADERS)
    assert r_create.status_code == 201
    guest_id = r_create.json()["id"]

    r = client.get(f"/guests/{guest_id}", headers=_HEADERS)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == guest_id
    assert body["first_name"] == "Ali"


def test_get_guest_not_found(client):
    r = client.get("/guests/9999999", headers=_HEADERS)
    assert r.status_code == 404


def test_get_guest_requires_api_key(client):
    r = client.get("/guests/1")
    assert r.status_code == 401


# ─── PATCH /guests/{id} ───────────────────────────────────────────────────────


def test_update_guest_personal_info(client):
    r_create = client.post("/guests", json=_minimal_guest(), headers=_HEADERS)
    assert r_create.status_code == 201
    guest_id = r_create.json()["id"]

    r = client.patch(
        f"/guests/{guest_id}",
        json={"email": "new@example.com", "phone": "+905001234567"},
        headers=_HEADERS,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["email"] == "new@example.com"
    assert body["phone"] == "+905001234567"


def test_update_guest_booking_fields_recalculates_risk(client):
    r_create = client.post("/guests", json=_minimal_guest(lead_time=1), headers=_HEADERS)
    assert r_create.status_code == 201
    guest_id = r_create.json()["id"]
    first_risk = r_create.json()["risk_score"]

    # Change a booking field — risk should be recomputed (might differ or stay same)
    r = client.patch(
        f"/guests/{guest_id}",
        json={"lead_time": 300, "previous_cancellations": 5},
        headers=_HEADERS,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # risk_score should still be a valid float
    assert body["risk_score"] is not None
    assert body["risk_label"] in {"high", "medium", "low"}
    # lead_time should be updated
    assert body["lead_time"] == 300


def test_update_guest_not_found(client):
    r = client.patch("/guests/9999999", json={"email": "x@y.com"}, headers=_HEADERS)
    assert r.status_code == 404


def test_update_guest_requires_api_key(client):
    r = client.patch("/guests/1", json={"email": "x@y.com"})
    assert r.status_code == 401


# ─── DELETE /guests/{id} ──────────────────────────────────────────────────────


def test_delete_guest(client):
    r_create = client.post("/guests", json=_minimal_guest(), headers=_HEADERS)
    assert r_create.status_code == 201
    guest_id = r_create.json()["id"]

    r_del = client.delete(f"/guests/{guest_id}", headers=_HEADERS)
    assert r_del.status_code == 204

    # Verify it's gone
    r_get = client.get(f"/guests/{guest_id}", headers=_HEADERS)
    assert r_get.status_code == 404


def test_delete_guest_not_found(client):
    r = client.delete("/guests/9999999", headers=_HEADERS)
    assert r.status_code == 404


def test_delete_guest_requires_api_key(client):
    r = client.delete("/guests/1")
    assert r.status_code == 401


# ─── Idempotency & edge cases ─────────────────────────────────────────────────


def test_create_multiple_guests_different_ids(client):
    r1 = client.post("/guests", json=_minimal_guest(first_name="G1"), headers=_HEADERS)
    r2 = client.post("/guests", json=_minimal_guest(first_name="G2"), headers=_HEADERS)
    assert r1.status_code == 201
    assert r2.status_code == 201
    assert r1.json()["id"] != r2.json()["id"]


def test_created_at_and_updated_at_present(client):
    r = client.post("/guests", json=_minimal_guest(), headers=_HEADERS)
    assert r.status_code == 201
    body = r.json()
    assert "created_at" in body
    assert "updated_at" in body
    assert body["created_at"] is not None
