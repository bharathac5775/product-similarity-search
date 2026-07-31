"""Tests for the FastAPI service (health + find_similar_products + error codes)."""
from fastapi.testclient import TestClient

from app import build_app
from product_similarity.engine import SimilarityEngine


def _client(sample_ldjson_path):
    eng = SimilarityEngine.from_path(sample_ldjson_path)
    return TestClient(build_app(eng))


def test_health(sample_ldjson_path):
    c = _client(sample_ldjson_path)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_find_ok(sample_ldjson_path):
    c = _client(sample_ldjson_path)
    r = c.get("/find_similar_products", params={"product_id": "p1", "num_similar": 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 3


def test_find_unknown_id_404(sample_ldjson_path):
    c = _client(sample_ldjson_path)
    r = c.get("/find_similar_products", params={"product_id": "nope", "num_similar": 3})
    assert r.status_code == 404


def test_find_bad_num_422(sample_ldjson_path):
    c = _client(sample_ldjson_path)
    r = c.get("/find_similar_products", params={"product_id": "p1", "num_similar": 0})
    assert r.status_code == 422
