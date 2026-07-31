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
    # each result carries an image_url field for display
    assert "image_url" in body["results"][0]


def test_gallery_returns_html(sample_ldjson_path):
    c = _client(sample_ldjson_path)
    r = c.get("/gallery", params={"product_id": "p1", "num_similar": 3})
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "QUERY" in r.text  # the query product card is rendered


def test_home_page_lists_products(sample_ldjson_path):
    c = _client(sample_ldjson_path)
    r = c.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    # the storefront renders product cards linking to detail pages
    assert "/product/p1" in r.text


def test_home_search_narrows_results(sample_ldjson_path):
    c = _client(sample_ldjson_path)
    r = c.get("/", params={"q": "saree"})
    assert r.status_code == 200
    # a shoe (p6) should not appear when searching sarees
    assert "/product/p1" in r.text
    assert "/product/p6" not in r.text


def test_product_detail_shows_similar(sample_ldjson_path):
    c = _client(sample_ldjson_path)
    r = c.get("/product/p1")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Similar products" in r.text


def test_product_detail_unknown_id_404(sample_ldjson_path):
    c = _client(sample_ldjson_path)
    r = c.get("/product/nope")
    assert r.status_code == 404
    assert "text/html" in r.headers["content-type"]


def test_find_unknown_id_404(sample_ldjson_path):
    c = _client(sample_ldjson_path)
    r = c.get("/find_similar_products", params={"product_id": "nope", "num_similar": 3})
    assert r.status_code == 404


def test_find_bad_num_422(sample_ldjson_path):
    c = _client(sample_ldjson_path)
    r = c.get("/find_similar_products", params={"product_id": "p1", "num_similar": 0})
    assert r.status_code == 422
