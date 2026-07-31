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


def test_search_by_image_returns_results(sample_ldjson_path):
    # Build an images-ON engine with a fake embedder (no torch/network).
    import numpy as np
    from product_similarity.config import Settings
    from product_similarity.data_loader import load_products

    df = load_products(sample_ldjson_path)
    n = len(df)
    dim = 8
    vecs = np.eye(n, dim, dtype="float32")

    class FakeEmbedder:
        def __init__(self, v):
            self._v = v.astype("float32")

        def embed_urls(self, urls):
            return self._v[: len(urls)]

        def embed_image(self, data):
            # pretend the uploaded photo matches product row 2
            return self._v[2]

    eng = SimilarityEngine(Settings(use_images=True, image_sample_size=n), image_embedder=FakeEmbedder(vecs))
    eng.build(df)
    c = TestClient(build_app(eng))

    r = c.post(
        "/search_by_image",
        files={"file": ("photo.png", b"fake-image-bytes", "image/png")},
    )
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    # the matching product (row 2) should be linked in the results
    target_pid = list(df.index)[2]
    assert f"/product/{target_pid}" in r.text


def test_search_by_image_when_images_off_is_friendly(sample_ldjson_path):
    # Fixture engine has images OFF -> should not 500; returns a clear message.
    c = _client(sample_ldjson_path)
    r = c.post(
        "/search_by_image",
        files={"file": ("photo.png", b"fake-image-bytes", "image/png")},
    )
    assert r.status_code in (200, 503)
    assert "image" in r.text.lower()


def test_find_unknown_id_404(sample_ldjson_path):
    c = _client(sample_ldjson_path)
    r = c.get("/find_similar_products", params={"product_id": "nope", "num_similar": 3})
    assert r.status_code == 404


def test_find_bad_num_422(sample_ldjson_path):
    c = _client(sample_ldjson_path)
    r = c.get("/find_similar_products", params={"product_id": "p1", "num_similar": 0})
    assert r.status_code == 422
