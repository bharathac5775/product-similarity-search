"""Tests for reverse-image search: query with an image VECTOR, get nearest products.

Uses a fake embedder so no torch/network is needed.
"""
import numpy as np

from product_similarity.config import Settings
from product_similarity.data_loader import load_products
from product_similarity.engine import SimilarityEngine


class FakeEmbedder:
    def __init__(self, vectors: np.ndarray):
        self._vectors = vectors.astype("float32")

    def embed_urls(self, urls):
        return self._vectors[: len(urls)]


def test_search_by_image_vector_finds_matching_product(sample_ldjson_path):
    df = load_products(sample_ldjson_path)
    n = len(df)
    dim = 8
    # distinct image vector per product
    vecs = np.eye(n, dim, dtype="float32")

    settings = Settings(use_images=True, image_sample_size=n, w_image=1.0)
    eng = SimilarityEngine(settings, image_embedder=FakeEmbedder(vecs))
    eng.build(df)

    # Query with the exact image vector of product at row 2 -> it should rank #1.
    target_pid = list(df.index)[2]
    out = eng.find_similar_by_image_vector(vecs[2], num_similar=3)
    assert out[0] == target_pid


def test_search_by_image_requires_images_on(sample_ldjson_path):
    # With images OFF there are no image vectors -> clear error, not a crash.
    eng = SimilarityEngine.from_path(sample_ldjson_path, Settings(use_images=False))
    try:
        eng.find_similar_by_image_vector(np.ones(8, dtype="float32"), 3)
        assert False, "expected RuntimeError when images are off"
    except RuntimeError:
        pass
