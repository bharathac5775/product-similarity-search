"""Tests that CLIP image features drive the reverse-image (photo upload) search.

Images are deliberately NOT fused into click-similarity — a product's picture must
never change its "similar products" ranking (see test_image_decoupling.py). The
image signal lives only in the separate reverse-image index, which these tests
exercise with hand-crafted vectors (fake embedder, no network or torch).
"""
import numpy as np

from product_similarity.config import Settings
from product_similarity.data_loader import load_products
from product_similarity.engine import SimilarityEngine


class FakeEmbedder:
    """Returns a preset image vector per URL (order matches the sample)."""

    def __init__(self, vectors: np.ndarray):
        self._vectors = vectors.astype("float32")

    def embed_urls(self, urls):
        return self._vectors[: len(urls)]


def test_image_features_drive_reverse_image_search(sample_ldjson_path):
    df = load_products(sample_ldjson_path)
    n = len(df)
    dim = 8

    # Give every product a distinct image vector...
    vecs = np.eye(n, dim, dtype="float32")
    # ...except make p6 (a shoe) share p1's (a saree) image vector exactly.
    i_p1 = list(df.index).index("p1")
    i_p6 = list(df.index).index("p6")
    vecs[i_p6] = vecs[i_p1]

    settings = Settings(use_images=True, image_sample_size=n, w_image=5.0)
    eng = SimilarityEngine(settings, image_embedder=FakeEmbedder(vecs))
    eng.build(df)

    # Reverse-image search DOES use the visual signal: querying p1's image returns
    # p1 and its visual twin p6 at the top.
    hits = eng.find_similar_by_image_vector(vecs[i_p1], 3)
    assert "p1" in hits
    assert "p6" in hits

    # But click-similarity must NOT be perturbed by the shared image vector: the
    # shoe stays out of the saree's text/attribute neighbours.
    click = eng.find_similar_products("p1", 3)
    assert "p6" not in click


def test_images_off_by_default_still_ranks(sample_ldjson_path):
    # Without images, behaviour is unchanged: sarees still cluster.
    eng = SimilarityEngine.from_path(sample_ldjson_path, Settings(use_images=False))
    out = eng.find_similar_products("p1", 2)
    assert any(pid in {"p2", "p3", "p8"} for pid in out)
