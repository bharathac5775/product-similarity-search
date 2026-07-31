"""Tests that CLIP image features, once fused, actually influence the ranking.

Uses a fake embedder so no network or torch is needed: we hand-craft image
vectors and check that they change who ranks as "similar".
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


def test_image_features_change_ranking(sample_ldjson_path):
    df = load_products(sample_ldjson_path)
    n = len(df)
    dim = 8

    # Give every product a distinct image vector...
    vecs = np.eye(n, dim, dtype="float32")
    # ...except make p6 (a shoe) share p1's (a saree) image vector exactly.
    i_p1 = list(df.index).index("p1")
    i_p6 = list(df.index).index("p6")
    vecs[i_p6] = vecs[i_p1]

    # With a strong image weight, p6 should be pulled into p1's neighbours.
    settings = Settings(use_images=True, image_sample_size=n, w_image=5.0)
    eng = SimilarityEngine(settings, image_embedder=FakeEmbedder(vecs))
    eng.build(df)

    out = eng.find_similar_products("p1", 3)
    assert "p6" in out


def test_images_off_by_default_still_ranks(sample_ldjson_path):
    # Without images, behaviour is unchanged: sarees still cluster.
    eng = SimilarityEngine.from_path(sample_ldjson_path, Settings(use_images=False))
    out = eng.find_similar_products("p1", 2)
    assert any(pid in {"p2", "p3", "p8"} for pid in out)
