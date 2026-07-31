"""Regression: click-similarity must ignore images (decoupled retrieval paths).

Clicking a product uses text/attribute similarity ONLY. Uploading a photo uses
CLIP visual similarity ONLY. These two paths must not contaminate each other:
turning images on must NOT change the click-similarity ranking.

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


def test_click_similarity_ignores_images(sample_ldjson_path):
    df = load_products(sample_ldjson_path)
    ids = list(df.index)
    pid = ids[0]  # p1: "Cotton Kalamkari Saree Blue"

    off = SimilarityEngine(Settings(use_images=False))
    off.build(df.copy())

    # Craft image vectors that CONTRADICT text similarity: make the query product
    # (a saree) visually identical to a completely unrelated product (a shoe), and
    # everything else visually distinct. If images leak into click-similarity, the
    # shoe gets pulled up the ranking; text-only, it does not.
    n = len(df)
    dim = 8
    vecs = np.eye(n, dim, dtype="float32")
    shoe_row = ids.index("p7")
    vecs[shoe_row] = vecs[0]  # p7 shares p1's image vector

    # High image weight so any fusion would be clearly visible in the ranking.
    on = SimilarityEngine(
        Settings(use_images=True, image_sample_size=n, w_image=5.0),
        image_embedder=FakeEmbedder(vecs),
    )
    on.build(df.copy())

    # Click-similarity ranking is identical whether images are on or off — the
    # visually-matched shoe must NOT be dragged in by its image.
    assert off.find_similar_products(pid, 5) == on.find_similar_products(pid, 5)

    # Upload (reverse-image) search still works when images are on, and it DOES
    # use the visual signal: querying p1's image returns p1 and its visual twin p7.
    img_hits = on.find_similar_by_image_vector(vecs[0], 3)
    assert img_hits[0] == pid
    assert "p7" in img_hits
