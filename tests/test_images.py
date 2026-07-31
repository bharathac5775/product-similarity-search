"""Tests for optional CLIP image embeddings (mocked; no network, no torch needed)."""
from unittest.mock import patch

import numpy as np

from product_similarity.images import ImageEmbedder


def test_dead_url_returns_zero_vector():
    emb = ImageEmbedder()
    # Simulate every download failing (dead URL) -> graceful zero-vector fallback.
    with patch.object(emb, "_fetch", return_value=None):
        vec = emb.embed_urls(["http://dead/url.jpg"])
    assert vec.shape[0] == 1
    assert np.allclose(vec[0], 0.0)
