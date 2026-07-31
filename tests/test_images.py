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


def test_embed_image_bytes_returns_vector():
    emb = ImageEmbedder(dim=512)
    fake_img = object()  # stand-in PIL image
    fake_vec = np.ones((1, 512), dtype="float32")

    class FakeModel:
        def encode(self, images, convert_to_numpy=True):
            assert images == [fake_img]
            return fake_vec

    with patch.object(emb, "_open", return_value=fake_img), patch.object(
        emb, "_model_lazy", return_value=FakeModel()
    ):
        vec = emb.embed_image(b"\x89PNG-fake-bytes")
    assert vec.shape == (512,)
    assert np.allclose(vec, 1.0)


def test_embed_image_bad_bytes_returns_none():
    emb = ImageEmbedder()
    # Undecodable bytes -> _open returns None -> embed_image returns None (no crash).
    with patch.object(emb, "_open", return_value=None):
        assert emb.embed_image(b"not-an-image") is None

