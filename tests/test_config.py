"""Tests for configuration defaults and environment overrides."""
from product_similarity.config import Settings, settings


def test_defaults():
    assert settings.text_backend == "tfidf"
    assert settings.use_faiss is False
    assert settings.use_images is False  # off by default (heavy)
    # Data-driven ordering: text dominant, price medium, rating low.
    assert settings.w_text >= settings.w_price >= settings.w_rating


def test_image_defaults():
    assert settings.w_image > 0
    assert settings.image_sample_size > 0


def test_env_override(monkeypatch):
    monkeypatch.setenv("PSS_USE_FAISS", "true")
    monkeypatch.setenv("PSS_TEXT_BACKEND", "embeddings")
    monkeypatch.setenv("PSS_USE_IMAGES", "true")
    monkeypatch.setenv("PSS_IMAGE_SAMPLE_SIZE", "250")
    s = Settings.from_env()
    assert s.use_faiss is True
    assert s.text_backend == "embeddings"
    assert s.use_images is True
    assert s.image_sample_size == 250
