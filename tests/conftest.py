"""Shared pytest fixtures for the product-similarity test suite."""
import os

import pytest

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.ldjson")


@pytest.fixture
def sample_ldjson_path() -> str:
    """Absolute path to the tiny hand-made sample dataset."""
    return FIXTURE


@pytest.fixture
def sample_df(sample_ldjson_path):
    """The sample dataset, loaded and cleaned via the data loader."""
    from product_similarity.data_loader import load_products

    return load_products(sample_ldjson_path)
