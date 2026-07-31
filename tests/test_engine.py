"""Tests for the SimilarityEngine and find_similar_products behaviour."""
import pytest

from product_similarity.engine import SimilarityEngine


def _engine(path):
    return SimilarityEngine.from_path(path)


def test_returns_requested_count(sample_ldjson_path):
    eng = _engine(sample_ldjson_path)
    out = eng.find_similar_products("p1", 3)
    assert len(out) == 3


def test_excludes_self(sample_ldjson_path):
    eng = _engine(sample_ldjson_path)
    out = eng.find_similar_products("p1", 5)
    assert "p1" not in out


def test_returns_valid_ids(sample_ldjson_path):
    eng = _engine(sample_ldjson_path)
    out = eng.find_similar_products("p1", 3)
    assert all(pid in eng.df.index for pid in out)


def test_saree_neighbours_are_sarees(sample_ldjson_path):
    eng = _engine(sample_ldjson_path)
    out = eng.find_similar_products("p1", 2)
    # nearest to a Facon saree should include another saree, not a shoe
    assert any(pid in {"p2", "p3", "p8"} for pid in out)


def test_unknown_id_raises(sample_ldjson_path):
    eng = _engine(sample_ldjson_path)
    with pytest.raises(KeyError):
        eng.find_similar_products("does-not-exist", 3)


def test_bad_num_raises(sample_ldjson_path):
    eng = _engine(sample_ldjson_path)
    with pytest.raises(ValueError):
        eng.find_similar_products("p1", 0)
