"""Tests for the display helpers used by the storefront HTML pages.

These are pure ``df`` lookups (no ML): map ids to display dicts, and search the
catalog with paging. They keep the HTML routes thin and testable.
"""
from product_similarity.engine import SimilarityEngine


def _engine(path):
    return SimilarityEngine.from_path(path)


def test_get_products_shape_and_order(sample_ldjson_path):
    eng = _engine(sample_ldjson_path)
    rows = eng.get_products(["p6", "p1"])
    # order is preserved (input order, not df order)
    assert [r["uniq_id"] for r in rows] == ["p6", "p1"]
    r = rows[0]
    for key in ("uniq_id", "product_name", "brand", "price", "rating", "colours", "image_url"):
        assert key in r
    assert isinstance(r["colours"], list)


def test_get_products_skips_unknown_ids(sample_ldjson_path):
    eng = _engine(sample_ldjson_path)
    rows = eng.get_products(["p1", "does-not-exist", "p2"])
    assert [r["uniq_id"] for r in rows] == ["p1", "p2"]


def test_search_filters_by_name_or_brand(sample_ldjson_path):
    eng = _engine(sample_ldjson_path)
    rows, total = eng.search("saree", page=1, per_page=50)
    ids = {r["uniq_id"] for r in rows}
    # p1, p2, p3, p8 are sarees in the fixture
    assert {"p1", "p2", "p3", "p8"}.issubset(ids)
    assert total == len(rows)
    # a shoe should not match "saree"
    assert "p6" not in ids


def test_search_empty_query_returns_full_catalog(sample_ldjson_path):
    eng = _engine(sample_ldjson_path)
    rows, total = eng.search("", page=1, per_page=50)
    assert total == len(eng.df)
    assert len(rows) == len(eng.df)


def test_search_paging(sample_ldjson_path):
    eng = _engine(sample_ldjson_path)
    page1, total = eng.search("", page=1, per_page=3)
    page2, total2 = eng.search("", page=2, per_page=3)
    assert total == total2 == len(eng.df)
    assert len(page1) == 3
    # pages don't overlap
    assert not ({r["uniq_id"] for r in page1} & {r["uniq_id"] for r in page2})
