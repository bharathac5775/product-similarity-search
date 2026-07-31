"""Tests for the data loader: parsing, imputation, and the missingness flag."""
import math

from product_similarity.data_loader import (
    load_products,
    parse_price,
    parse_weight_grams,
)


def test_parse_price_plain():
    assert parse_price("200.00") == 200.0


def test_parse_price_comma():
    assert parse_price("1,200.00") == 1200.0


def test_parse_price_empty_is_none():
    assert parse_price("") is None


def test_parse_weight_grams_plain():
    assert parse_weight_grams("250 g") == 250.0


def test_parse_weight_grams_kg():
    assert parse_weight_grams("1.2 kg") == 1200.0


def test_parse_weight_junk_is_none():
    assert parse_weight_grams("999999999") is None


def test_load_products_index_and_columns(sample_ldjson_path):
    df = load_products(sample_ldjson_path)
    assert df.index.name == "uniq_id"
    assert "p1" in df.index
    for col in [
        "product_name",
        "brand",
        "colour_set",
        "price",
        "rating",
        "weight_g",
        "weight_known",
        "image_url",
    ]:
        assert col in df.columns


def test_load_products_weight_flag(sample_ldjson_path):
    df = load_products(sample_ldjson_path)
    assert df.loc["p1", "weight_known"] == 0  # junk weight -> guessed
    assert df.loc["p2", "weight_known"] == 1  # real 250 g


def test_load_products_price_imputed(sample_ldjson_path):
    df = load_products(sample_ldjson_path)
    # p4 had empty price -> imputed to median of present prices, not NaN
    assert not math.isnan(df.loc["p4", "price"])


def test_load_products_colour_set(sample_ldjson_path):
    df = load_products(sample_ldjson_path)
    assert df.loc["p4", "colour_set"] == frozenset({"black", "white"})
    assert df.loc["p7", "colour_set"] == frozenset()  # missing colour


def test_load_products_brand_lowercase_and_missing(sample_ldjson_path):
    df = load_products(sample_ldjson_path)
    assert df.loc["p1", "brand"] == "facon"
    assert df.loc["p6", "brand"] == ""  # missing brand
