"""Load and clean the Amazon Fashion .ldjson into a model-ready DataFrame.

The raw data is messy: prices are text with commas and ~10% missing; ~79% of
weights are the sentinel ``999999999``; brand/colour are often absent; colour is
multi-valued (``"black|white"``). This module turns each record into clean,
numeric-ready columns and records *which* values had to be imputed.
"""
from __future__ import annotations

import re

import pandas as pd

# Values that mean "no real weight" in the source data.
_JUNK_WEIGHTS = {"", "0", "nan", "999999999"}
# Capture a leading number and an optional unit, e.g. "1.2 kg", "250 g".
_WEIGHT_RE = re.compile(r"([\d.]+)\s*(kg|g|gram|grams|mg)?", re.IGNORECASE)


def parse_price(value) -> float | None:
    """Parse a price string like ``"1,200.00"`` to a float; None if unparseable/empty."""
    if value is None:
        return None
    s = str(value).replace(",", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_weight_grams(value) -> float | None:
    """Parse a weight string to grams (kg->x1000, mg->/1000).

    Returns None for the junk sentinel, zero, empty, or unparseable values.
    """
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in _JUNK_WEIGHTS:
        return None
    m = _WEIGHT_RE.match(s)
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "g").lower()
    if unit == "kg":
        val *= 1000.0
    elif unit == "mg":
        val /= 1000.0
    return val


def _is_missing(value) -> bool:
    """True for None or a float NaN (how pandas represents an absent cell)."""
    if value is None:
        return True
    # NaN is the only value not equal to itself.
    return isinstance(value, float) and value != value


def _first_url(value) -> str:
    """Return the first URL from a ``|``-joined list, or "" if missing."""
    if _is_missing(value) or not value:
        return ""
    return str(value).split("|")[0].strip()


def _colour_set(value) -> frozenset[str]:
    """Split a ``|``-joined colour string into a lowercase set."""
    if _is_missing(value) or not value:
        return frozenset()
    parts = [p.strip().lower() for p in str(value).split("|")]
    return frozenset(p for p in parts if p)


def _infer_missing_brands(names: pd.Series, brands: pd.Series) -> pd.Series:
    """Fill empty brands from the product name.

    ~27% of rows have no brand in the source data, yet the brand is usually the
    leading word(s) of the name (e.g. "Puma Men's T-Shirt"). We first learn the
    set of brands that ARE present, then for each empty brand pick the *longest*
    known brand the name starts with; if none matches, fall back to the name's
    first word. Names/brands are already lowercased when this is called.
    """
    known = {b for b in brands if b}
    # Longest brands first so "peter england" wins over "peter".
    known_sorted = sorted(known, key=lambda b: len(b), reverse=True)

    def infer(name: str) -> str:
        for b in known_sorted:
            # Match on a word boundary so "max" doesn't match "maximus".
            if name == b or name.startswith(b + " "):
                return b
        first = name.split()
        return first[0] if first else ""

    return pd.Series(
        [b if b else infer(n) for n, b in zip(names, brands)],
        index=brands.index,
    )



def load_products(path: str) -> pd.DataFrame:
    """Load the .ldjson at ``path`` into a cleaned DataFrame indexed by uniq_id."""
    raw = pd.read_json(path, lines=True)

    df = pd.DataFrame(index=pd.Index(raw["uniq_id"], name="uniq_id"))

    df["product_name"] = (
        raw["product_name"].fillna("").astype(str).str.strip().str.lower().values
    )

    if "brand" in raw:
        df["brand"] = raw["brand"].fillna("").astype(str).str.strip().str.lower().values
    else:
        df["brand"] = ""

    # Recover missing brands from the product name (see _infer_missing_brands).
    df["brand"] = _infer_missing_brands(df["product_name"], df["brand"]).values

    if "colour" in raw:
        df["colour_set"] = raw["colour"].apply(_colour_set).values
    else:
        df["colour_set"] = [frozenset()] * len(raw)

    price = raw["sales_price"].apply(parse_price) if "sales_price" in raw else [None] * len(raw)
    df["price"] = pd.to_numeric(pd.Series(list(price), index=df.index), errors="coerce")

    df["rating"] = pd.to_numeric(pd.Series(raw["rating"].values, index=df.index), errors="coerce")

    weight = raw["weight"].apply(parse_weight_grams) if "weight" in raw else [None] * len(raw)
    df["weight_g"] = pd.to_numeric(pd.Series(list(weight), index=df.index), errors="coerce")
    df["weight_known"] = df["weight_g"].notna().astype(int)

    img_col = "medium" if "medium" in raw else ("large" if "large" in raw else None)
    if img_col:
        df["image_url"] = raw[img_col].apply(_first_url).values
    else:
        df["image_url"] = ""

    # Impute missing numerics with the median of the values that *are* present.
    for col in ["price", "rating", "weight_g"]:
        df[col] = df[col].fillna(df[col].median())

    return df
