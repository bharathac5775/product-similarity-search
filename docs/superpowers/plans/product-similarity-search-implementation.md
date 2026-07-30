# Product Similarity Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a product-similarity microservice that, given a product `uniq_id`, returns the N most similar Amazon-Fashion products — served via FastAPI, containerised (multi-stage Docker), deployed on Kubernetes (minikube), with an optional FAISS/HNSW fast path and an optional CLIP image-similarity phase.

**Architecture:** Offline at startup: clean the 30k-record `.ldjson`, build one **fused vector** per product (StandardScaler numeric + TF-IDF text + brand/colour match signals), L2-normalise, optionally build a FAISS HNSW index. Online per request: look up the product's vector, rank others by cosine similarity (exact or FAISS), drop self, tie-break, return `uniq_id`s. Layered single-purpose modules under `src/product_similarity/`.

**Tech Stack:** Python 3.11, pandas, numpy, scikit-learn (TF-IDF + StandardScaler + cosine), FastAPI + uvicorn, faiss-cpu (Part 3), sentence-transformers + torch (optional text embeddings + CLIP images), pytest.

## Global Constraints

- Python **3.11** (dev); Docker image base **python:3.10-slim**.
- `find_similar_products(product_id: str, num_similar: int) -> List[str]` — signature must match the exercise exactly.
- The dataset `.ldjson` is **gitignored / not committed** (see `.gitignore`: `data/*.ldjson`). Code must read it from `data/` at runtime; tests must NOT depend on the 74 MB file — use a small in-repo fixture.
- Endpoint: `GET /find_similar_products?product_id=<str>&num_similar=<int>`; plus `GET /health`.
- HTTP errors: **404** unknown product_id, **422** invalid `num_similar` (≤0 / wrong type), **200** success.
- Similarity: **cosine** on **L2-normalised** fused vectors. Numeric prep: **log1p(price)** then StandardScaler; rating & weight down-weighted; `weight_known` missingness flag.
- Weights & feature flags (`TEXT_BACKEND`, `USE_FAISS`) live in `config.py`; defaults data-driven.
- Dockerfile must be **multi-stage** (builder + slim runtime), non-root user, EXPOSE 8000, `CMD uvicorn app:app`.
- No dates anywhere in project files.
- Frequent commits; TDD (test first) for all logic modules.

**Data reality (from profiling 30k records) — implementers must honour:**
- `uniq_id` 100% unique. `product_name` 100%. `rating` 100% but skewed high. `sales_price` ~90% (text, commas). `weight` ~21% real (79% `999999999`; units `g`/`kg`). `brand` ~73% (6,458 distinct). `colour` ~20% (multi-value `a|b`). Median price ≈ ₹590, median real weight ≈ 231 g.

---

## File Structure

```
src/product_similarity/
  __init__.py       # package marker + version
  config.py         # Settings: paths, weights, flags (TEXT_BACKEND, USE_FAISS)
  data_loader.py    # load_products(path) -> clean DataFrame indexed by uniq_id
  features.py       # FeatureBuilder: fit/transform -> fused, L2-normalised matrix
  engine.py         # SimilarityEngine: build(df) + find_similar_products(id, k)
  faiss_index.py    # FaissHNSW wrapper (Part 3): build + search
  images.py         # CLIP image embeddings (last phase)
app.py              # FastAPI app: lifespan builds engine; /find_similar_products, /health
tests/
  conftest.py                 # tiny fixture DataFrame + sample .ldjson lines
  fixtures/sample.ldjson      # ~12 hand-made records covering edge cases
  test_data_loader.py
  test_features.py
  test_engine.py
  test_api.py
  test_faiss_index.py
k8s/
  deployment.yaml
  service.yaml
requirements.txt
Dockerfile          # replaces existing single-stage
README.md           # design writeup (grading)
Makefile            # convenience: install, test, run, docker, k8s (optional helper)
```

---

## Task 1: Project scaffolding, venv, dependencies

**Files:**
- Create: `requirements.txt`, `src/product_similarity/__init__.py`, `tests/__init__.py`, `pytest.ini`
- Create: `.gitignore` additions (venv, __pycache__)

**Interfaces:**
- Produces: installable dev environment; `product_similarity` importable package.

- [ ] **Step 1: Create `requirements.txt`**

```
# Core
pandas==2.2.3
numpy==2.1.3
scikit-learn==1.5.2
# API
fastapi==0.115.6
uvicorn[standard]==0.34.0
# Part 3 (vector search)
faiss-cpu==1.9.0
# Optional text embeddings + image (CLIP) — heavy; used only when flags enabled
sentence-transformers==3.3.1
pillow==11.0.0
requests==2.32.3
# Dev / test
pytest==8.3.4
httpx==0.28.1
```

- [ ] **Step 2: Create package + test markers**

`src/product_similarity/__init__.py`:
```python
"""Product similarity search package."""
__version__ = "0.1.0"
```
`tests/__init__.py`: (empty file)

`pytest.ini`:
```ini
[pytest]
pythonpath = . src
testpaths = tests
```

- [ ] **Step 3: Extend `.gitignore`**

Append:
```
# Python
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 4: Create venv and install**

Run:
```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```
Expected: all install OK (faiss-cpu, sentence-transformers may take a minute).

- [ ] **Step 5: Verify imports**

Run:
```bash
.venv/bin/python -c "import pandas, numpy, sklearn, fastapi, faiss; print('deps ok')"
```
Expected: `deps ok`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/product_similarity/__init__.py tests/__init__.py pytest.ini .gitignore
git commit -m "chore: project scaffolding, deps, venv config"
```

---

## Task 2: Test fixtures (tiny dataset covering edge cases)

**Files:**
- Create: `tests/fixtures/sample.ldjson`, `tests/conftest.py`

**Interfaces:**
- Produces: pytest fixture `sample_ldjson_path` (str) and `sample_df` (cleaned DataFrame) reused by later tests. The fixture file intentionally contains: real weight, junk weight `999999999`, kg weight, missing price, missing brand, missing colour, multi-colour.

- [ ] **Step 1: Create `tests/fixtures/sample.ldjson`** (one JSON object per line)

```json
{"uniq_id":"p1","product_name":"Cotton Kalamkari Saree Blue","brand":"Facon","sales_price":"200.00","weight":"999999999","rating":"5.0","colour":"blue","medium":"http://img/p1.jpg"}
{"uniq_id":"p2","product_name":"Cotton Kalamkari Saree Red","brand":"Facon","sales_price":"210.00","weight":"250 g","rating":"4.8","colour":"red","medium":"http://img/p2.jpg"}
{"uniq_id":"p3","product_name":"Silk Saree Green","brand":"Soch","sales_price":"1,200.00","weight":"1.2 kg","rating":"4.5","colour":"green","medium":"http://img/p3.jpg"}
{"uniq_id":"p4","product_name":"Mens Slim Fit T-Shirt Black","brand":"Max","sales_price":"","weight":"180 g","rating":"4.0","colour":"black|white","medium":"http://img/p4.jpg"}
{"uniq_id":"p5","product_name":"Mens Round Neck T-Shirt White","brand":"Max","sales_price":"499.00","weight":"999999999","rating":"3.9","colour":"white","medium":"http://img/p5.jpg"}
{"uniq_id":"p6","product_name":"Running Shoes Grey","brand":"","sales_price":"2500.00","weight":"800 g","rating":"4.2","colour":"grey","medium":"http://img/p6.jpg"}
{"uniq_id":"p7","product_name":"Leather Formal Shoes Brown","brand":"Bata","sales_price":"3000.00","weight":"900 g","rating":"3.5","medium":"http://img/p7.jpg"}
{"uniq_id":"p8","product_name":"Cotton Saree Yellow Handblock","brand":"Facon","sales_price":"250.00","weight":"260 g","rating":"5.0","colour":"yellow","medium":"http://img/p8.jpg"}
```

- [ ] **Step 2: Create `tests/conftest.py`**

```python
import os
import pytest

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.ldjson")


@pytest.fixture
def sample_ldjson_path() -> str:
    return FIXTURE


@pytest.fixture
def sample_df(sample_ldjson_path):
    from product_similarity.data_loader import load_products
    return load_products(sample_ldjson_path)
```

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/sample.ldjson tests/conftest.py
git commit -m "test: add fixture dataset covering data edge cases"
```

---

## Task 3: Data loader (clean/parse/impute)

**Files:**
- Create: `src/product_similarity/data_loader.py`
- Test: `tests/test_data_loader.py`

**Interfaces:**
- Produces:
  - `parse_price(value) -> float | None` — strip commas, to float, None if unparseable/empty.
  - `parse_weight_grams(value) -> float | None` — number+unit to grams; None for `999999999`/`0`/empty/unparseable.
  - `load_products(path: str) -> pandas.DataFrame` — indexed by `uniq_id`, columns: `product_name`(str, lowercased), `brand`(str, lowercased, "" if missing), `colour_set`(frozenset[str]), `price`(float, median-imputed), `rating`(float), `weight_g`(float, median-imputed), `weight_known`(int 0/1), `image_url`(str, first URL or ""). Medians computed from the file's own non-missing values.
- Consumes: nothing.

- [ ] **Step 1: Write failing tests**

```python
import math
from product_similarity.data_loader import parse_price, parse_weight_grams, load_products


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
    for col in ["product_name","brand","colour_set","price","rating","weight_g","weight_known","image_url"]:
        assert col in df.columns

def test_load_products_weight_flag(sample_ldjson_path):
    df = load_products(sample_ldjson_path)
    assert df.loc["p1","weight_known"] == 0   # junk weight
    assert df.loc["p2","weight_known"] == 1   # real 250 g

def test_load_products_price_imputed(sample_ldjson_path):
    df = load_products(sample_ldjson_path)
    # p4 had empty price -> imputed to median of present prices, not NaN
    assert not math.isnan(df.loc["p4","price"])

def test_load_products_colour_set(sample_ldjson_path):
    df = load_products(sample_ldjson_path)
    assert df.loc["p4","colour_set"] == frozenset({"black","white"})
    assert df.loc["p7","colour_set"] == frozenset()   # missing colour

def test_load_products_brand_lowercase_and_missing(sample_ldjson_path):
    df = load_products(sample_ldjson_path)
    assert df.loc["p1","brand"] == "facon"
    assert df.loc["p6","brand"] == ""   # missing brand
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_data_loader.py -v`
Expected: FAIL (module/functions not defined).

- [ ] **Step 3: Implement `data_loader.py`**

```python
"""Load and clean the Amazon Fashion .ldjson into a model-ready DataFrame."""
from __future__ import annotations

import re
import pandas as pd

_JUNK_WEIGHTS = {"", "0", "nan", "999999999"}
_WEIGHT_RE = re.compile(r"([\d.]+)\s*(kg|g|gram|grams|mg)?", re.IGNORECASE)


def parse_price(value) -> float | None:
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


def _first_url(value) -> str:
    if not value:
        return ""
    return str(value).split("|")[0].strip()


def _colour_set(value) -> frozenset[str]:
    if not value:
        return frozenset()
    parts = [p.strip().lower() for p in str(value).split("|")]
    return frozenset(p for p in parts if p)


def load_products(path: str) -> pd.DataFrame:
    raw = pd.read_json(path, lines=True)

    df = pd.DataFrame(index=raw["uniq_id"])
    df.index.name = "uniq_id"

    df["product_name"] = raw["product_name"].fillna("").astype(str).str.strip().str.lower().values
    df["brand"] = (
        raw["brand"].fillna("").astype(str).str.strip().str.lower().values
        if "brand" in raw else ""
    )
    df["colour_set"] = (
        raw["colour"].apply(_colour_set).values if "colour" in raw
        else [frozenset()] * len(raw)
    )

    price = raw["sales_price"].apply(parse_price) if "sales_price" in raw else pd.Series([None] * len(raw))
    price = pd.to_numeric(price.values, errors="coerce")
    df["price"] = pd.Series(price, index=df.index)

    df["rating"] = pd.to_numeric(raw["rating"], errors="coerce").values

    wt = raw["weight"].apply(parse_weight_grams) if "weight" in raw else pd.Series([None] * len(raw))
    wt = pd.to_numeric(wt.values, errors="coerce")
    df["weight_g"] = pd.Series(wt, index=df.index)
    df["weight_known"] = df["weight_g"].notna().astype(int)

    img_col = "medium" if "medium" in raw else ("large" if "large" in raw else None)
    df["image_url"] = raw[img_col].apply(_first_url).values if img_col else ""

    # Impute numerics with median of present values.
    for col in ["price", "rating", "weight_g"]:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    return df
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_data_loader.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/product_similarity/data_loader.py tests/test_data_loader.py
git commit -m "feat: data loader with price/weight parsing, imputation, missingness flag"
```

---

## Task 4: Config (weights, paths, flags)

**Files:**
- Create: `src/product_similarity/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings` dataclass instance `settings` with fields:
  - `data_path: str` (default `data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson`)
  - `w_text: float=1.0`, `w_price: float=0.5`, `w_rating: float=0.2`, `w_weight: float=0.2`, `w_brand: float=0.4`, `w_colour: float=0.3` (data-driven: text high, price medium, rating/weight low)
  - `text_backend: str="tfidf"` (`"tfidf"` | `"embeddings"`), `use_faiss: bool=False`, `tfidf_max_features: int=5000`
  - env overrides via `PSS_*` variables; a `from_env()` classmethod.

- [ ] **Step 1: Write failing tests**

```python
import os
from product_similarity.config import Settings, settings


def test_defaults():
    assert settings.text_backend == "tfidf"
    assert settings.use_faiss is False
    assert settings.w_text >= settings.w_price >= settings.w_rating

def test_env_override(monkeypatch):
    monkeypatch.setenv("PSS_USE_FAISS", "true")
    monkeypatch.setenv("PSS_TEXT_BACKEND", "embeddings")
    s = Settings.from_env()
    assert s.use_faiss is True
    assert s.text_backend == "embeddings"
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `config.py`**

```python
"""Central configuration: paths, blend weights, feature flags."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

_DEFAULT_DATA = (
    "data/marketing_sample_for_amazon_com-amazon_fashion_products"
    "__20200201_20200430__30k_data.ldjson"
)


def _as_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_path: str = _DEFAULT_DATA
    # Blend weights (data-driven: text dominant, price medium, rating/weight low).
    w_text: float = 1.0
    w_price: float = 0.5
    w_rating: float = 0.2
    w_weight: float = 0.2
    w_brand: float = 0.4
    w_colour: float = 0.3
    # Feature flags.
    text_backend: str = "tfidf"   # "tfidf" | "embeddings"
    use_faiss: bool = False
    tfidf_max_features: int = 5000

    @classmethod
    def from_env(cls) -> "Settings":
        base = cls()
        return replace(
            base,
            data_path=os.getenv("PSS_DATA_PATH", base.data_path),
            text_backend=os.getenv("PSS_TEXT_BACKEND", base.text_backend),
            use_faiss=_as_bool(os.getenv("PSS_USE_FAISS", str(base.use_faiss))),
            tfidf_max_features=int(os.getenv("PSS_TFIDF_MAX_FEATURES", base.tfidf_max_features)),
        )


settings = Settings.from_env()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/product_similarity/config.py tests/test_config.py
git commit -m "feat: config with data-driven weights and feature flags"
```

---

## Task 5: Feature builder (fused, L2-normalised vectors)

**Files:**
- Create: `src/product_similarity/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `Settings` (weights, `text_backend`, `tfidf_max_features`); cleaned DataFrame from `load_products`.
- Produces: class `FeatureBuilder`:
  - `__init__(self, settings)`
  - `fit_transform(self, df) -> scipy.sparse | numpy.ndarray` — returns an L2-normalised matrix, one row per product (row order == df index order). Blocks: numeric `[log1p(price), rating, weight_g, weight_known]` via StandardScaler × block weights; text `product_name` via TfidfVectorizer × `w_text`; categorical brand one-hot-ish is NOT built here (handled as post-filter in engine to avoid 6k columns) — instead a small `brand`/`colour` numeric proxy is skipped; brand/colour handled in engine re-rank (see Task 6). Final matrix L2-normalised (`sklearn.preprocessing.normalize`).
  - Stores fitted `self.scaler`, `self.vectorizer` for reuse.

Design note for implementer: to keep everything in ONE fused vector while avoiding 6,458 brand columns, the fused vector = weighted[ numeric-block ‖ tfidf-text-block ]. Brand-match and colour-overlap are applied as an **additive re-rank bonus** in the engine (Task 6), NOT as vector columns. This preserves "single fused vector for cosine/FAISS" while honouring the match-based decision. Document this trade-off in README.

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from scipy.sparse import issparse
from sklearn.preprocessing import normalize
from product_similarity.config import Settings
from product_similarity.features import FeatureBuilder


def test_fit_transform_shape(sample_df):
    fb = FeatureBuilder(Settings())
    X = fb.fit_transform(sample_df)
    assert X.shape[0] == len(sample_df)     # one row per product
    assert X.shape[1] > 4                    # numeric + many tfidf cols

def test_rows_are_l2_normalised(sample_df):
    fb = FeatureBuilder(Settings())
    X = fb.fit_transform(sample_df)
    arr = X.toarray() if issparse(X) else X
    norms = np.linalg.norm(arr, axis=1)
    # rows with any signal should be ~unit length
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-6)

def test_similar_sarees_closer_than_shoe(sample_df):
    # p1 and p8 are both Facon cotton sarees; p6 is running shoes
    fb = FeatureBuilder(Settings())
    X = fb.fit_transform(sample_df)
    arr = X.toarray() if issparse(X) else X
    idx = {pid: i for i, pid in enumerate(sample_df.index)}
    def cos(a, b): return float(arr[idx[a]] @ arr[idx[b]])
    assert cos("p1", "p8") > cos("p1", "p6")
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_features.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `features.py`**

```python
"""Build one fused, L2-normalised vector per product."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize


class FeatureBuilder:
    def __init__(self, settings):
        self.settings = settings
        self.scaler: StandardScaler | None = None
        self.vectorizer: TfidfVectorizer | None = None

    def _numeric_block(self, df) -> np.ndarray:
        price = np.log1p(df["price"].to_numpy(dtype=float))
        rating = df["rating"].to_numpy(dtype=float)
        weight = df["weight_g"].to_numpy(dtype=float)
        weight_known = df["weight_known"].to_numpy(dtype=float)
        raw = np.column_stack([price, rating, weight, weight_known])
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(raw)
        s = self.settings
        col_weights = np.array([s.w_price, s.w_rating, s.w_weight, s.w_weight])
        return scaled * col_weights

    def _text_block(self, df):
        s = self.settings
        self.vectorizer = TfidfVectorizer(max_features=s.tfidf_max_features)
        tfidf = self.vectorizer.fit_transform(df["product_name"].tolist())
        return tfidf * s.w_text

    def fit_transform(self, df):
        numeric = csr_matrix(self._numeric_block(df))
        text = self._text_block(df)
        fused = hstack([numeric, text]).tocsr()
        return normalize(fused, norm="l2", axis=1)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_features.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/product_similarity/features.py tests/test_features.py
git commit -m "feat: fused L2-normalised feature vectors (numeric + tfidf)"
```

---

## Task 6: Similarity engine + `find_similar_products`

**Files:**
- Create: `src/product_similarity/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `FeatureBuilder`, `load_products`, `Settings`.
- Produces: class `SimilarityEngine`:
  - `classmethod from_path(cls, path, settings=None) -> SimilarityEngine` — loads + builds.
  - `build(self, df) -> None` — fits features, stores matrix + index mapping.
  - `find_similar_products(self, product_id: str, num_similar: int) -> list[str]` — cosine top-N (exact), drop self, add brand-match + colour-overlap bonus, tie-break by (similarity, rating, price). Raises `KeyError` if id unknown; raises `ValueError` if `num_similar <= 0`.
  - Module-level `find_similar_products(product_id, num_similar)` bound to a lazily-built default engine (matches exercise's free-function signature).

- [ ] **Step 1: Write failing tests**

```python
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
    # nearest to a Facon saree should include another saree (p2/p3/p8), not a shoe
    assert any(pid in {"p2","p3","p8"} for pid in out)

def test_unknown_id_raises(sample_ldjson_path):
    eng = _engine(sample_ldjson_path)
    with pytest.raises(KeyError):
        eng.find_similar_products("does-not-exist", 3)

def test_bad_num_raises(sample_ldjson_path):
    eng = _engine(sample_ldjson_path)
    with pytest.raises(ValueError):
        eng.find_similar_products("p1", 0)
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_engine.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `engine.py`**

```python
"""SimilarityEngine: build vectors and answer nearest-neighbour queries."""
from __future__ import annotations

import numpy as np
from scipy.sparse import issparse

from .config import Settings, settings as default_settings
from .data_loader import load_products
from .features import FeatureBuilder


class SimilarityEngine:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or default_settings
        self.df = None
        self.matrix = None
        self._pos = {}          # uniq_id -> row index
        self._ids = []          # row index -> uniq_id
        self._fb = None

    @classmethod
    def from_path(cls, path: str, settings: Settings | None = None) -> "SimilarityEngine":
        eng = cls(settings)
        eng.build(load_products(path))
        return eng

    def build(self, df) -> None:
        self.df = df
        self._fb = FeatureBuilder(self.settings)
        self.matrix = self._fb.fit_transform(df)
        self._ids = list(df.index)
        self._pos = {pid: i for i, pid in enumerate(self._ids)}

    def _cosine_scores(self, row_idx: int) -> np.ndarray:
        # rows are L2-normalised, so cosine == dot product
        q = self.matrix[row_idx]
        scores = self.matrix @ q.T
        return scores.toarray().ravel() if issparse(scores) else np.asarray(scores).ravel()

    def find_similar_products(self, product_id: str, num_similar: int) -> list[str]:
        if num_similar <= 0:
            raise ValueError("num_similar must be a positive integer")
        if product_id not in self._pos:
            raise KeyError(product_id)

        i = self._pos[product_id]
        sims = self._cosine_scores(i)

        s = self.settings
        base_colours = self.df.iloc[i]["colour_set"]
        base_brand = self.df.iloc[i]["brand"]

        # additive re-rank bonuses (brand match + colour overlap)
        ratings = self.df["rating"].to_numpy(dtype=float)
        prices = self.df["price"].to_numpy(dtype=float)
        brands = self.df["brand"].to_numpy()
        colour_sets = self.df["colour_set"].to_numpy()

        candidates = []
        for j in range(len(self._ids)):
            if j == i:
                continue
            score = float(sims[j])
            if base_brand and brands[j] == base_brand:
                score += s.w_brand
            if base_colours and colour_sets[j]:
                overlap = len(base_colours & colour_sets[j]) / len(base_colours | colour_sets[j])
                score += s.w_colour * overlap
            # tie-break key: higher score, then higher rating, then lower price
            candidates.append((score, ratings[j], -prices[j], self._ids[j]))

        candidates.sort(reverse=True)
        return [pid for _, _, _, pid in candidates[:num_similar]]


_default_engine: SimilarityEngine | None = None


def find_similar_products(product_id: str, num_similar: int) -> list[str]:
    global _default_engine
    if _default_engine is None:
        _default_engine = SimilarityEngine.from_path(default_settings.data_path)
    return _default_engine.find_similar_products(product_id, num_similar)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_engine.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/product_similarity/engine.py tests/test_engine.py
git commit -m "feat: SimilarityEngine with cosine ranking, brand/colour re-rank, tie-break"
```

---

## Task 7: FastAPI service + error handling

**Files:**
- Create: `app.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `SimilarityEngine`, `Settings`.
- Produces: FastAPI `app` with:
  - startup (lifespan) building a single `SimilarityEngine` from `settings.data_path` into `app.state.engine`.
  - `GET /health` -> `{"status":"ok"}`.
  - `GET /find_similar_products?product_id&num_similar` -> `{"product_id":..., "num_similar":..., "results":[{"uniq_id","product_name","brand","score"?}...]}` (list of ids at minimum). 404 unknown id, 422 invalid num_similar.
- For tests, expose a helper `build_app(engine)` to inject a fixture engine (avoids loading the 74MB file).

- [ ] **Step 1: Write failing tests**

```python
from fastapi.testclient import TestClient
from product_similarity.engine import SimilarityEngine
from app import build_app


def _client(sample_ldjson_path):
    eng = SimilarityEngine.from_path(sample_ldjson_path)
    return TestClient(build_app(eng))


def test_health(sample_ldjson_path):
    c = _client(sample_ldjson_path)
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

def test_find_ok(sample_ldjson_path):
    c = _client(sample_ldjson_path)
    r = c.get("/find_similar_products", params={"product_id":"p1","num_similar":3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 3

def test_find_unknown_id_404(sample_ldjson_path):
    c = _client(sample_ldjson_path)
    r = c.get("/find_similar_products", params={"product_id":"nope","num_similar":3})
    assert r.status_code == 404

def test_find_bad_num_422(sample_ldjson_path):
    c = _client(sample_ldjson_path)
    r = c.get("/find_similar_products", params={"product_id":"p1","num_similar":0})
    assert r.status_code == 422
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_api.py -v`
Expected: FAIL (app not found).

- [ ] **Step 3: Implement `app.py`**

```python
"""FastAPI microservice wrapping the product similarity engine."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from product_similarity.config import settings
from product_similarity.engine import SimilarityEngine


def build_app(engine: SimilarityEngine) -> FastAPI:
    app = FastAPI(title="Product Similarity Search")
    app.state.engine = engine

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/find_similar_products")
    def find_similar_products(
        product_id: str = Query(..., min_length=1),
        num_similar: int = Query(..., gt=0),
    ):
        eng: SimilarityEngine = app.state.engine
        try:
            ids = eng.find_similar_products(product_id, num_similar)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"product_id '{product_id}' not found")
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        results = [
            {
                "uniq_id": pid,
                "product_name": eng.df.loc[pid, "product_name"],
                "brand": eng.df.loc[pid, "brand"],
            }
            for pid in ids
        ]
        return {"product_id": product_id, "num_similar": num_similar, "results": results}

    return app


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = SimilarityEngine.from_path(settings.data_path)
    yield


# Production app: builds the engine from the real dataset at startup.
app = FastAPI(title="Product Similarity Search", lifespan=lifespan)


@app.get("/health")
def _health():
    return {"status": "ok"}


@app.get("/find_similar_products")
def _find(product_id: str = Query(..., min_length=1), num_similar: int = Query(..., gt=0)):
    eng: SimilarityEngine = app.state.engine
    try:
        ids = eng.find_similar_products(product_id, num_similar)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"product_id '{product_id}' not found")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    results = [
        {"uniq_id": pid, "product_name": eng.df.loc[pid, "product_name"], "brand": eng.df.loc[pid, "brand"]}
        for pid in ids
    ]
    return {"product_id": product_id, "num_similar": num_similar, "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_api.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite + commit**

```bash
.venv/bin/pytest -v
git add app.py tests/test_api.py
git commit -m "feat: FastAPI service with health + find_similar_products, 404/422 handling"
```

---

## Task 8: Multi-stage Dockerfile + requirements

**Files:**
- Modify: `Dockerfile` (replace single-stage with multi-stage)
- Create: `.dockerignore`

**Interfaces:**
- Produces: a container image serving `app:app` on port 8000, non-root, deps installed in a venv copied from a builder stage. Data file expected at `/app/data/...ldjson` (copied in via build context OR mounted; document both).

- [ ] **Step 1: Create `.dockerignore`**

```
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/
.git/
docs/
tests/
*.md
```

- [ ] **Step 2: Replace `Dockerfile` with multi-stage build**

```dockerfile
# syntax=docker/dockerfile:1

# ---- Stage 1: builder ----
FROM python:3.10-slim AS builder
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: runtime ----
FROM python:3.10-slim AS runtime
WORKDIR /app
# copy the prebuilt virtualenv only (no build tools shipped)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
# app code + data
COPY src/ ./src/
COPY app.py ./
COPY data/ ./data/
# non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Build the image**

Run:
```bash
docker build -t product-similarity:latest .
```
Expected: build succeeds; final image based on slim runtime.

- [ ] **Step 4: Smoke-run the container**

Run:
```bash
docker run --rm -d -p 8000:8000 --name pss product-similarity:latest
sleep 20
curl -s localhost:8000/health
curl -s "localhost:8000/find_similar_products?product_id=<REAL_ID>&num_similar=3"
docker stop pss
```
Expected: `{"status":"ok"}` and a JSON results list. (Get a REAL_ID via `head -1 data/*.ldjson | python -c "import sys,json;print(json.loads(sys.stdin.read())['uniq_id'])"`.)

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "build: multi-stage Dockerfile (builder + slim non-root runtime)"
```

---

## Task 9: Kubernetes manifests + minikube deploy

**Files:**
- Create: `k8s/deployment.yaml`, `k8s/service.yaml`

**Interfaces:**
- Produces: a Deployment (1 replica, liveness+readiness probes on `/health`, resource requests/limits) and a Service (NodePort) exposing port 8000. Uses the locally built image via minikube's Docker daemon.

- [ ] **Step 1: Create `k8s/deployment.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: product-similarity
  labels:
    app: product-similarity
spec:
  replicas: 1
  selector:
    matchLabels:
      app: product-similarity
  template:
    metadata:
      labels:
        app: product-similarity
    spec:
      containers:
        - name: product-similarity
          image: product-similarity:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 20
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 40
            periodSeconds: 20
          resources:
            requests:
              cpu: "250m"
              memory: "512Mi"
            limits:
              cpu: "1"
              memory: "1536Mi"
```

- [ ] **Step 2: Create `k8s/service.yaml`**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: product-similarity
  labels:
    app: product-similarity
spec:
  type: NodePort
  selector:
    app: product-similarity
  ports:
    - port: 8000
      targetPort: 8000
      protocol: TCP
```

- [ ] **Step 3: Start minikube and build image into its daemon**

Run:
```bash
minikube start
eval $(minikube -p minikube docker-env)
docker build -t product-similarity:latest .
```
Expected: minikube running; image built inside minikube's Docker.

- [ ] **Step 4: Deploy and verify**

Run:
```bash
kubectl apply -f k8s/
kubectl rollout status deployment/product-similarity --timeout=180s
kubectl get pods -l app=product-similarity
minikube service product-similarity --url
```
Then `curl <url>/health` → `{"status":"ok"}`.
Expected: pod Ready, health OK. (Reset docker env after: `eval $(minikube docker-env -u)`.)

- [ ] **Step 5: Commit**

```bash
git add k8s/deployment.yaml k8s/service.yaml
git commit -m "deploy: kubernetes manifests with health probes and resource limits"
```

---

## Task 10: FAISS HNSW fast path (Part 3)

**Files:**
- Create: `src/product_similarity/faiss_index.py`
- Modify: `src/product_similarity/engine.py` (use FAISS when `settings.use_faiss`)
- Test: `tests/test_faiss_index.py`

**Interfaces:**
- Consumes: the dense fused matrix (FAISS needs dense float32; convert from sparse). L2-normalised rows → use `IndexHNSWFlat` with inner-product (== cosine).
- Produces: class `FaissHNSW`:
  - `build(self, matrix_dense: np.ndarray) -> None`
  - `search(self, row_vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]` (scores, indices)
- Engine: when `use_faiss`, ranking uses FAISS to get top candidates then applies the same brand/colour re-rank + tie-break.

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from product_similarity.engine import SimilarityEngine
from product_similarity.config import Settings


def test_faiss_agrees_with_exact(sample_ldjson_path):
    exact = SimilarityEngine.from_path(sample_ldjson_path, Settings(use_faiss=False))
    faiss_eng = SimilarityEngine.from_path(sample_ldjson_path, Settings(use_faiss=True))
    a = set(exact.find_similar_products("p1", 3))
    b = set(faiss_eng.find_similar_products("p1", 3))
    # allow small approximation slack: at least 2 of 3 overlap on a tiny set
    assert len(a & b) >= 2
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_faiss_index.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `faiss_index.py`**

```python
"""FAISS HNSW index for approximate nearest-neighbour search (Part 3).

Reference: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor
search using Hierarchical Navigable Small World graphs", IEEE TPAMI 2018
(arXiv:1603.09320). HNSW chosen over IVF/Annoy: no training step, high recall at
low latency, incremental inserts, strong performance at mid-to-large scale.
"""
from __future__ import annotations

import faiss
import numpy as np


class FaissHNSW:
    def __init__(self, m: int = 32, ef_construction: int = 200, ef_search: int = 64):
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index = None

    def build(self, matrix_dense: np.ndarray) -> None:
        x = np.ascontiguousarray(matrix_dense.astype("float32"))
        dim = x.shape[1]
        index = faiss.IndexHNSWFlat(dim, self.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        index.add(x)
        self.index = index

    def search(self, row_vector: np.ndarray, k: int):
        q = np.ascontiguousarray(row_vector.astype("float32")).reshape(1, -1)
        scores, idx = self.index.search(q, k)
        return scores[0], idx[0]
```

- [ ] **Step 4: Wire into `engine.py`**

In `SimilarityEngine.build`, after building `self.matrix`, add:
```python
        self._faiss = None
        if self.settings.use_faiss:
            from .faiss_index import FaissHNSW
            dense = self.matrix.toarray() if issparse(self.matrix) else np.asarray(self.matrix)
            self._dense = dense
            self._faiss = FaissHNSW()
            self._faiss.build(dense)
```
And in `find_similar_products`, replace the exact scoring when FAISS is on:
```python
        if getattr(self, "_faiss", None) is not None:
            # over-fetch to leave room for self-exclusion + re-rank
            k = min(len(self._ids), max(num_similar * 5, num_similar + 1))
            _, idxs = self._faiss.search(self._dense[i], k)
            neigh = [j for j in idxs if j != i]
        else:
            sims = self._cosine_scores(i)
            neigh = [j for j in range(len(self._ids)) if j != i]
```
Then compute `score` for `j in neigh` (use `float(self._dense[i] @ self._dense[j])` in FAISS mode, else `sims[j]`), apply brand/colour bonus + tie-break as before.

Implementer note: refactor the candidate loop to iterate over `neigh` and pick the score source based on FAISS mode. Keep behaviour identical otherwise.

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/test_faiss_index.py tests/test_engine.py -v`
Expected: PASS (both exact and FAISS paths).

- [ ] **Step 6: Commit**

```bash
git add src/product_similarity/faiss_index.py src/product_similarity/engine.py tests/test_faiss_index.py
git commit -m "feat: FAISS HNSW approximate search path (Part 3) with exact-agreement test"
```

---

## Task 11: README (design writeup — grading)

**Files:**
- Create/replace: `README.md`

**Interfaces:** none (documentation). Must cover, in beginner-clear prose: problem framing (unsupervised KNN, not regression/trees); data profiling + cleaning decisions (log-price, weight_known flag, brand/colour match); fused-vector architecture; cosine-vs-Euclidean; TF-IDF vs embeddings trade-off; how to run locally / via Docker / on minikube (exact commands); FAISS HNSW + paper citation + why HNSW; image phase note; testing; how to get the dataset (Kaggle link) since it's gitignored.

- [ ] **Step 1: Write `README.md`** covering all sections above with runnable commands (`pip install -r requirements.txt`, `uvicorn app:app`, `docker build/run`, `minikube` flow, `pytest`).

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with design reasoning, run/deploy instructions, FAISS reference"
```

---

## Task 12: CLIP image similarity (optional, LAST phase)

**Files:**
- Create: `src/product_similarity/images.py`
- Modify: `engine.py` (optional image block behind `PSS_USE_IMAGES`)
- Test: `tests/test_images.py` (mock downloads; no network in CI)

**Interfaces:**
- Produces: `ImageEmbedder.embed_urls(urls: list[str]) -> np.ndarray` using CLIP (`sentence-transformers` `clip-ViT-B-32`), with graceful skip for dead URLs (zero vector fallback). Runs on a sample of products; documented as scale-ready but sampled.

- [ ] **Step 1: Write failing test (mocked)**

```python
import numpy as np
from unittest.mock import patch
from product_similarity.images import ImageEmbedder


def test_dead_url_returns_zero_vector():
    emb = ImageEmbedder()
    with patch.object(emb, "_fetch", return_value=None):
        vec = emb.embed_urls(["http://dead/url.jpg"])
    assert vec.shape[0] == 1
    assert np.allclose(vec[0], 0.0)
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_images.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `images.py`**

```python
"""Optional CLIP image embeddings (last phase). Scale-ready but sampled."""
from __future__ import annotations

import io
import numpy as np
import requests
from PIL import Image


class ImageEmbedder:
    def __init__(self, model_name: str = "clip-ViT-B-32", dim: int = 512):
        self.model_name = model_name
        self.dim = dim
        self._model = None

    def _model_lazy(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _fetch(self, url: str):
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            return None

    def embed_urls(self, urls: list[str]) -> np.ndarray:
        imgs, keep = [], []
        for k, u in enumerate(urls):
            img = self._fetch(u)
            if img is not None:
                imgs.append(img); keep.append(k)
        out = np.zeros((len(urls), self.dim), dtype="float32")
        if imgs:
            vecs = self._model_lazy().encode(imgs, convert_to_numpy=True)
            for pos, k in enumerate(keep):
                out[k] = vecs[pos]
        return out
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_images.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/product_similarity/images.py tests/test_images.py
git commit -m "feat: optional CLIP image embeddings with dead-URL fallback (last phase)"
```

---

## Final verification

- [ ] Run full suite: `.venv/bin/pytest -v` → all pass.
- [ ] Local run: `.venv/bin/uvicorn app:app` → hit `/health` and `/find_similar_products`.
- [ ] Docker: build + run + curl.
- [ ] minikube: apply + rollout + curl.
- [ ] Push: `git push`.
