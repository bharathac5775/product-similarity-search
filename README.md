# Product Similarity Search

A microservice that, given a product's `uniq_id`, returns the *N* most similar
products from the Amazon Fashion 2020 dataset (~30,000 items). It exposes a
FastAPI endpoint, ships as a lean multi-stage Docker image, deploys to Kubernetes,
offers an optional **FAISS/HNSW** fast path for large-scale search, and an optional
**CLIP** image-similarity phase.

> This document explains not just *how* to run the project, but *why* each design
> decision was made — the reasoning and trade-offs behind the approach.

---

## Table of contents

1. [Problem framing: why this is K-Nearest-Neighbors, not regression](#1-problem-framing)
2. [The data, and how we clean it](#2-the-data-and-how-we-clean-it)
3. [How we measure similarity](#3-how-we-measure-similarity)
4. [Architecture](#4-architecture)
5. [Fast large-scale search: FAISS / HNSW](#5-fast-large-scale-search-faiss--hnsw)
6. [Optional: CLIP image similarity (multimodal)](#6-optional-clip-image-similarity-multimodal)
7. [How to run](#7-how-to-run) — local, Docker, Kubernetes
8. [Testing](#8-testing)
9. [Configuration](#9-configuration)
10. [Design decisions & trade-offs (summary)](#10-design-decisions--trade-offs-summary)

---

## 1. Problem framing

"Find similar products" is an **unsupervised K-Nearest-Neighbors (KNN) retrieval**
problem — **not** a prediction problem.

- The dataset has **no labels** telling us which products *are* similar. So supervised
  methods (logistic regression, linear regression, decision trees / random forests)
  do not apply: there is nothing to predict and nothing to train against.
- The right tool is **similarity search**: represent every product as a numeric
  **vector**, then rank other products by how *close* their vectors are.
  `find_similar_products(product_id, num_similar)` is literally *"find the
  `num_similar` nearest neighbors of this product."*

KNN is *instance-based* / *lazy*: there is no training step — we just compare on
demand.

---

## 2. The data, and how we clean it

The raw file is line-delimited JSON (`.ldjson`), one product per line. Profiling all
30,000 records revealed it is messy, and the cleaning strategy is driven by that
reality:

| Attribute | Coverage | Issue | How we handle it |
|---|---|---|---|
| `uniq_id` | 100%, unique | — | primary key / index |
| `product_name` | 100% | — | lowercase + strip → **strongest signal** |
| `rating` | 100% | skewed high (most 3–5★) | keep, but **down-weighted** (low discrimination) |
| `sales_price` | ~90% | text `"200.00"`, commas, right-skewed | parse → float; impute median; **`log1p` transform** |
| `weight` | ~21% real | **79% is `999999999`**, units `g`/`kg` | parse → grams; junk → missing → median; add **`weight_known` flag** |
| `brand` | ~73% | 6,458 distinct | **match signal**, not one-hot |
| `colour` | ~20% | multi-value `"black\|white"` | **set overlap** bonus |
| images | ~96–99% | `\|`-joined URLs | first URL, for the image phase |

Two decisions worth highlighting:

**(a) Missingness flag (`weight_known`).** We impute missing weights with the median
so the math works — but every imputed product would then look identical on weight,
creating *fake* similarity. So we also record a 1/0 flag for "was this weight real?"
and let the model treat guessed values with less trust. This is standard best-practice
for missing data.

**(b) Log-transform price.** Prices are right-skewed (median ≈ ₹590, max ≈ ₹9,988). On
the raw scale a few expensive items dominate. `log1p(price)` compresses the range so
price behaves sensibly before scaling.

Code: `src/product_similarity/data_loader.py`.

---

## 3. How we measure similarity

Each product becomes **one fused vector** built from three weighted blocks, then
L2-normalized:

```
fused = [ w·(scaled numeric) | w_text·(TF-IDF of name) ]   → L2-normalize
        + brand-match / colour-overlap applied as a re-rank bonus (see below)
```

**Numeric block** — `[log1p(price), rating, weight_g, weight_known]` passed through
`StandardScaler` (mean 0, unit variance) so that price (hundreds) and rating (0–5) get
an equal voice instead of price dominating.

**Text block** — **TF-IDF** on the product name. TF-IDF scores a word high when it is
*frequent in this product but rare across the catalog* — so distinctive words
("saree", "kalamkari") count a lot and generic words ("cotton", "the") count little.
Products sharing distinctive words end up with close vectors.

**Brand & colour** — brand has 6,458 distinct values; one-hot-encoding it would create
thousands of mostly-empty columns. Instead we apply **match signals** as an additive
re-rank bonus in the engine: *same brand* adds a fixed bonus; *shared colours* add a
bonus proportional to set overlap (Jaccard). Missing values simply contribute nothing.
This keeps the fused vector small while honouring "similar brand/colour."

**Why cosine similarity, not Euclidean distance?** Cosine compares the *direction* of
two vectors (what a product is *about*), ignoring *magnitude* (how long the name is,
how big the price number is). For TF-IDF text this matters: a longer product name
shouldn't make an item "less similar." Because we L2-normalize, cosine similarity
equals the dot product — which also makes the exact path and the FAISS inner-product
path mathematically consistent.

**Weights are configurable and data-driven** (`config.py`): text is the dominant
signal; price is medium; rating and weight are low (rating is skewed; weight is 79%
missing). This makes the similarity blend fully configurable.

Code: `src/product_similarity/features.py`, `src/product_similarity/engine.py`.

---

## 4. Architecture

```
                ┌──────────── OFFLINE (once, at startup) ────────────┐
raw .ldjson → DataLoader (clean/parse/impute) → FeatureBuilder
                                                 ├ numeric  → StandardScaler
                                                 ├ text     → TF-IDF
                                                 └ (brand/colour handled in engine)
                                                      ↓ weighted concat + L2-normalize
                                                 FUSED VECTORS → optional FAISS HNSW index
                └────────────────────────────────────────────────────┘

request → GET /find_similar_products?product_id&num_similar
        → SimilarityEngine: look up vector → cosine top-K (exact) OR FAISS (ANN)
        → drop self → brand/colour re-rank → tie-break (rating, price)
        → List[uniq_id] → FastAPI JSON
```

The expensive work (cleaning + vectorizing) happens **once at startup** and is held in
memory, so each request is a fast lookup + compare (~37 ms exact, ~5 ms with FAISS,
over 30k products).

Layered, single-purpose modules:

| File | Responsibility |
|---|---|
| `data_loader.py` | messy `.ldjson` → clean DataFrame |
| `config.py` | weights, paths, feature flags |
| `features.py` | build fused, L2-normalized vectors |
| `engine.py` | `SimilarityEngine` + `find_similar_products` |
| `faiss_index.py` | FAISS HNSW wrapper (fast large-scale search) |
| `images.py` | CLIP image embeddings (optional) |
| `app.py` | FastAPI service |

---

## 5. Fast large-scale search: FAISS / HNSW

Exact search compares the query against all 30,000 vectors per request. That is fine
here (~37 ms) but grows linearly and becomes too slow at millions of products. FAISS
gives us **Approximate Nearest Neighbors (ANN)**: near-identical results, far faster.

**Index chosen: `IndexHNSWFlat` (HNSW).** HNSW builds a multi-layer "navigable" graph —
a coarse top layer for big jumps across the space and finer layers to home in — so a
query reaches its neighborhood in a few hops instead of a full scan.

**Why HNSW over alternatives:**
- vs **Annoy** (random-projection trees): HNSW generally gives higher recall at the
  same latency.
- vs **IVF/PQ**: HNSW needs **no training step** and delivers high recall out of the box.

**Reference:** Yu. A. Malkov and D. A. Yashunin, *"Efficient and robust approximate
nearest neighbor search using Hierarchical Navigable Small World graphs,"* IEEE TPAMI,
2018 ([arXiv:1603.09320](https://arxiv.org/abs/1603.09320)).

**Measured on this dataset (30k products, top-10):**

| | Exact | FAISS HNSW |
|---|---|---|
| avg query latency | ~37 ms | **~5 ms** (~8× faster) |
| top-10 recall vs exact | 100% | **~90%** |

Because vectors are L2-normalized, we use `METRIC_INNER_PRODUCT`, so FAISS ranks by the
same cosine measure as the exact path. Toggle with `PSS_USE_FAISS=true`.

Code: `src/product_similarity/faiss_index.py`.

---

## 6. Optional: CLIP image similarity (multimodal)

Products carry image URLs. To compare *how products look*, we embed images with a
pretrained **CLIP** model (`clip-ViT-B-32`) into 512-dim vectors — this is *transfer
learning* (we reuse a model trained on hundreds of millions of image/text pairs; we do
not train anything). Visually similar products get similar vectors.

**These image vectors are fused into the main search** as a third block:

```
fused = [ numeric | text (TF-IDF) | image (CLIP) ]  → L2-normalize
```

So when enabled, `find_similar_products` ranks by appearance *and* text *and* numbers
together, weighted by `w_image` — a genuine multimodal recommender. Toggle with
`PSS_USE_IMAGES=true`.

Practical notes:
- **Scale-ready but sampled:** downloading + embedding all 30k images (some 2020 URLs
  are dead) is slow and flaky, so by default only the first `PSS_IMAGE_SAMPLE_SIZE`
  products are embedded; the code path is identical at full scale. Products outside the
  sample keep a zero image vector, so text/numeric signals still rank them.
- **Graceful degradation:** an unreachable URL yields a zero vector rather than crashing.
- **Kept out of the core image:** `torch`/CLIP are heavy (~2 GB), so they live in
  `requirements-optional.txt` and are *not* baked into the Docker image. Install with
  `pip install -r requirements-optional.txt` before enabling images.

Code: `src/product_similarity/images.py` (embedder) and the image block in
`src/product_similarity/engine.py` (fusion).

---

## 7. How to run

### Get the dataset

The 74 MB dataset is **git-ignored** (not committed). Download it from Kaggle —
[Amazon Fashion Products 2020](https://www.kaggle.com/datasets/promptcloud/amazon-fashion-products-2020)
— and place the `.ldjson` file under `data/`:

```
data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson
```

### Local (Python)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
```

Then:

```bash
curl "localhost:8000/health"
curl "localhost:8000/find_similar_products?product_id=<UNIQ_ID>&num_similar=5"
```

### Docker (multi-stage, lean ~1.1 GB image)

```bash
docker build -t product-similarity:latest .
docker run --rm -p 8000:8000 product-similarity:latest
curl "localhost:8000/health"
```

### Kubernetes (minikube)

```bash
minikube start
eval $(minikube -p minikube docker-env)     # point docker at minikube
docker build -t product-similarity:latest . # build the image inside the cluster
kubectl apply -f k8s/
kubectl rollout status deployment/product-similarity
minikube service product-similarity --url    # get a reachable URL
eval $(minikube docker-env -u)                # reset docker env when done
```

The Deployment includes `/health` **liveness and readiness probes** and resource
requests/limits.

---

## 8. Testing

```bash
.venv/bin/pytest -v
```

28 tests cover: price/weight parsing and imputation, the missingness flag, colour-set
handling, fused-vector shape + L2-normalization, that similar items rank closer, the
engine's ranking / self-exclusion / error cases, the API's 200/404/422 responses, the
FAISS-vs-exact agreement, and the image dead-URL fallback (mocked, no network).

Tests use a tiny in-repo fixture (`tests/fixtures/sample.ldjson`), so they never
require the 74 MB dataset.

---

## 9. Configuration

All tunables live in `src/product_similarity/config.py` and can be overridden via
`PSS_*` environment variables:

| Env var | Default | Meaning |
|---|---|---|
| `PSS_DATA_PATH` | dataset path | where the `.ldjson` lives |
| `PSS_TEXT_BACKEND` | `tfidf` | `tfidf` or `embeddings` |
| `PSS_USE_FAISS` | `false` | enable the FAISS HNSW fast path |
| `PSS_USE_IMAGES` | `false` | fuse CLIP image features into the search |
| `PSS_IMAGE_SAMPLE_SIZE` | `1000` | how many products' images to embed |
| `PSS_TFIDF_MAX_FEATURES` | `5000` | TF-IDF vocabulary cap |

Blend weights (`w_text`, `w_price`, `w_rating`, `w_weight`, `w_brand`, `w_colour`,
`w_image`) are also defined there.

---

## 10. Design decisions & trade-offs (summary)

| Decision | Why |
|---|---|
| Unsupervised **KNN retrieval** | No similarity labels exist → regression/trees don't apply |
| **Single fused vector** per product | Lets exact search, FAISS, and images share one representation |
| **Cosine** on **L2-normalized** vectors | Compares *content*, not magnitude; makes exact ≡ FAISS inner-product |
| **StandardScaler** on numerics | Prevents price from drowning out rating |
| **log1p(price)** | Tames right-skew so outliers don't dominate |
| **`weight_known` flag** | Prevents imputed weights from faking similarity |
| **Brand/colour as re-rank bonus** | Avoids 6,458 one-hot columns; robust to sparsity |
| **Data-driven weights** | Trust reliable signals (name, price) over weak ones (rating, weight) |
| **TF-IDF default + optional embeddings** | Fast, transparent baseline; semantic path available |
| **HNSW** for ANN | No training, high recall, low latency, scales |
| **CLIP image block, fused + sampled** | Adds visual similarity as a weighted block; sampled to stay fast and dead-URL-tolerant |
| **Multi-stage Docker, heavy deps optional** | 9 GB → ~1.1 GB image; faster K8s pulls, smaller attack surface |
| **Startup-time build, in-memory** | Slow work once; fast per-request lookups |
```
