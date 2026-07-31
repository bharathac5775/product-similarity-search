# Product Similarity Search

A content-based recommender that, given a product's `uniq_id`, returns the *N* most
similar products from the Amazon Fashion 2020 dataset (~30,000 items). It is built
around one machine-learning idea — **represent every product as a vector, then rank
by nearest neighbours** — and wraps that engine in a FastAPI service, a
server-rendered storefront, a lean multi-stage Docker image, and Kubernetes
manifests. Two ML capabilities are opt-in: an **approximate-nearest-neighbour** fast
path (**FAISS/HNSW**) and **multimodal visual search** with **CLIP** image embeddings.

> This document explains not just *how* to run the project, but *why* each modelling
> decision was made — the reasoning and trade-offs behind the approach.

---

## Table of contents

1. [Problem framing: why this is k-Nearest-Neighbours, not a prediction task](#1-problem-framing)
2. [The data, and the ML-driven cleaning it needs](#2-the-data-and-the-ml-driven-cleaning-it-needs)
3. [Feature engineering: turning a product into a vector](#3-feature-engineering-turning-a-product-into-a-vector)
4. [How we measure similarity (the ranking model)](#4-how-we-measure-similarity-the-ranking-model)
5. [Architecture](#5-architecture)
6. [Fast large-scale search: FAISS / HNSW](#6-fast-large-scale-search-faiss--hnsw)
7. [Multimodal visual search: CLIP (two *separate* retrieval paths)](#7-multimodal-visual-search-clip)
8. [The service: API + storefront](#8-the-service-api--storefront)
9. [How to run](#9-how-to-run) — local, Docker, Kubernetes
10. [Testing](#10-testing)
11. [Configuration](#11-configuration)
12. [Design decisions & trade-offs (summary)](#12-design-decisions--trade-offs-summary)

---

## 1. Problem framing

"Find similar products" is an **unsupervised k-Nearest-Neighbours (kNN) retrieval**
problem — **not** a supervised prediction problem. Getting this framing right is the
single most important modelling decision in the project.

- The dataset carries **no labels** telling us which products *are* similar. There is
  no target column to predict and nothing to train against, so supervised methods
  (logistic/linear regression, decision trees, random forests, gradient boosting) do
  not apply — they would need a `(features → label)` signal that simply does not exist
  here.
- The right tool is **similarity search / representation learning**: encode every
  product as a numeric **vector** in a shared space, then rank other products by how
  *close* their vectors are. `find_similar_products(product_id, num_similar)` is
  literally *"return the `num_similar` nearest neighbours of this product's vector."*

kNN is **instance-based / lazy**: there is no fitted model with learned parameters to
predict from. The "training" is really *representation building* — we fit a TF-IDF
vocabulary and a feature scaler over the catalogue once, then compare on demand. That
distinction (transform-fit, not predict-fit) is why the whole pipeline is built as a
one-time offline vectorisation followed by cheap online comparison.

---

## 2. The data, and the ML-driven cleaning it needs

The raw file is line-delimited JSON (`.ldjson`), one product per line. Profiling all
30,000 records shows it is messy, and **every cleaning decision is a modelling
decision** — bad imputation manufactures fake similarity, so we treat cleaning as part
of the ML pipeline, not a preprocessing afterthought.

| Attribute | Coverage | Issue | ML-aware handling |
|---|---|---|---|
| `uniq_id` | 100%, unique | — | primary key / index |
| `product_name` | 100% | — | lowercase + strip → TF-IDF → **strongest signal** |
| `rating` | 100% | skewed high (most 3–5★) | keep, but **down-weighted** (low discriminative power) |
| `sales_price` | ~90% | text `"1,200.00"`, commas, right-skewed | parse → float; median-impute; **`log1p` transform** |
| `weight` | ~21% real | **79% is the sentinel `999999999`**, units `g`/`kg` | parse → grams; junk → missing → median; add **`weight_known` flag** |
| `brand` | ~73% raw → **100% after inference** | 6,458 distinct values; 27% empty | recover from name (below); used as a **match signal**, not one-hot |
| `colour` | ~20% | multi-value `"black\|white"` | parse to a **set** → Jaccard-overlap bonus |
| images | ~96–99% | `\|`-joined URLs | first URL → CLIP embedding (opt-in) |

Four decisions worth highlighting — each is about not letting the *absence* of data
lie to the model:

**(a) Missingness flag (`weight_known`).** With 79% of weights missing, median-imputing
them makes 4 in 5 products look *identical* on weight — manufacturing similarity that
isn't real. We impute so the arithmetic works, **and** record a 1/0 `weight_known`
flag as its own feature, so the model can tell a measured weight from a guessed one.
This is standard best practice for informative missingness.

**(b) Log-transform price.** Prices are right-skewed (median ≈ ₹590, max ≈ ₹9,988). On
the raw scale a handful of expensive items dominate the distance metric. `log1p(price)`
compresses the tail so price contributes proportionally before scaling.

**(c) Brand inference (recover the missing 27%).** 8,143 of ~30k products (27%) have an
empty `brand`, yet the brand is almost always the leading word(s) of the name
(*"Puma Men's T-Shirt"*, *"Peter England Formal Shirt"*). `_infer_missing_brands`
first learns the set of brands that **are** present, then for each empty brand assigns
the **longest known brand the name starts with** (matched on a word boundary, so
`"max"` doesn't match `"maximus"`), falling back to the name's first word. This is a
lightweight data-driven imputation — no hard-coded brand list — and it takes missing
brands from **27% → 0%**, which materially improves the brand-match re-rank signal.

**(d) Colour as a set, not a string.** `"black|white"` is parsed into
`frozenset{"black","white"}` so similarity can use **Jaccard overlap** (shared ÷ union)
rather than exact string equality — a black/white shirt is partially similar to a
black shirt.

Code: `src/product_similarity/data_loader.py`.

---

## 3. Feature engineering: turning a product into a vector

Each product becomes **one fused, L2-normalised vector**. The vector is the model —
everything downstream (exact search, FAISS, tie-breaks) operates on it.

```
click-similarity vector = [ w · scaled_numeric | w_text · TF-IDF(name) ]  → L2-normalise
                           (+ brand-match and colour-overlap applied as a re-rank bonus)
```

**Numeric block** — `[log1p(price), rating, weight_g, weight_known]` passed through
`StandardScaler` (zero mean, unit variance) so that price (hundreds) and rating (0–5)
get an equal voice instead of price dominating purely because its numbers are bigger.
Each column is then multiplied by its configured weight.

**Text block (the dominant signal)** — **TF-IDF** over the product name.
TF-IDF = *Term Frequency* × *Inverse Document Frequency*: a word scores high when it is
**frequent in this product but rare across the catalogue**. So distinctive words
(*"saree"*, *"kalamkari"*) carry weight while generic words (*"cotton"*, *"the"*) are
damped automatically — no stop-word list needed. Products that share distinctive
vocabulary land near each other. The vocabulary is capped at `PSS_TFIDF_MAX_FEATURES`
(default **5000**) most-informative terms: enough to be expressive, small enough to
keep the sparse matrix fast and to avoid overfitting to one-off tokens.

**Brand & colour — match signals, not columns.** Brand has 6,458 distinct values;
one-hot encoding would add thousands of near-empty columns (the curse of
dimensionality) and make the vectors huge and sparse in an unhelpful way. Instead we
keep them **out of the vector** and apply them as an **additive re-rank bonus** in the
engine: an exact brand match adds a fixed bonus (`w_brand`); shared colours add a bonus
proportional to Jaccard overlap (`w_colour`). Missing values contribute nothing, so
the approach degrades gracefully on sparse data.

**Weights are data-driven and configurable** (`config.py`): text is dominant
(`w_text=1.0`); price is medium (`w_price=0.5`); rating and weight are weak
(`0.2` each — rating is skewed, weight is 79% missing). We trust reliable signals over
noisy ones by construction.

Code: `src/product_similarity/features.py`.

---

## 4. How we measure similarity (the ranking model)

**Cosine similarity, not Euclidean distance.** Cosine compares the *direction* of two
vectors (what a product is *about*) and ignores *magnitude* (how long the name is, how
large the raw price number is). For TF-IDF text this is essential — a longer product
name should not make an item "less similar." Because we **L2-normalise** every vector,
cosine similarity reduces to a plain **dot product**, which also makes the exact path
and the FAISS inner-product path *mathematically identical* in what they rank.

The full ranking, per query (`engine.py`):

1. Look up the query product's vector.
2. Score every other product by cosine (exact) **or** fetch a candidate set from FAISS
   (approximate) — see §6.
3. **Drop the query itself** (a product is trivially its own nearest neighbour).
4. **Re-rank bonuses:** `+w_brand` if the brand matches; `+w_colour × Jaccard(colours)`.
5. **Deterministic tie-break:** by `(score, rating, −price)` — when two products score
   equally, prefer the higher-rated, then the cheaper one, so results are stable and
   reproducible.

Code: `src/product_similarity/engine.py`.

---

## 5. Architecture

```
                ┌──────────── OFFLINE (once, at startup) ────────────┐
raw .ldjson → DataLoader (parse / impute / brand-infer) → FeatureBuilder
                                                 ├ numeric → StandardScaler
                                                 └ text    → TF-IDF
                                                      ↓ weighted concat + L2-normalise
                                                 FUSED VECTORS ─┬─→ (optional) FAISS HNSW index
                                                                └─→ (optional) separate CLIP
                                                                     image index  (for photo upload)
                └────────────────────────────────────────────────────┘

click a product → GET /find_similar_products?product_id&num_similar
        → SimilarityEngine: vector lookup → cosine top-K (exact) OR FAISS (ANN)
        → drop self → brand/colour re-rank → tie-break (rating, price) → List[uniq_id]

upload a photo  → POST /search_by_image  (opt-in, images on)
        → CLIP-embed the uploaded bytes → cosine over the SEPARATE image index → List[uniq_id]
```

The expensive work (cleaning + vectorising, and optionally embedding images / building
the FAISS graph) happens **once at startup** and is held in memory, so each request is
a fast lookup + compare (~37 ms exact, ~5 ms with FAISS, over 30k products).

Layered, single-purpose modules:

| File | Responsibility |
|---|---|
| `data_loader.py` | messy `.ldjson` → clean DataFrame (parse, impute, infer brands) |
| `config.py` | weights, paths, feature flags (all `PSS_*`-overridable) |
| `features.py` | build the fused, L2-normalised text+numeric vectors |
| `engine.py` | `SimilarityEngine`: click-similarity, reverse-image search, display helpers |
| `faiss_index.py` | FAISS HNSW wrapper (approximate fast path) |
| `images.py` | CLIP embedder — embed URLs (index) and uploaded bytes (query) |
| `app.py` | FastAPI service: JSON API + server-rendered storefront |

---

## 6. Fast large-scale search: FAISS / HNSW

Exact search compares the query against all 30,000 vectors per request. That is fine
here (~37 ms) but grows **linearly** and becomes too slow at millions of products.
FAISS gives us **Approximate Nearest Neighbours (ANN)**: near-identical results, far
faster.

**Index: `IndexHNSWFlat` (Hierarchical Navigable Small World).** HNSW builds a
multi-layer "navigable" graph — a coarse top layer for big jumps across the space,
finer layers to home in — so a query reaches its neighbourhood in a few hops instead of
a full scan. Parameters (`faiss_index.py`): `M=32` neighbours per node,
`efConstruction=200`, `efSearch=64` (recall/latency knobs).

**Why HNSW over alternatives:**
- vs **Annoy** (random-projection trees): HNSW generally gives higher recall at the
  same latency.
- vs **IVF/PQ**: HNSW needs **no training step** and delivers high recall out of the box.

**Consistency with the exact path.** Because vectors are L2-normalised, cosine equals
the inner product, so we build the index with `METRIC_INNER_PRODUCT`. The engine
over-fetches `5×` candidates from FAISS and then re-ranks them with the exact cosine +
brand/colour bonuses, so the FAISS path returns the *same* top-N as exact search in
practice while doing far less work.

**Measured on this dataset (30k products, top-10):**

| | Exact | FAISS HNSW |
|---|---|---|
| avg query latency | ~37 ms | **~5 ms** (~8× faster) |
| top-10 recall vs exact | 100% | **~90%** |

**Reference:** Yu. A. Malkov and D. A. Yashunin, *"Efficient and robust approximate
nearest neighbor search using Hierarchical Navigable Small World graphs,"* IEEE TPAMI,
2018 ([arXiv:1603.09320](https://arxiv.org/abs/1603.09320)).

Toggle with `PSS_USE_FAISS=1`. Code: `src/product_similarity/faiss_index.py`.

---

## 7. Multimodal visual search: CLIP

Products carry image URLs. To compare *how products look*, we embed images with a
pretrained **CLIP** model (`clip-ViT-B-32` via `sentence-transformers`) into **512-dim**
vectors. This is **transfer learning**: CLIP was trained on hundreds of millions of
image/text pairs; we do not train anything — we read out its embeddings. Visually
similar products get similar vectors, so image similarity reuses the same
cosine machinery as the rest of the system.

### Two *separate* retrieval paths — this is deliberate

Earlier the CLIP vector was *fused* into the main search vector. That was wrong: it let
a product's **picture contaminate its text/attribute ranking**, so "similar products"
drifted depending on whether images were on. Production visual-search systems
(Google Lens, Pinterest, Amazon) keep the two apart, and so do we:

| User action | Endpoint | Signal used |
|---|---|---|
| **Click a product** | `GET /find_similar_products`, `/product/{id}` | text + numeric + brand/colour **only** |
| **Upload a photo** | `POST /search_by_image` | **CLIP image embedding only** |

The consequence that matters: **turning images on no longer changes click-similarity at
all.** `find_similar_products` is identical with images on or off (verified — see
`tests/test_image_decoupling.py`). Enabling images only *adds* the photo-upload feature;
it does not perturb the text ranking. The CLIP vectors live in their own
row-normalised index (`_image_vectors` in `engine.py`), queried exclusively by
`find_similar_by_image_vector`.

### Practical notes

- **Scale-ready but sampled.** Downloading + embedding all 30k images (many 2020 URLs
  are now dead) is slow and flaky, so by default only the first `PSS_IMAGE_SAMPLE_SIZE`
  products are embedded into the image index. The code path is identical at full scale;
  products outside the sample simply carry no image vector and are skipped by
  photo-search (they are still fully ranked by *click*-similarity, which never used
  images).
- **Graceful degradation.** An unreachable URL or an undecodable upload yields `None`/a
  zero vector and a friendly message rather than a crash.
- **Kept out of the core image.** `sentence-transformers`/`torch`/`pillow` are heavy
  (~2 GB) and live in `requirements-optional.txt`, **not** in the Docker image. Install
  them locally before enabling images:
  `pip install -r requirements-optional.txt`.
- **⚠️ FAISS + images together on macOS.** FAISS and torch each bundle their own
  `libomp.dylib`; loading both in one process can segfault. If you run
  `PSS_USE_FAISS=1 PSS_USE_IMAGES=1` locally, prefix the command with
  `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE`. (The default Kubernetes config keeps
  FAISS off, so this only affects the local combo.)

Code: `src/product_similarity/images.py` (embedder) and the reverse-image index in
`src/product_similarity/engine.py`.

---

## 8. The service: API + storefront

One FastAPI app (`app.py`) exposes both a JSON API and a server-rendered storefront
(Jinja2 templates + static CSS — no build step, no separate frontend server).

| Route | Type | Purpose |
|---|---|---|
| `GET /` | HTML | storefront home: keyword search + paged product grid |
| `GET /product/{id}` | HTML | product detail + a "Similar products" row (click-similarity) |
| `POST /search_by_image` | HTML | **reverse-image search**: upload a photo → CLIP visual matches |
| `GET /find_similar_products?product_id&num_similar` | JSON | the core API: nearest-neighbour `uniq_id`s + display fields |
| `GET /gallery?product_id&num_similar` | HTML | quick visual demo page (query + neighbours as images) |
| `GET /health` | JSON | liveness / readiness probe |

**Two kinds of "search", don't confuse them:**
- The **search bar** (`/?q=shirt`) is a plain case-insensitive **substring match** over
  name + brand — it is *not* the ML model, just catalogue filtering, and returns every
  match (paged, `PER_PAGE=24`).
- **"Similar products"** on a product page is the **ML kNN model** (`NUM_SIMILAR=12`).

Code: `app.py`, `templates/`, `static/app.css`.

---

## 9. How to run

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

Then open <http://localhost:8000/> for the storefront, or hit the API:

```bash
curl "localhost:8000/health"
curl "localhost:8000/find_similar_products?product_id=<UNIQ_ID>&num_similar=5"
```

**Optional ML features** (install the heavy extras first —
`.venv/bin/pip install -r requirements-optional.txt`):

```bash
# FAISS approximate search
PSS_USE_FAISS=1 PYTHONPATH=src .venv/bin/uvicorn app:app --port 8000

# CLIP photo-upload search (embeds the first 400 images at startup, ~1–2 min)
PSS_USE_IMAGES=1 PSS_IMAGE_SAMPLE_SIZE=400 PYTHONPATH=src .venv/bin/uvicorn app:app --port 8000

# both together on macOS — note the OpenMP guard (see §7)
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
  PSS_USE_FAISS=1 PSS_USE_IMAGES=1 PSS_IMAGE_SAMPLE_SIZE=400 \
  PYTHONPATH=src .venv/bin/uvicorn app:app --port 8000
```

### Docker (multi-stage, lean image)

The dataset is baked in; heavy ML extras are not, keeping the image slim.

```bash
docker build -t product-similarity:latest .
docker run --rm -p 8000:8000 product-similarity:latest
curl "localhost:8000/health"
```

### Kubernetes (minikube)

```bash
minikube start
eval $(minikube -p minikube docker-env)      # point docker at minikube's daemon
docker build -t product-similarity:latest .  # build the image INSIDE the cluster
kubectl apply -f k8s/                         # ConfigMap + Deployment + Service
kubectl rollout status deployment/product-similarity
kubectl port-forward svc/product-similarity 8080:80   # reliable access on macOS
eval $(minikube docker-env -u)                # reset docker env when done
```

Building inside minikube's daemon avoids the `:latest` image-cache trap
(`imagePullPolicy: IfNotPresent` + tag reuse serving a stale image). The Deployment
runs as a non-root user with a read-only root filesystem, drops all capabilities, sets
CPU/memory requests+limits, and gates traffic on `/health` **startup, readiness, and
liveness** probes. Runtime behaviour is driven by the ConfigMap (`k8s/configmap.yaml`),
so you can change flags without rebuilding.

---

## 10. Testing

```bash
PYTHONPATH=src .venv/bin/pytest -q
```

**50 tests**, all using a tiny in-repo fixture (`tests/fixtures/sample.ldjson`) so they
never need the 74 MB dataset or any network/torch. Coverage includes:

- **Data cleaning:** price/weight parsing, median imputation, the `weight_known` flag,
  colour-set splitting, and **brand inference** (`test_data_loader.py`).
- **Feature engineering:** fused-vector shape and L2-normalisation (`test_features.py`).
- **Ranking model:** similar items rank closer, self-exclusion, tie-breaks, error cases
  (`test_engine.py`), plus display/search helpers (`test_engine_display.py`).
- **FAISS:** the ANN path agrees with exact search (`test_faiss_index.py`).
- **Images (mocked, no network):** CLIP dead-URL fallback and upload-bytes embedding
  (`test_images.py`), reverse-image ranking (`test_image_search.py`), and the key
  regression that **click-similarity is identical with images on vs off**
  (`test_image_decoupling.py`, `test_image_fusion.py`).
- **API:** 200/404/422 responses and the storefront/upload routes (`test_api.py`).

---

## 11. Configuration

All tunables live in `src/product_similarity/config.py` and can be overridden via
`PSS_*` environment variables (also set in `k8s/configmap.yaml`):

| Env var | Default | Meaning |
|---|---|---|
| `PSS_DATA_PATH` | dataset path | where the `.ldjson` lives |
| `PSS_TEXT_BACKEND` | `tfidf` | text vectoriser (`tfidf`; `embeddings` reserved) |
| `PSS_USE_FAISS` | `0` | enable the FAISS HNSW approximate fast path |
| `PSS_USE_IMAGES` | `0` | build the CLIP image index → enables photo-upload search |
| `PSS_IMAGE_SAMPLE_SIZE` | `1000` | how many products' images to embed into the index |
| `PSS_TFIDF_MAX_FEATURES` | `5000` | TF-IDF vocabulary cap |

Blend weights — `w_text=1.0`, `w_price=0.5`, `w_rating=0.2`, `w_weight=0.2`,
`w_brand=0.4`, `w_colour=0.3`, `w_image=0.6` — are defined there too.
(`w_image` now scales only the standalone image index, since images no longer fuse into
click-similarity.)

---

## 12. Design decisions & trade-offs (summary)

| Decision | Why |
|---|---|
| Unsupervised **kNN retrieval** | No similarity labels exist → regression/trees don't apply |
| **Single fused vector** per product | Lets exact search and FAISS share one representation |
| **Cosine** on **L2-normalised** vectors | Compares *content*, not magnitude; makes exact ≡ FAISS inner-product |
| **StandardScaler** on numerics | Prevents price's larger numbers from drowning out rating |
| **log1p(price)** | Tames right-skew so a few expensive items don't dominate distance |
| **`weight_known` flag** | Stops 79%-imputed weights from faking similarity (informative missingness) |
| **Brand inference from name** | Recovers 27% missing brands (→0%) with no hard-coded list |
| **Brand/colour as re-rank bonus** | Avoids 6,458 one-hot columns; degrades gracefully on sparsity |
| **TF-IDF, 5000-term cap** | Distinctive-word signal; small/fast sparse matrix; resists overfitting |
| **Data-driven weights** | Trust reliable signals (name, price) over weak ones (rating, weight) |
| **HNSW** for ANN | No training, high recall, low latency, scales to millions |
| **CLIP as a *separate* image index** | Visual search without contaminating text similarity; sampled + dead-URL-tolerant |
| **Multi-stage Docker, heavy ML deps optional** | ~9 GB → lean image; faster K8s pulls, smaller attack surface |
| **Startup-time build, in-memory** | Slow work once; fast per-request lookups |
