# Product Similarity Search — Design Spec

**Exercise:** SAP CXII Technical Exercise (`sap-cxii-tech-ex-01`)
**Dataset:** Amazon Fashion products 2020 (`data/…_30k_data.ldjson`, 30,000 records)

---

## 1. Problem framing

Given a product's `uniq_id`, return the `num_similar` most similar products.

This is an **unsupervised K-Nearest-Neighbors (KNN) retrieval** problem, **not** a
prediction problem:

- There are **no labels** ("these two products are similar" is never given), so
  supervised methods — logistic regression, linear regression, decision trees /
  random forests — do not apply. There is nothing to predict and nothing to train.
- The correct tool is **similarity search**: represent each product as a vector,
  then rank other products by closeness. `find_similar_products(id, k)` is literally
  "find the k nearest neighbors of this product."

This framing is stated explicitly in the README so the reasoning is visible to reviewers.

---

## 2. Data profiling (drives every design decision)

Profiled on all 30,000 records:

| Attribute | Coverage | Notes | Consequence |
|---|---|---|---|
| `uniq_id` | 100%, 0 duplicates | clean primary key | index / lookup key |
| `product_name` | 100%, median 8 words | reliable | **dominant signal** |
| `rating` | 100% | skewed high (p10=3.0, median=4.0, p90=5.0) | keep but **down-weight** (low discrimination) |
| `sales_price` | 90% (10% missing), text `"200.00"` | right-skewed (median ₹590, max ₹9,988) | **log-transform** then scale |
| `weight` | **21% real** (79% is `999999999`), text `"200 g"`/`"1.2 kg"` | very sparse | minor signal + **missingness flag** |
| `brand` | 73%, 6,458 distinct | high cardinality | **match signal**, not one-hot |
| `colour` | 20%, multi-value `"black\|white"` | sparse | **set-overlap bonus** |
| images (`medium`/`large`) | ~96–99%, `\|`-joined URLs | for image phase | keep first URL |

**Senior read:** text-dominant retrieval with weak structured signals. Weighting all
features equally would let sparse/low-variance signals degrade results — so defaults are
**data-driven**, not naive.

---

## 3. Architecture (Approach A — single fused vector)

```
                ┌──────────── OFFLINE (once, at startup) ────────────┐
raw .ldjson → DataLoader (clean/parse/impute) → FeatureBuilder
                                                 ├ numeric:  [log-price, rating, weight, weight_known] → StandardScaler
                                                 ├ text:     product_name → TF-IDF  (or embeddings, flag)
                                                 └ categorical: brand-match / colour-overlap (weighted cols)
                                                      ↓ weighted concat + L2-normalize
                                                 FUSED VECTORS (matrix) → optional FAISS HNSW index
                └────────────────────────────────────────────────────┘

request → GET /find_similar_products?product_id&num_similar
        → SimilarityEngine: lookup vector → cosine top-K (exact) OR FAISS (ANN)
        → drop self, tie-break (rating, price) → List[uniq_id] → FastAPI JSON
```

Each product becomes **one fused vector**. Similarity = **cosine** on **L2-normalized**
vectors. The same vectors feed Part 1 (exact), Part 3 (FAISS), and the image phase (CLIP block).

### Folder layout
```
src/product_similarity/
  ├ __init__.py
  ├ config.py         # weights, paths, feature flags
  ├ data_loader.py    # .ldjson → clean DataFrame
  ├ features.py       # build fused vectors (numeric + text + categorical)
  ├ engine.py         # SimilarityEngine, find_similar_products()
  ├ faiss_index.py    # Part 3: FAISS HNSW wrapper
  └ images.py         # last phase: CLIP embeddings
app.py                # FastAPI service (Dockerfile targets app:app)
tests/                # pytest suite
k8s/                  # deployment.yaml, service.yaml
Dockerfile            # multi-stage (see §7)
requirements.txt
README.md             # design writeup (grading)
```

Layered so each unit has one purpose and is independently testable.

---

## 4. Data cleaning (`data_loader.py`)

| Field | Rule |
|---|---|
| `uniq_id` | keep as index (100% clean) |
| `product_name` | lowercase + strip |
| `sales_price` | strip commas → float; missing → **median** (~₹590); `log1p()` then scale |
| `rating` | → float; kept, weighted low |
| `weight` | parse number+unit → **grams** (kg→×1000); `999999999`/`0`/empty → missing → median (~231g); add **`weight_known` (1/0)** flag |
| `brand` | lowercase/strip; missing → `""` (matches nothing) |
| `colour` | split `"a\|b"` → set; missing → empty set |
| images | keep first URL (display + image phase) |

**Two senior-level decisions (documented in README):**

1. **Missingness flag (`weight_known`).** Imputing missing weights with the median makes
   every imputed product *look* identical on weight → fake similarity. The flag lets the
   engine avoid trusting guessed values. Best practice for missing data.
2. **Log-transform price.** Price is right-skewed; `log1p(price)` before scaling stops a
   few expensive items from dominating the numeric block.

Output: a clean DataFrame indexed by `uniq_id`, built once at startup.

---

## 5. Feature building (`features.py`) & similarity (`engine.py`)

**Three blocks, each scaled then weighted, then concatenated & L2-normalized:**

1. **Numeric** — `[log_price, rating, weight_g, weight_known]` via `StandardScaler`
   (mean 0, unit variance) so no single scale dominates.
2. **Text** — `product_name`:
   - **Tier 1 (default): TF-IDF** — fast, transparent, no heavy deps. Matches shared
     distinctive words (rare words weighted high, common words low).
   - **Tier 2 (optional flag `TEXT_BACKEND=embeddings`): sentence-transformer
     `all-MiniLM-L6-v2`** — captures semantic similarity (saree ↔ lehenga), pairs with
     FAISS, bridges to the CLIP image phase (same embed-then-ANN pattern).
3. **Categorical** — folded into the fused vector as compact weighted columns
   (brand-match, colour-overlap), per the "match-based" decision. Optional exact-brand
   re-rank on the top candidates.

**Weights** live in `config.py`; **defaults are data-driven** (text high, price medium,
rating & weight low). Satisfies the README's "configurable weighting."

**Similarity measure — cosine (not Euclidean):**
- Cosine compares **direction** (what a product is about), ignoring **magnitude**
  (name length, price size). Robust for TF-IDF/high-dim/sparse text; a longer name or
  bigger price shouldn't reduce similarity.
- On **L2-normalized** vectors, cosine == inner product, so the FAISS inner-product
  index is mathematically consistent with the exact path.

**`find_similar_products(product_id: str, num_similar: int) -> List[str]`:**
1. look up product's row → its fused vector,
2. cosine vs. the whole matrix (exact) or FAISS (ANN),
3. exclude the product itself,
4. tie-break by `rating` then `sales_price` (per README),
5. return top `num_similar` `uniq_id`s.

Signature matches the README exactly.

---

## 6. Serving (`app.py`, FastAPI)

Engine built **once at startup** (clean + vectorize + optional index), held in memory;
each request is a fast lookup.

**Endpoints:**
- `GET /find_similar_products?product_id=X&num_similar=5` → similar IDs (+ readable
  metadata: name, brand, score).
- `GET /health` → `{"status":"ok"}` for Kubernetes probes.

**Error handling (real HTTP codes, per README):**

| Situation | Code |
|---|---|
| unknown `product_id` | **404** |
| `num_similar` ≤ 0 or wrong type / missing | **422** (FastAPI validation) |
| success | **200** + JSON |

---

## 7. Docker & Kubernetes

**Dockerfile — multi-stage (user requirement; replaces the single-stage repo file):**
- Stage 1 `builder`: install deps into a venv (fastapi, uvicorn, pandas, scikit-learn,
  numpy, faiss-cpu; later sentence-transformers/torch, CLIP).
- Stage 2 `runtime` (`python:3.10-slim`): copy only venv + app code; **non-root user**;
  `EXPOSE 8000`; `CMD uvicorn app:app`.
- Benefits: smaller & more secure image, better layer caching, faster pod startup.

**Kubernetes (`k8s/`, real deploy on minikube):**
- `deployment.yaml` — the container, **liveness + readiness probes** on `/health`,
  resource requests/limits.
- `service.yaml` — expose in-cluster.
- Flow (documented in README): `minikube start` → build image into minikube →
  `kubectl apply -f k8s/` → `minikube service`.

Data file (74 MB) is copied into the image so the container is self-contained; trade-off
(image size vs. simplicity / no external mount) documented.

---

## 8. Part 3 — FAISS (ANN optimization)

Exact KNN scans all 30,000 vectors per request. FAISS builds an index for
**Approximate Nearest Neighbors (ANN)** — near-identical neighbors, much faster, scales
to millions.

- **Index:** `IndexHNSWFlat` (HNSW). Graph-based ANN with excellent speed/recall balance
  at this scale and good scaling behavior.
- **Reference:** Malkov & Yashunin, *"Efficient and robust approximate nearest neighbor
  search using Hierarchical Navigable Small World graphs,"* IEEE TPAMI 2018
  (arXiv:1603.09320). Rationale (HNSW vs. IVF/Annoy: no training step, high recall at low
  latency, incremental inserts) documented in README.
- **Config flag `USE_FAISS`** toggles exact vs. ANN — enables an **agreement test**
  (FAISS top-N ≈ exact top-N) and lets us demonstrate both.

---

## 9. Image similarity (last phase, self-contained)

- **CLIP** (pretrained) → image embeddings; one model handling images well and pairing
  naturally with text embeddings.
- Runs on a **sample** (~500–1000 products) to prove the concept without 30k flaky
  downloads; architected to scale, trade-off documented.
- **Dead-URL handling:** skip + fallback (image block optional per product).
- Plugs into the same fused-vector + weighting design.

---

## 10. Testing (focused pytest suite)

- `data_loader`: `"200 g"→200`, `"1.2 kg"→1200`, `"1,200.00"→1200.0`, `999999999→missing`,
  price imputation.
- `engine`: self excluded; returns exactly `num_similar`; sane ordering; tie-break order.
- `api`: 200 valid, **404** unknown id, **422** bad `num_similar`.
- `faiss`: FAISS top-N ≈ exact top-N (agreement).

---

## 11. Build sequence (risk-ordered — each step ships a complete system)

1. Core: loader + features (TF-IDF + numeric + categorical) + engine + `find_similar_products` + tests → **Part 1**.
2. FastAPI + error codes + `/health` + tests → **Part 2a**.
3. Multi-stage Dockerfile + `requirements.txt` → **Part 2b**.
4. Kubernetes on minikube — deployment/service/probes, real deploy → **must-have**.
5. FAISS HNSW + agreement test + paper citation → **Part 3 bonus**.
6. CLIP images on a sample, fused-in, dead-URL fallback → **optional showcase, last**.

If time runs out at any step, everything before it is complete and gradeable.

---

## 12. What earns the marks (evaluation criteria)

"Clear reasoning, well-structured code, thoughtful design decisions — not perfection."

- **Reasoning:** KNN-not-regression framing; cosine-vs-Euclidean; data-driven weights;
  missingness flag; log-price; TF-IDF-vs-embeddings trade-off; HNSW choice + paper.
- **Structure:** layered, single-purpose modules; config-driven weights/flags; tests.
- **Design:** honest handling of messy/sparse data; graceful degradation; scalable path.
