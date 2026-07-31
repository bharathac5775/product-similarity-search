"""SimilarityEngine: build vectors once, then answer nearest-neighbour queries.

This implements the core function:

    find_similar_products(product_id: str, num_similar: int) -> list[str]

Ranking uses cosine similarity on the L2-normalised fused vectors (so cosine ==
dot product). On top of the base score we add small bonuses for a matching brand
and overlapping colours, then break ties by (score, rating, -price).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import issparse

from .config import Settings, settings as default_settings
from .data_loader import load_products
from .features import FeatureBuilder


class SimilarityEngine:
    def __init__(self, settings: Settings | None = None, image_embedder=None):
        self.settings = settings or default_settings
        # Optional injected embedder (real one is lazy-created; tests pass a fake).
        self._image_embedder = image_embedder
        self.df = None
        self.matrix = None
        self._pos: dict[str, int] = {}  # uniq_id -> row index
        self._ids: list[str] = []  # row index -> uniq_id
        self._fb: FeatureBuilder | None = None
        self._image_vectors = None  # (n, dim) row-normalised, for reverse-image search

    @classmethod
    def from_path(cls, path: str, settings: Settings | None = None) -> "SimilarityEngine":
        eng = cls(settings)
        eng.build(load_products(path))
        return eng

    def _build_image_block(self, df) -> np.ndarray | None:
        """Embed a sample of product images into a weighted dense block.

        Only the first ``image_sample_size`` products are embedded (startup speed +
        dead-URL tolerance). Products outside the sample or with unreachable images
        keep a zero image vector, so text/numeric signals still rank them.
        """
        s = self.settings
        if self._image_embedder is None:
            from .images import ImageEmbedder

            self._image_embedder = ImageEmbedder()

        n = len(df)
        sample = min(s.image_sample_size, n)
        urls = df["image_url"].tolist()

        sample_vecs = self._image_embedder.embed_urls(urls[:sample])
        dim = sample_vecs.shape[1]
        block = np.zeros((n, dim), dtype="float32")
        block[:sample] = sample_vecs

        # Keep the RAW image vectors (row-normalised) so reverse-image search can
        # compare an uploaded photo against them directly. Rows with no image
        # (outside the sample / dead URL) stay all-zero and are skipped at query.
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        self._image_vectors = np.divide(
            block, norms, out=np.zeros_like(block), where=norms > 0
        )

        return block * s.w_image

    def build(self, df) -> None:
        self.df = df
        self._fb = FeatureBuilder(self.settings)

        image_block = self._build_image_block(df) if self.settings.use_images else None
        self.matrix = self._fb.fit_transform(df, image_block=image_block)

        self._ids = list(df.index)
        self._pos = {pid: i for i, pid in enumerate(self._ids)}

        # Optional FAISS HNSW fast path (Part 3). FAISS needs a dense float32
        # matrix, so we materialise one only when the flag is enabled.
        self._faiss = None
        self._dense = None
        if self.settings.use_faiss:
            from .faiss_index import FaissHNSW

            self._dense = (
                self.matrix.toarray() if issparse(self.matrix) else np.asarray(self.matrix)
            ).astype("float32")
            self._faiss = FaissHNSW()
            self._faiss.build(self._dense)

    def _cosine_scores(self, row_idx: int) -> np.ndarray:
        # Rows are L2-normalised, so cosine similarity == dot product.
        q = self.matrix[row_idx]
        scores = self.matrix @ q.T
        return scores.toarray().ravel() if issparse(scores) else np.asarray(scores).ravel()

    def find_similar_products(self, product_id: str, num_similar: int) -> list[str]:
        if num_similar <= 0:
            raise ValueError("num_similar must be a positive integer")
        if product_id not in self._pos:
            raise KeyError(product_id)

        i = self._pos[product_id]

        s = self.settings
        base_brand = self.df.iloc[i]["brand"]
        base_colours = self.df.iloc[i]["colour_set"]

        ratings = self.df["rating"].to_numpy(dtype=float)
        prices = self.df["price"].to_numpy(dtype=float)
        brands = self.df["brand"].to_numpy()
        colour_sets = self.df["colour_set"].to_numpy()

        # Choose candidate set + per-candidate base score.
        if self._faiss is not None:
            # Over-fetch so self-exclusion and re-ranking still leave enough results.
            k = min(len(self._ids), max(num_similar * 5, num_similar + 1))
            _, idxs = self._faiss.search(self._dense[i], k)
            neighbours = [int(j) for j in idxs if 0 <= int(j) < len(self._ids)]

            def base_score(j: int) -> float:
                return float(self._dense[i] @ self._dense[j])
        else:
            sims = self._cosine_scores(i)
            neighbours = range(len(self._ids))

            def base_score(j: int) -> float:
                return float(sims[j])

        candidates = []
        for j in neighbours:
            if j == i:
                continue
            score = base_score(j)
            if base_brand and brands[j] == base_brand:
                score += s.w_brand
            if base_colours and colour_sets[j]:
                union = base_colours | colour_sets[j]
                if union:
                    overlap = len(base_colours & colour_sets[j]) / len(union)
                    score += s.w_colour * overlap
            # Tie-break: higher score, then higher rating, then lower price.
            candidates.append((score, ratings[j], -prices[j], self._ids[j]))

        candidates.sort(reverse=True)
        return [pid for _, _, _, pid in candidates[:num_similar]]

    def find_similar_by_image_vector(self, vec, num_similar: int) -> list[str]:
        """Reverse-image search: given a CLIP image vector, return nearest products.

        Compares the (normalised) query vector against the stored product image
        vectors by cosine similarity. Products with no image vector (outside the
        embedded sample or with a dead URL) are skipped. Requires images ON.
        """
        if num_similar <= 0:
            raise ValueError("num_similar must be a positive integer")
        if self._image_vectors is None:
            raise RuntimeError(
                "image search is unavailable: start the app with PSS_USE_IMAGES=1"
            )

        q = np.asarray(vec, dtype="float32").ravel()
        norm = np.linalg.norm(q)
        if norm == 0:
            raise ValueError("query image produced an empty vector")
        q = q / norm

        scores = self._image_vectors @ q  # cosine (both sides normalised)
        # Only rank products that actually have an image vector (non-zero row).
        has_image = np.linalg.norm(self._image_vectors, axis=1) > 0
        order = np.argsort(-scores)
        out = []
        for j in order:
            if not has_image[j]:
                continue
            out.append(self._ids[j])
            if len(out) >= num_similar:
                break
        return out

    # ------------------------------------------------------------------
    # Display helpers used by the storefront HTML pages (pure df lookups).
    # ------------------------------------------------------------------
    def _row_to_dict(self, pid: str) -> dict:
        """Turn one product row into a display-ready dict for templates."""
        row = self.df.loc[pid]
        return {
            "uniq_id": pid,
            "product_name": row["product_name"],
            "brand": row["brand"],
            "price": float(row["price"]),
            "rating": float(row["rating"]),
            "colours": sorted(row["colour_set"]),
            "image_url": row["image_url"],
        }

    def get_products(self, ids: list[str]) -> list[dict]:
        """Map ids to display dicts, preserving input order and skipping unknowns."""
        return [self._row_to_dict(pid) for pid in ids if pid in self._pos]

    def search(self, query: str, page: int = 1, per_page: int = 24) -> tuple[list[dict], int]:
        """Substring search over product_name + brand, case-insensitive, paged.

        An empty/blank query returns the full catalog in its natural order.
        Returns ``(page_rows, total_matches)`` where page is 1-based.
        """
        q = (query or "").strip().lower()
        if q:
            names = self.df["product_name"].astype(str).str.lower()
            brands = self.df["brand"].astype(str).str.lower()
            mask = names.str.contains(q, regex=False) | brands.str.contains(q, regex=False)
            matched_ids = list(self.df.index[mask])
        else:
            matched_ids = list(self.df.index)

        total = len(matched_ids)
        page = max(1, page)
        start = (page - 1) * per_page
        page_ids = matched_ids[start : start + per_page]
        return self.get_products(page_ids), total


_default_engine: SimilarityEngine | None = None


def find_similar_products(product_id: str, num_similar: int) -> list[str]:
    """Module-level convenience matching the public function signature.

    Lazily builds a default engine from the configured dataset on first call.
    """
    global _default_engine
    if _default_engine is None:
        _default_engine = SimilarityEngine.from_path(default_settings.data_path)
    return _default_engine.find_similar_products(product_id, num_similar)
