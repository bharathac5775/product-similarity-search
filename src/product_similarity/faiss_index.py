"""FAISS HNSW index for approximate nearest-neighbour search (Part 3).

Reference
---------
Yu. A. Malkov and D. A. Yashunin, "Efficient and robust approximate nearest
neighbor search using Hierarchical Navigable Small World graphs," IEEE
Transactions on Pattern Analysis and Machine Intelligence, 2018.
arXiv:1603.09320.

Why HNSW?
---------
* **No training step** (unlike IVF/PQ), so it is simple to build and update.
* **High recall at low latency** — graph traversal reaches a query's neighbourhood
  in a few hops instead of scanning every vector.
* **Scales** to millions of vectors, which is the point of the optimisation.

Because the fused vectors are L2-normalised, cosine similarity equals the inner
product, so we use ``METRIC_INNER_PRODUCT`` — the FAISS ranking is then consistent
with the exact cosine path.
"""
from __future__ import annotations

import faiss
import numpy as np


class FaissHNSW:
    def __init__(self, m: int = 32, ef_construction: int = 200, ef_search: int = 64):
        # m: neighbours per node; higher = better recall, more memory.
        # ef_*: search/build breadth; higher = better recall, slower.
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
        """Return (scores, indices) for the k nearest neighbours of one vector."""
        q = np.ascontiguousarray(row_vector.astype("float32")).reshape(1, -1)
        scores, idx = self.index.search(q, k)
        return scores[0], idx[0]
