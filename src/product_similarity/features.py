"""Build one fused, L2-normalised vector per product.

Each product becomes a single vector by concatenating two weighted blocks:

1. **Numeric** ``[log1p(price), rating, weight_g, weight_known]`` scaled with
   ``StandardScaler`` (so no single scale dominates), then multiplied by the
   per-signal weights from :class:`~product_similarity.config.Settings`.
2. **Text** the ``product_name`` via TF-IDF, multiplied by the text weight.

The fused matrix is L2-normalised row-wise, so cosine similarity reduces to a
dot product and is consistent with a FAISS inner-product index.

Brand and colour are intentionally *not* encoded as columns here (brand has
~6,500 distinct values). They are applied as an additive re-rank bonus in the
engine, honouring the "match-based" design while keeping the fused vector small.
"""
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
        # price, rating, weight, weight_known -> weight_known shares the weight weight
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
