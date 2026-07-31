"""Tests for the fused feature vectors."""
import numpy as np
from scipy.sparse import issparse

from product_similarity.config import Settings
from product_similarity.features import FeatureBuilder


def test_fit_transform_shape(sample_df):
    fb = FeatureBuilder(Settings())
    X = fb.fit_transform(sample_df)
    assert X.shape[0] == len(sample_df)  # one row per product
    assert X.shape[1] > 4  # numeric block + many tfidf columns


def test_rows_are_l2_normalised(sample_df):
    fb = FeatureBuilder(Settings())
    X = fb.fit_transform(sample_df)
    arr = X.toarray() if issparse(X) else X
    norms = np.linalg.norm(arr, axis=1)
    # every product has text + numeric signal, so each row should be ~unit length
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-6)


def test_similar_sarees_closer_than_shoe(sample_df):
    # p1 and p8 are both cotton sarees; p6 is running shoes.
    fb = FeatureBuilder(Settings())
    X = fb.fit_transform(sample_df)
    arr = X.toarray() if issparse(X) else X
    idx = {pid: i for i, pid in enumerate(sample_df.index)}

    def cos(a, b):
        return float(arr[idx[a]] @ arr[idx[b]])

    assert cos("p1", "p8") > cos("p1", "p6")
