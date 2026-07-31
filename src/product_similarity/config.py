"""Central configuration: dataset path, blend weights, and feature flags.

Everything tunable lives here so behaviour can be changed without touching logic.
Defaults are *data-driven*: the product name (text) carries most of the signal,
price is a medium signal, and rating/weight are weak (rating is skewed high;
~79% of weights are missing), so they are down-weighted deliberately.
"""
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
    """Immutable settings; override any field via ``PSS_*`` environment variables."""

    data_path: str = _DEFAULT_DATA

    # Blend weights (data-driven).
    w_text: float = 1.0
    w_price: float = 0.5
    w_rating: float = 0.2
    w_weight: float = 0.2
    w_brand: float = 0.4
    w_colour: float = 0.3

    # Feature flags.
    text_backend: str = "tfidf"  # "tfidf" | "embeddings"
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
            tfidf_max_features=int(
                os.getenv("PSS_TFIDF_MAX_FEATURES", base.tfidf_max_features)
            ),
        )


settings = Settings.from_env()
