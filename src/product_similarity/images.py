"""CLIP image embeddings for multimodal similarity (opt-in via PSS_USE_IMAGES).

Turns product images into vectors using a pretrained CLIP model
(``clip-ViT-B-32`` via ``sentence-transformers``). Visually similar products get
similar vectors, so image similarity uses the same cosine/ANN machinery as the
rest of the system.

Design notes
------------
* **Transfer learning:** we do not train a model; we reuse CLIP, which was trained
  on hundreds of millions of image/text pairs, and read out its embeddings.
* **Scale-ready but sampled:** downloading + embedding all ~30k images (many 2020
  URLs are now dead) is slow and flaky, so in practice this runs on a sample. The
  code path is identical at full scale.
* **Graceful degradation:** a dead/unreachable URL yields a zero vector rather than
  crashing, so one bad image never breaks a batch.
* **Lazy heavy imports:** ``torch``/``PIL`` are imported only when actually
  embedding, so importing this module (and running the mocked test) needs neither.
  Install them with ``pip install -r requirements-optional.txt``.
"""
from __future__ import annotations

import io

import numpy as np


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

    def _open(self, data: bytes):
        """Decode raw image bytes into a PIL RGB image; None on any failure."""
        try:
            from PIL import Image

            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            return None

    def _fetch(self, url: str):
        """Download a URL into a PIL image; return None on any failure."""
        try:
            import requests

            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            return self._open(resp.content)
        except Exception:
            return None

    def embed_image(self, data: bytes):
        """Embed raw uploaded image bytes into a single (dim,) vector.

        Returns None if the bytes can't be decoded (so callers can show a
        friendly error rather than crash). Used by the photo-upload search.
        """
        img = self._open(data)
        if img is None:
            return None
        vec = self._model_lazy().encode([img], convert_to_numpy=True)
        return np.asarray(vec[0], dtype="float32")

    def embed_urls(self, urls: list[str]) -> np.ndarray:
        """Embed a list of image URLs into an (n, dim) matrix.

        Rows for URLs that could not be fetched are left as zero vectors.
        """
        images, keep = [], []
        for k, url in enumerate(urls):
            img = self._fetch(url)
            if img is not None:
                images.append(img)
                keep.append(k)

        out = np.zeros((len(urls), self.dim), dtype="float32")
        if images:
            vecs = self._model_lazy().encode(images, convert_to_numpy=True)
            for pos, k in enumerate(keep):
                out[k] = vecs[pos]
        return out
