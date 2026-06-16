"""Thin wrapper around a fastembed ONNX model. Loaded once in the daemon.

`fastembed` is imported lazily so importing this module never drags in
onnxruntime — the base install (no `skills` extra) can import the package and
the daemon degrades gracefully when the model is unavailable.
"""
from __future__ import annotations

import numpy as np
from loguru import logger

# Cross-lingual default: the user prompts in Chinese, skill descriptions are
# mostly English, so the model must align both. jina-embeddings-v2-base-zh is a
# native bilingual zh/en model (768-dim, ~0.64GB) — markedly better zh→en
# retrieval than the lighter paraphrase-multilingual-MiniLM (which scored only
# ~62% top-1 on Chinese prompts in testing). Downloads once into cache_dir.
DEFAULT_MODEL = "jinaai/jina-embeddings-v2-base-zh"


class EmbedderUnavailable(RuntimeError):
    """Raised when fastembed/onnxruntime isn't installed (no `skills` extra)."""


class Embedder:
    def __init__(
        self, model_name: str = DEFAULT_MODEL, cache_dir: str | None = None
    ) -> None:
        self.model_name = model_name
        # Persistent model cache. fastembed otherwise caches to a temp dir the
        # OS can purge, forcing a slow re-download on the next daemon start.
        self.cache_dir = cache_dir
        self._model: object | None = None

    def ensure_loaded(self) -> None:
        """Load the ONNX model (downloads on first use). Idempotent."""
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # no `skills` extra
            raise EmbedderUnavailable(
                "fastembed not installed; install jarvis-cli[skills]"
            ) from exc
        logger.info(
            "skills: loading embedding model {} (cache={})",
            self.model_name, self.cache_dir,
        )
        kwargs: dict[str, str] = {"model_name": self.model_name}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        self._model = TextEmbedding(**kwargs)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized embeddings, shape (len(texts), dim)."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        self.ensure_loaded()
        assert self._model is not None
        vecs = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
