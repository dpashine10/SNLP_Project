"""Single entry point for turning text into embedding vectors.

Wraps Sentence Transformers so no other Week 4 module imports it directly.
The same class serves both indexing (many documents) and search (one query),
which guarantees papers and queries are embedded with the same model and
normalization. It knows nothing about ChromaDB; it only returns vectors.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from src.week4.config import SUPPORTED_DEVICES, DeviceName

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Lazily loaded Sentence Transformers model producing normalized embeddings.

    The model is loaded on first use and reused for every later call, so
    creating an ``Embedder`` is cheap and importing this module does not load
    PyTorch. All returned vectors are L2-normalized, which guarantees that
    similarity between indexed papers and search queries is computed
    comparably regardless of the underlying vector store.

    The embedder never invents, repairs or modifies text. Deciding what to do
    with incomplete records is the responsibility of ``data_adapter``.
    """

    def __init__(self, model_name: str, *, device: DeviceName, batch_size: int) -> None:
        """Create an embedder without loading the model.

        Args:
            model_name: Sentence Transformers model identifier.
            device: ``"auto"``, ``"cpu"`` or ``"cuda"``. See ``_resolve_device``.
            batch_size: Number of texts encoded per forward pass.

        Raises:
            ValueError: If ``model_name`` is empty, ``device`` is unsupported
                or ``batch_size`` is less than 1.
        """
        if not model_name.strip():
            raise ValueError("model_name must not be empty.")
        if device not in SUPPORTED_DEVICES:
            raise ValueError(f"Unsupported device {device!r}; expected one of {SUPPORTED_DEVICES}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}.")
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def model_name(self) -> str:
        """Identifier of the model producing the embeddings."""
        return self._model_name

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed many texts, encoding ``batch_size`` texts at a time.

        Args:
            texts: Texts to embed. Row ``i`` of the result corresponds to
                ``texts[i]``.

        Returns:
            A ``float32`` array of shape ``(len(texts), embedding_dim)``.

        Raises:
            ValueError: If ``texts`` is empty or contains a blank string.
            RuntimeError: If the model cannot be loaded.
        """
        if not texts:
            raise ValueError("texts must not be empty.")
        if any(not text.strip() for text in texts):
            raise ValueError("texts must not contain blank strings.")
        return self._encode(list(texts))

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query.

        Uses the same model and normalization as ``embed_documents`` so query
        and document vectors are directly comparable.

        Args:
            query: Free-text search query.

        Returns:
            A ``float32`` array of shape ``(embedding_dim,)``.

        Raises:
            ValueError: If ``query`` is blank.
            RuntimeError: If the model cannot be loaded.
        """
        if not query.strip():
            raise ValueError("query must not be blank.")
        vector: np.ndarray = self._encode([query])[0]
        return vector

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts with the shared model into normalized ``float32`` vectors."""
        vectors = self._get_model().encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _get_model(self) -> SentenceTransformer:
        """Return the model, loading it on the resolved device on first call.

        The ``sentence_transformers`` import happens here, not at module
        level, to keep importing this module lightweight.

        Raises:
            RuntimeError: If the model cannot be downloaded or loaded.
        """
        if self._model is None:
            device = self._resolve_device(self._device)
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(self._model_name, device=device)
            except Exception as error:
                raise RuntimeError(
                    f"Could not load embedding model {self._model_name!r}: {error}"
                ) from error
            logger.info(
                "Loaded embedding model %s on %s (dimension %s).",
                self._model_name,
                device,
                model.get_embedding_dimension(),
            )
            self._model = model
        return self._model

    @staticmethod
    def _resolve_device(device: DeviceName) -> str:
        """Map the configured device to a PyTorch device string.

        ``"auto"`` selects ``"cuda"`` if available, otherwise ``"mps"``
        (Apple Silicon GPU) if available, otherwise ``"cpu"``.
        ``"cpu"`` and ``"cuda"`` are used as given.

        Raises:
            RuntimeError: If ``"cuda"`` is requested but unavailable. An
                explicit request is never silently downgraded to CPU.
        """
        import torch

        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested but CUDA is not available.")
        return device
