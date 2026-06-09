"""Embedding providers for semantic retrieval (Req 13.1, 13.2, 13.3).

Semantic retrieval rests on a **swappable** embedding interface, the
:class:`EmbeddingProvider` protocol, so any model — local or hosted — can be
dropped in via configuration without touching retrieval code (Req 13.1).

Two concrete providers are supplied:

- :class:`LocalEmbeddingProvider` — the default. Wraps
  ``sentence-transformers/all-MiniLM-L6-v2``, a 384-dimensional model that runs
  offline from the local model cache (Req 13.2, 13.3). The heavy
  ``sentence_transformers`` dependency is imported lazily on first embed so that
  importing this module and constructing the provider never require the package
  or a model download.
- :class:`DeterministicEmbeddingProvider` — a lightweight, dependency-free,
  network-free provider that derives a stable 384-dimensional unit vector from
  text via cryptographic hashing. It is intended for hermetic, offline tests
  (vector-index and retrieval tests) where loading a real model is undesirable;
  identical input always yields an identical vector and distinct inputs yield
  distinct vectors.

The :class:`CoreContainer` (task 15.1) selects :class:`LocalEmbeddingProvider`
by default (matching ``Settings.embedding_mode == "local"`` and
``embedding_model == "sentence-transformers/all-MiniLM-L6-v2"``); tests can wire
the deterministic provider instead.

Requirements: 13.1, 13.2, 13.3.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Protocol, runtime_checkable

# Dimensionality of sentence-transformers/all-MiniLM-L6-v2 and of every provider
# defined here, so providers are interchangeable in the vector index (Req 13.1).
EMBEDDING_DIM = 384

# The default local model (Req 13.2).
DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Swappable embedding interface (Req 13.1).

    Any implementation exposes the embedding dimensionality via ``dim`` and can
    embed a batch of texts (:meth:`embed`) or a single text (:meth:`embed_one`).
    All providers in this module are 384-dimensional so they are interchangeable.
    """

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into a list of ``dim``-length float vectors."""
        ...

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text into a ``dim``-length float vector."""
        ...


class LocalEmbeddingProvider:
    """Default provider wrapping ``all-MiniLM-L6-v2`` (Req 13.2, 13.3).

    The model is 384-dimensional and runs offline from the local model cache.
    ``sentence_transformers`` is imported lazily on first embed so that importing
    this module and constructing the provider never trigger the dependency or a
    model download; the only network touch is the optional first-run weight
    download, which can be pre-baked for fully offline operation.
    """

    dim: int = EMBEDDING_DIM

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL) -> None:
        self.model_name = model_name
        # Lazy-loaded SentenceTransformer instance; populated on first embed.
        self._model = None

    def _ensure_model(self) -> None:
        """Lazily import sentence_transformers and load the model on first use.

        Raises:
            RuntimeError: If ``sentence_transformers`` is not installed, with a
                clear, actionable message. Callers that need a guaranteed-offline
                provider (e.g. tests) should use
                :class:`DeterministicEmbeddingProvider` instead.
        """
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only when dep missing
            raise RuntimeError(
                "LocalEmbeddingProvider requires the 'sentence-transformers' package "
                f"to load model {self.model_name!r}. Install it (pip install "
                "sentence-transformers) or use DeterministicEmbeddingProvider for "
                "offline/hermetic runs."
            ) from exc
        self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using the local MiniLM model."""
        self._ensure_model()
        assert self._model is not None  # for type checkers
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=False,
            normalize_embeddings=False,
        )
        # SentenceTransformer may return tensors/arrays; coerce to plain floats.
        return [[float(v) for v in vec] for vec in vectors]

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text using the local MiniLM model."""
        return self.embed([text])[0]


class DeterministicEmbeddingProvider:
    """Hashing-based, offline, dependency-free 384-dim provider for tests.

    Produces a deterministic 384-dimensional unit vector for any text using
    SHA-256 as a stable pseudo-random source (Python's built-in ``hash`` is
    salted per-process and therefore unsuitable). The same text always maps to
    the same vector, and distinct texts map to distinct vectors, so vector-index
    add/query round-trips and retrieval can be exercised hermetically without
    loading a real embedding model or touching the network.

    This implements :class:`EmbeddingProvider` and is interchangeable with
    :class:`LocalEmbeddingProvider` (same ``dim``).
    """

    dim: int = EMBEDDING_DIM

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into deterministic unit vectors."""
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        """Derive a stable, L2-normalized ``dim``-length vector from ``text``.

        The text is hashed in successive blocks (``SHA-256(counter || text)``)
        to produce enough bytes for ``dim`` 32-bit floats, which are then
        L2-normalized so cosine similarity is well-behaved in the vector index.
        """
        raw = text.encode("utf-8")
        floats: list[float] = []
        counter = 0
        # Each SHA-256 block yields 32 bytes -> eight 32-bit floats.
        while len(floats) < self.dim:
            block = hashlib.sha256(counter.to_bytes(4, "big") + raw).digest()
            for i in range(0, len(block), 4):
                if len(floats) >= self.dim:
                    break
                # Map 4 bytes to an unsigned int, then to a float in [-1, 1).
                (value,) = struct.unpack(">I", block[i : i + 4])
                floats.append((value / 0xFFFFFFFF) * 2.0 - 1.0)
            counter += 1

        norm = math.sqrt(sum(component * component for component in floats))
        if norm == 0.0:
            # Degenerate (effectively impossible) case: return a unit basis vector.
            unit = [0.0] * self.dim
            unit[0] = 1.0
            return unit
        return [component / norm for component in floats]
