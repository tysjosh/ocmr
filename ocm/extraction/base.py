"""W1 Extractor interface and error type.

The Extractor is the first write-pipeline stage (W1). It turns unstructured
input text into a strict-JSON, Pydantic-validated :class:`ExtractionResult`
holding candidate ``entities``, ``events``, ``claims``, ``documents``,
``decisions``, and ``relations``.

Two implementations exist:

* :class:`~ocm.extraction.mock_extractor.MockExtractor` — the deterministic,
  offline default that requires no API key or network access (
  3.7).
* ``LLMExtractor`` — the opt-in OpenAI-compatible backend.

If an extractor cannot produce output that validates into
:class:`ExtractionResult`, it raises :class:`ExtractionError`; the write
pipeline turns that into a rejected input plus a recorded validation
failure.

"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ocm.memory.contracts import ExtractionResult


class ExtractionError(Exception):
    """Raised when extraction fails to produce a valid ``ExtractionResult``.

    The :class:`~ocm.extraction.mock_extractor.MockExtractor` raises this when
    its assembled candidate payload fails Pydantic validation into
    :class:`ExtractionResult`. The opt-in LLM extractor raises it on
    timeouts or non-JSON / schema-invalid responses.
    """


@runtime_checkable
class Extractor(Protocol):
    """Structural interface for a W1 extractor.

    Implementations expose a stable ``version`` string (recorded as
    ``extractor_version`` for provenance) and an :meth:`extract`
    method returning a validated :class:`ExtractionResult`.
    """

    version: str

    def extract(self, text: str, source_ref: str) -> ExtractionResult:
        """Extract candidate memory items from ``text``.

        Args:
            text: Unstructured input (user message, tool output, document).
            source_ref: Provenance handle for where ``text`` came from.

        Returns:
            A Pydantic-validated :class:`ExtractionResult`.

        Raises:
            ExtractionError: If valid output cannot be produced.
        """
        ...
