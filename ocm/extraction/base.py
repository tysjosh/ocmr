"""W1 Extractor interface and error type.

The Extractor is the first write-pipeline stage (W1). It turns unstructured
input text into a strict-JSON, Pydantic-validated :class:`ExtractionResult`
holding candidate ``entities``, ``events``, ``claims``, ``documents``,
``decisions``, and ``relations`` (Req 3.1, 3.2).

Two implementations exist:

* :class:`~ocm.extraction.mock_extractor.MockExtractor` — the deterministic,
  offline default that requires no API key or network access (Req 3.4, 3.5,
  3.7).
* ``LLMExtractor`` — the opt-in OpenAI-compatible backend (Req 3.6).

If an extractor cannot produce output that validates into
:class:`ExtractionResult`, it raises :class:`ExtractionError`; the write
pipeline turns that into a rejected input plus a recorded validation
failure (Req 3.3).

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ocm.memory.contracts import ExtractionResult


class ExtractionError(Exception):
    """Raised when extraction fails to produce a valid ``ExtractionResult``.

    The :class:`~ocm.extraction.mock_extractor.MockExtractor` raises this when
    its assembled candidate payload fails Pydantic validation into
    :class:`ExtractionResult` (Req 3.3). The opt-in LLM extractor raises it on
    timeouts or non-JSON / schema-invalid responses (Req 3.3, 3.6).
    """


@runtime_checkable
class Extractor(Protocol):
    """Structural interface for a W1 extractor.

    Implementations expose a stable ``version`` string (recorded as
    ``extractor_version`` for provenance, Req 12.1) and an :meth:`extract`
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
            ExtractionError: If valid output cannot be produced (Req 3.3).
        """
        ...
