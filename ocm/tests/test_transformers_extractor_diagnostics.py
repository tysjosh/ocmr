"""The parse-failure message must show what the model actually emitted.

A ``json.JSONDecodeError`` says where parsing stopped, not what was there, and
the raw generation is discarded once the error is raised. Without a quoted window
a malformed generation cannot be diagnosed after the fact, which is how the
trailing-comma hypothesis for ``evidence-*:s2`` ended up unverifiable.
"""

from __future__ import annotations

import pytest

from ocm.extraction.base import ExtractionError
from ocm.extraction.transformers_extractor import TransformersExtractor


def _extract(raw: str):
    """Run the extractor over a canned generation."""
    extractor = TransformersExtractor(complete=lambda messages: raw)
    return extractor.extract("Alice owns Project Orion.", "s1")


def test_trailing_comma_message_quotes_the_offending_text() -> None:
    """The signature seen on the GPU run: a dangling comma before the brace."""
    with pytest.raises(ExtractionError) as excinfo:
        _extract('{"entities": [], "relations": [],}')
    message = str(excinfo.value)
    assert "Expecting property name" in message
    assert "near:" in message
    assert "[],}" in message  # the actual defect is now visible


def test_unquoted_key_message_quotes_the_offending_text() -> None:
    with pytest.raises(ExtractionError) as excinfo:
        _extract('{entities: []}')
    assert "near:" in str(excinfo.value)
    assert "entities" in str(excinfo.value)


def test_single_quotes_message_quotes_the_offending_text() -> None:
    with pytest.raises(ExtractionError) as excinfo:
        _extract("{'entities': []}")
    assert "near:" in str(excinfo.value)


def test_window_is_bounded_so_logs_stay_readable() -> None:
    """A long generation must not dump its entirety into the log."""
    filler = '"x": "' + "y" * 5000 + '",'
    with pytest.raises(ExtractionError) as excinfo:
        _extract("{" + filler + "}")
    assert len(str(excinfo.value)) < 500


def test_valid_json_still_parses() -> None:
    result = _extract('{"entities": [], "relations": [], "extractor_version": "v1"}')
    assert result.extractor_version == "v1"


def test_fenced_json_still_parses() -> None:
    result = _extract('```json\n{"entities": [], "extractor_version": "v1"}\n```')
    assert result.extractor_version == "v1"


def test_message_still_classifies_as_a_model_fault() -> None:
    """The added window must not make the message look like an environment fault."""
    from ocm.extraction.strict_extractor import classify

    with pytest.raises(ExtractionError) as excinfo:
        _extract('{"entities": [],}')
    assert classify(str(excinfo.value)) == "model"
