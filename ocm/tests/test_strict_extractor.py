"""Tests for the fail-fast W1 wrapper (:mod:`ocm.extraction.strict_extractor`).

The property under test is the one that protects result validity: an environment
fault must escape :class:`~ocm.memory.write_pipeline.WritePipeline`'s
``except ExtractionError`` clause and abort the run, while a model fault must
still be absorbed as a recorded validation failure (Req 3.3).
"""

from __future__ import annotations

import pytest

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.extraction.base import ExtractionError
from ocm.extraction.strict_extractor import (
    ExtractionEnvironmentError,
    StrictExtractor,
    classify,
)
from ocm.memory.contracts import ExtractionResult

#: The message a broken Triton toolchain actually produced on the GPU host.
REAL_TRITON_MESSAGE = (
    "transformers generation failed: CalledProcessError(1, ['/usr/bin/gcc', "
    "'/tmp/tmp2o3jcq5i/cuda_utils.c', '-O3', '-shared', '-fPIC']) "
    "fatal error: Python.h: No such file or directory"
)


class _Boom:
    """An extractor that always raises the given message."""

    version = "boom-v1"

    def __init__(self, message: str) -> None:
        self.message = message
        self.calls = 0

    def extract(self, text: str, source_ref: str) -> ExtractionResult:
        self.calls += 1
        raise ExtractionError(self.message)


@pytest.mark.parametrize(
    "message",
    [
        REAL_TRITON_MESSAGE,
        "transformers generation failed: RuntimeError('CUDA out of memory')",
        "transformers generation failed: RuntimeError('CUDA error: no kernel image is available')",
    ],
)
def test_environment_faults_classified_as_environment(message: str) -> None:
    assert classify(message) == "environment"


@pytest.mark.parametrize(
    "message",
    [
        "transformers extractor returned empty output",
        "transformers extractor output contained no JSON object",
        "transformers extractor produced invalid JSON: Expecting value",
        "transformers extractor output failed validation: 1 validation error",
    ],
)
def test_model_faults_classified_as_model(message: str) -> None:
    assert classify(message) == "model"


def test_environment_fault_raises_a_non_extraction_error() -> None:
    """The abort type must not be an ExtractionError, or the pipeline swallows it."""
    strict = StrictExtractor(_Boom(REAL_TRITON_MESSAGE))
    with pytest.raises(ExtractionEnvironmentError) as excinfo:
        strict.extract("Alice owns Project Orion.", "s1")
    assert not isinstance(excinfo.value, ExtractionError)
    assert "Python.h" in str(excinfo.value)
    assert "python3.12-dev" in str(excinfo.value)  # carries the remedy


def test_model_fault_is_re_raised_unchanged_and_counted() -> None:
    strict = StrictExtractor(_Boom("transformers extractor produced invalid JSON: x"))
    with pytest.raises(ExtractionError):
        strict.extract("Alice owns Project Orion.", "s1")
    assert strict.stats["model_failures"] == 1
    assert strict.stats["environment_failures"] == 0
    assert strict.stats["model_failure_rate"] == 1.0


def test_environment_fault_propagates_through_the_write_pipeline() -> None:
    """The regression this guards: a broken environment must not be absorbed."""
    container = CoreContainer(
        Settings(deterministic_test_mode=True, chroma_mode="memory"),
        extractor=StrictExtractor(_Boom(REAL_TRITON_MESSAGE)),
    )
    with pytest.raises(ExtractionEnvironmentError):
        container.write_pipeline.run("Alice owns Project Orion.", source_ref="s1")


def test_model_fault_is_still_absorbed_by_the_write_pipeline() -> None:
    """Req 3.3 behaviour is preserved for genuine model failures."""
    container = CoreContainer(
        Settings(deterministic_test_mode=True, chroma_mode="memory"),
        extractor=StrictExtractor(_Boom("transformers extractor returned empty output")),
    )
    record = container.write_pipeline.run("Alice owns Project Orion.", source_ref="s1")
    assert record is not None


def test_tolerate_flag_restores_the_degraded_path() -> None:
    strict = StrictExtractor(
        _Boom(REAL_TRITON_MESSAGE), tolerate_environment_errors=True
    )
    with pytest.raises(ExtractionError):  # not ExtractionEnvironmentError
        strict.extract("Alice owns Project Orion.", "s1")
    assert strict.stats["environment_failures"] == 1


def test_successful_extraction_passes_through() -> None:
    class _Ok:
        version = "ok-v1"

        def extract(self, text: str, source_ref: str) -> ExtractionResult:
            return ExtractionResult(extractor_version="ok-v1")

    strict = StrictExtractor(_Ok())
    result = strict.extract("Alice owns Project Orion.", "s1")
    assert result.extractor_version == "ok-v1"
    assert strict.stats == {
        "calls": 1,
        "model_failures": 0,
        "model_failure_rate": 0.0,
        "environment_failures": 0,
        "model_failure_examples": [],
    }


# --------------------------------------------------------------------------- #
# Preflight: the second silent-degradation mode
# --------------------------------------------------------------------------- #
class _Empty:
    """Parses cleanly but extracts nothing. Raises no error anywhere."""

    version = "empty-v1"

    def extract(self, text: str, source_ref: str) -> ExtractionResult:
        return ExtractionResult(extractor_version="empty-v1")


class _Relational:
    """Emits one relation per call, like a working extractor."""

    version = "rel-v1"

    def extract(self, text: str, source_ref: str) -> ExtractionResult:
        return ExtractionResult(
            entities=[{"name": "Alice", "type": "Person"}],
            relations=[
                {
                    "subject": "Alice",
                    "predicate": "OWNS",
                    "object": "Orion",
                    "confidence": 0.9,
                }
            ],
            extractor_version="rel-v1",
        )


def test_preflight_rejects_an_extractor_that_yields_no_relations() -> None:
    """A valid-but-empty extractor must not be allowed to start a sweep."""
    from ocm.evaluation.rahgm.run_ocmr_arm import _preflight

    with pytest.raises(SystemExit) as excinfo:
        _preflight(_Empty())
    assert "no relations" in str(excinfo.value)


def test_preflight_accepts_a_working_extractor() -> None:
    from ocm.evaluation.rahgm.run_ocmr_arm import _preflight

    _preflight(_Relational())  # must not raise


def test_preflight_is_a_no_op_for_the_offline_mock() -> None:
    from ocm.evaluation.rahgm.run_ocmr_arm import _preflight

    _preflight(None)


def test_preflight_aborts_on_an_environment_fault() -> None:
    from ocm.evaluation.rahgm.run_ocmr_arm import _preflight

    with pytest.raises(SystemExit) as excinfo:
        _preflight(StrictExtractor(_Boom(REAL_TRITON_MESSAGE)))
    assert "Python.h" in str(excinfo.value)
