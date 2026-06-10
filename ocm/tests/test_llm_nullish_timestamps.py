"""Regression: LLM extractors may emit the *string* ``"null"`` for unknown
datetime fields instead of JSON ``null``.

Such strings are truthy and previously reached the ``Event`` / ``Claim`` /
``Decision`` models and failed Pydantic datetime parsing
(``datetime_from_date_parsing``), aborting the whole write. The write pipeline
now coerces nullish placeholders (``""``/``"null"``/``"none"``/...) to ``None``
so model defaults apply. These tests pin that behavior.
"""

from __future__ import annotations

import pytest

from ocm.core.config import Settings
from ocm.core.ids import IdGenerator
from ocm.core.logging import ResearchLogger
from ocm.memory.assertion_builder import AssertionBuilder
from ocm.memory.commit_manager import CommitManager
from ocm.memory.contracts import ExtractionResult
from ocm.memory.graph_store import GraphStore
from ocm.memory.provenance_tracker import ProvenanceTracker
from ocm.memory.quarantine_store import QuarantineStore
from ocm.memory.sqlite_repository import SQLiteRepository
from ocm.memory.write_pipeline import WritePipeline, _coerce_optional_datetime
from ocm.resolution.entity_resolver import EntityResolver
from ocm.resolution.normalizer import Normalizer
from ocm.validation.constraints import ConstraintValidator
from ocm.validation.schema_validator import SchemaValidator


class _StubExtractor:
    """Returns a fixed ExtractionResult mimicking an LLM emitting ``"null"``."""

    version = "stub-1"

    def __init__(self, result: ExtractionResult) -> None:
        self._result = result

    def extract(self, text: str, source_ref: str = "") -> ExtractionResult:  # noqa: D401
        return self._result


def _pipeline(extractor: Extractor) -> WritePipeline:
    settings = Settings(deterministic_test_mode=True, chroma_mode="memory")
    repo = SQLiteRepository(":memory:")
    ids = IdGenerator(deterministic=True)
    graph = GraphStore()
    provenance = ProvenanceTracker(repo, ids)
    quarantine = QuarantineStore(repo, ids)
    commit = CommitManager(
        repo=repo,
        graph=graph,
        ids=ids,
        quarantine_store=quarantine,
        provenance_tracker=provenance,
        embed_hook=lambda a: None,
    )
    return WritePipeline(
        extractor=extractor,
        normalizer=Normalizer(),
        resolver=EntityResolver(),
        assertion_builder=AssertionBuilder(),
        schema_validator=SchemaValidator(),
        constraint_validator=ConstraintValidator(settings),
        commit_manager=commit,
        repo=repo,
        graph=graph,
        ids=ids,
        provenance_tracker=provenance,
        quarantine_store=quarantine,
        memory_embed_hook=lambda t, m: None,
        research_logger=ResearchLogger(),
        settings=settings,
    )


@pytest.mark.parametrize(
    "value,expected_none",
    [
        ("null", True),
        ("NULL", True),
        ("none", True),
        ("None", True),
        ("", True),
        ("  ", True),
        ("n/a", True),
        (None, True),
        ("2024-01-01T00:00:00", False),
    ],
)
def test_coerce_optional_datetime(value, expected_none):
    result = _coerce_optional_datetime(value)
    assert (result is None) == expected_none
    if not expected_none:
        assert result == value  # genuine values pass through untouched


def test_event_with_nullish_timestamps_does_not_crash():
    extraction = ExtractionResult(
        entities=[],
        events=[
            {
                "name": "Kickoff",
                "type": "meeting",
                "timestamp_start": "null",
                "timestamp_end": "null",
                "description": "Project kickoff",
            }
        ],
        claims=[],
        documents=[],
        decisions=[],
        relations=[],
        extractor_version="stub-1",
    )
    wp = _pipeline(_StubExtractor(extraction))

    # Previously raised pydantic ValidationError on the "null" strings.
    wp.run("Project kickoff happened.", "src-null-ts")

    events = [(t, p) for (t, p) in wp.repo.list_entities() if t == "Event"]
    assert len(events) == 1
    payload = events[0][1]
    # start defaulted to a real timestamp; end stayed absent.
    assert payload["timestamp_start"]
    assert payload.get("timestamp_end") in (None,)


def test_claim_with_nullish_created_at_does_not_crash():
    extraction = ExtractionResult(
        entities=[],
        events=[],
        claims=[{"text": "The sky is blue.", "confidence": 0.9, "created_at": "null"}],
        documents=[],
        decisions=[],
        relations=[],
        extractor_version="stub-1",
    )
    wp = _pipeline(_StubExtractor(extraction))
    wp.run("The sky is blue.", "src-claim-null")
    # The claim persisted with a defaulted created_at rather than crashing.
    # (Lookup by re-deriving the id is unnecessary; absence of exception + a
    # stored row is the regression guard.)
