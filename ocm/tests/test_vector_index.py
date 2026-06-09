"""Unit tests for the semantic ``Vector_Index`` in in-memory mode (task 12.3).

Validates: Requirements 13.4, 13.6

These example-based tests exercise :class:`ocm.retrieval.vector_index.VectorIndex`
hermetically using the offline :class:`DeterministicEmbeddingProvider` and
``chroma_mode="memory"`` (Req 13.6 — an in-memory mode is supported for tests;
Req 13.4 — the same cosine-space add/query surface backs the on-disk Chroma
collection). The deterministic provider yields a stable 384-dim unit vector per
text, so identical text round-trips to a near-1.0 cosine similarity while
distinct text scores lower. The same metadata-filtering semantics
(``status`` / ``memory_type`` equality and ``$in`` / ``$or`` combinators) hold
whether Chroma or the pure-Python fallback backs the collection.

The four behaviours covered:

* **add/query round-trip** — items added are retrievable, and querying with a
  stored item's exact text returns that item ranked first with similarity ~1.0.
* **status metadata filtering** — ``where={"status": ...}`` constrains results
  to the requested status; the default (no filter) returns every status.
* **similarity conversion** — every returned ``similarity`` is in ``[0, 1]``,
  identical text scores ~1.0, and unrelated text scores strictly lower.
* **memory_type filtering** — ``where={"memory_type": ...}`` (and ``$in``)
  constrains results to the requested type(s).
"""

from __future__ import annotations

import pytest

from ocm.retrieval.embeddings import DeterministicEmbeddingProvider
from ocm.retrieval.vector_index import (
    MEMORY_TYPE_ASSERTION,
    MEMORY_TYPE_CLAIM,
    MEMORY_TYPE_DOCUMENT,
    STATUS_ACCEPTED,
    STATUS_QUARANTINED,
    VectorHit,
    VectorIndex,
)


@pytest.fixture
def index() -> VectorIndex:
    """A hermetic, offline in-memory vector index (Req 13.6)."""
    return VectorIndex(
        provider=DeterministicEmbeddingProvider(),
        chroma_mode="memory",
    )


def _ids(hits: list[VectorHit]) -> list[str]:
    return [hit.memory_id for hit in hits]


# ---------------------------------------------------------------------------
# Construction / configuration
# ---------------------------------------------------------------------------
def test_invalid_chroma_mode_rejected() -> None:
    """An unsupported chroma_mode is rejected with a clear error (Req 13.4, 13.6)."""
    with pytest.raises(ValueError):
        VectorIndex(provider=DeterministicEmbeddingProvider(), chroma_mode="bogus")


# ---------------------------------------------------------------------------
# 1. add/query round-trip
# ---------------------------------------------------------------------------
def test_add_query_round_trip_returns_exact_item_first(index: VectorIndex) -> None:
    """Querying a stored item's exact text returns it first with similarity ~1.0."""
    index.add("claim:1", "Alice owns Project Orion", MEMORY_TYPE_CLAIM)
    index.add("claim:2", "Bob manages the data pipeline", MEMORY_TYPE_CLAIM)
    index.add("claim:3", "The quarterly budget was approved", MEMORY_TYPE_CLAIM)

    hits = index.query("Alice owns Project Orion", top_k=3)

    assert hits, "expected at least one hit for a stored query text"
    assert hits[0].memory_id == "claim:1"
    assert hits[0].text == "Alice owns Project Orion"
    assert hits[0].memory_type == MEMORY_TYPE_CLAIM
    assert hits[0].status == STATUS_ACCEPTED
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-6)


def test_query_returns_all_added_items_within_top_k(index: VectorIndex) -> None:
    """Every added item is retrievable when top_k covers the collection size."""
    index.add("m:1", "first memory item", MEMORY_TYPE_CLAIM)
    index.add("m:2", "second memory item", MEMORY_TYPE_DOCUMENT)
    index.add("m:3", "third memory item", MEMORY_TYPE_ASSERTION)

    hits = index.query("first memory item", top_k=10)

    assert set(_ids(hits)) == {"m:1", "m:2", "m:3"}


def test_add_is_idempotent_per_id(index: VectorIndex) -> None:
    """Re-adding the same memory_id replaces (upserts) rather than duplicating."""
    index.add("claim:1", "original text", MEMORY_TYPE_CLAIM)
    index.add("claim:1", "updated text", MEMORY_TYPE_CLAIM)

    hits = index.query("updated text", top_k=10)

    assert _ids(hits).count("claim:1") == 1
    assert hits[0].memory_id == "claim:1"
    assert hits[0].text == "updated text"


def test_query_with_non_positive_top_k_returns_empty(index: VectorIndex) -> None:
    """A non-positive top_k short-circuits to an empty result."""
    index.add("claim:1", "some text", MEMORY_TYPE_CLAIM)
    assert index.query("some text", top_k=0) == []
    assert index.query("some text", top_k=-5) == []


# ---------------------------------------------------------------------------
# 2. status metadata filtering
# ---------------------------------------------------------------------------
def _seed_mixed_status(index: VectorIndex) -> None:
    """Seed one accepted and one quarantined item with related text."""
    index.add("claim:accepted", "Alice owns Project Orion", MEMORY_TYPE_CLAIM, STATUS_ACCEPTED)
    index.add(
        "claim:quarantined",
        "Alice owns Project Orion",
        MEMORY_TYPE_CLAIM,
        STATUS_QUARANTINED,
    )


def test_status_filter_accepted_only(index: VectorIndex) -> None:
    """where={"status": "accepted"} returns only accepted items (Req 16.2)."""
    _seed_mixed_status(index)

    hits = index.query("Alice owns Project Orion", top_k=10, where={"status": STATUS_ACCEPTED})

    assert _ids(hits) == ["claim:accepted"]
    assert all(hit.status == STATUS_ACCEPTED for hit in hits)


def test_status_filter_quarantined_only(index: VectorIndex) -> None:
    """where={"status": "quarantined"} returns only quarantined items (Req 16.3)."""
    _seed_mixed_status(index)

    hits = index.query(
        "Alice owns Project Orion", top_k=10, where={"status": STATUS_QUARANTINED}
    )

    assert _ids(hits) == ["claim:quarantined"]
    assert all(hit.status == STATUS_QUARANTINED for hit in hits)


def test_default_no_filter_returns_all_statuses(index: VectorIndex) -> None:
    """Without a where filter, items of every status are returned."""
    _seed_mixed_status(index)

    hits = index.query("Alice owns Project Orion", top_k=10)

    assert set(_ids(hits)) == {"claim:accepted", "claim:quarantined"}
    assert {hit.status for hit in hits} == {STATUS_ACCEPTED, STATUS_QUARANTINED}


def test_status_in_filter_includes_multiple_statuses(index: VectorIndex) -> None:
    """An $in status filter returns every listed status."""
    _seed_mixed_status(index)

    hits = index.query(
        "Alice owns Project Orion",
        top_k=10,
        where={"status": {"$in": [STATUS_ACCEPTED, STATUS_QUARANTINED]}},
    )

    assert set(_ids(hits)) == {"claim:accepted", "claim:quarantined"}


# ---------------------------------------------------------------------------
# 3. similarity conversion (cosine distance -> [0, 1], higher == more similar)
# ---------------------------------------------------------------------------
def test_similarity_is_bounded_and_identical_text_is_one(index: VectorIndex) -> None:
    """Every similarity is in [0, 1]; identical text scores ~1.0."""
    index.add("claim:1", "the cat sat on the mat", MEMORY_TYPE_CLAIM)
    index.add("claim:2", "a completely different unrelated sentence", MEMORY_TYPE_CLAIM)

    hits = index.query("the cat sat on the mat", top_k=10)

    assert all(0.0 <= hit.similarity <= 1.0 for hit in hits)
    top = next(hit for hit in hits if hit.memory_id == "claim:1")
    assert top.similarity == pytest.approx(1.0, abs=1e-6)


def test_similarity_higher_for_more_similar_text(index: VectorIndex) -> None:
    """Exact-match text ranks strictly above unrelated text by similarity."""
    index.add("claim:match", "quarterly revenue grew by twelve percent", MEMORY_TYPE_CLAIM)
    index.add("claim:other", "the weather in Lagos is humid today", MEMORY_TYPE_CLAIM)

    hits = index.query("quarterly revenue grew by twelve percent", top_k=10)
    by_id = {hit.memory_id: hit.similarity for hit in hits}

    assert by_id["claim:match"] > by_id["claim:other"]
    assert by_id["claim:match"] == pytest.approx(1.0, abs=1e-6)


def test_results_ranked_by_descending_similarity(index: VectorIndex) -> None:
    """Hits come back nearest-first (non-increasing similarity)."""
    index.add("claim:1", "alpha beta gamma delta", MEMORY_TYPE_CLAIM)
    index.add("claim:2", "wholly unrelated content here", MEMORY_TYPE_CLAIM)
    index.add("claim:3", "another distinct unrelated phrase", MEMORY_TYPE_CLAIM)

    sims = [hit.similarity for hit in index.query("alpha beta gamma delta", top_k=10)]

    assert sims == sorted(sims, reverse=True)
    assert sims[0] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. memory_type filtering
# ---------------------------------------------------------------------------
def _seed_mixed_types(index: VectorIndex) -> None:
    index.add("claim:1", "shared embedding text", MEMORY_TYPE_CLAIM)
    index.add("doc:1", "shared embedding text", MEMORY_TYPE_DOCUMENT)
    index.add("assertion:1", "shared embedding text", MEMORY_TYPE_ASSERTION)


def test_memory_type_filter_returns_only_that_type(index: VectorIndex) -> None:
    """where={"memory_type": "claim"} returns only claim-typed items."""
    _seed_mixed_types(index)

    hits = index.query("shared embedding text", top_k=10, where={"memory_type": MEMORY_TYPE_CLAIM})

    assert _ids(hits) == ["claim:1"]
    assert all(hit.memory_type == MEMORY_TYPE_CLAIM for hit in hits)


def test_memory_type_in_filter_returns_listed_types(index: VectorIndex) -> None:
    """An $in memory_type filter returns each listed type and excludes others."""
    _seed_mixed_types(index)

    hits = index.query(
        "shared embedding text",
        top_k=10,
        where={"memory_type": {"$in": [MEMORY_TYPE_CLAIM, MEMORY_TYPE_DOCUMENT]}},
    )

    assert set(_ids(hits)) == {"claim:1", "doc:1"}
    assert "assertion:1" not in _ids(hits)


def test_combined_status_and_memory_type_filter(index: VectorIndex) -> None:
    """An $or/equality combination filters on both status and memory_type."""
    index.add("claim:a", "shared embedding text", MEMORY_TYPE_CLAIM, STATUS_ACCEPTED)
    index.add("claim:q", "shared embedding text", MEMORY_TYPE_CLAIM, STATUS_QUARANTINED)
    index.add("doc:a", "shared embedding text", MEMORY_TYPE_DOCUMENT, STATUS_ACCEPTED)

    hits = index.query(
        "shared embedding text",
        top_k=10,
        where={"$and": [{"status": STATUS_ACCEPTED}, {"memory_type": MEMORY_TYPE_CLAIM}]},
    )

    assert _ids(hits) == ["claim:a"]
