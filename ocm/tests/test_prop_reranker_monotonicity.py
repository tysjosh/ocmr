"""Property 9: Reranker contradiction monotonicity (Feature: ontology-constrained-memory).

Validates Requirements 17.1, 17.2, 17.3.

The reranker (R3) scores each candidate with::

    score = alpha*semantic_similarity + beta*graph_relevance + gamma*confidence
          + delta*provenance_quality + eta*recency - lambda*contradiction_penalty

with default weights ``alpha=0.40, beta=0.25, gamma=0.15, delta=0.10,
eta=0.05, lambda=0.30`` (Req 17.1, 17.2). Because ``lambda = 0.30 > 0``, an item
carrying a positive ``contradiction_penalty`` must score *strictly* lower than an
otherwise-identical item whose penalty is ``0`` (Req 17.3, contradiction
monotonicity).

This module asserts that property two ways:

* **Pure scoring half** — :meth:`Reranker.score_components` over two component
  maps that differ only in ``contradiction_penalty`` (``0`` vs ``p > 0``) yields
  exactly ``score(0) - lambda*p`` and therefore a strictly lower number.
* **Full pipeline half** — :meth:`Reranker.rerank` over two semantic hits whose
  raw signals are pinned identical (via ``metadata``) ranks the contradicted hit
  (marked via ``contradicted_ids`` or a ``quarantined`` status) strictly lower.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ocm.core.config import RerankWeights
from ocm.retrieval.reranker import Reranker
from ocm.tests.markers import pbt_property

# Default weights (Req 17.2): lambda_ = 0.30 drives the penalty term.
_WEIGHTS = RerankWeights()

# Raw signals live in [0, 1]; contradiction penalty is strictly positive in (0, 1].
_unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
# A graded contradiction penalty is a meaningful magnitude in (0, 1]. We bound it
# away from the smallest representable subnormals so ``lambda * penalty`` cannot
# underflow to 0.0 — that degenerate floating-point case is outside the realistic
# input space (penalties are derived scores, never ~1e-300 dust).
_MIN_PENALTY = 1e-6
_positive_penalty = st.floats(
    min_value=_MIN_PENALTY,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)


def _components(
    *,
    semantic_similarity: float,
    graph_relevance: float,
    confidence: float,
    provenance_quality: float,
    recency: float,
    contradiction_penalty: float,
) -> dict[str, float]:
    """Assemble a raw-signal component map in the reranker's expected shape."""
    return {
        "semantic_similarity": semantic_similarity,
        "graph_relevance": graph_relevance,
        "confidence": confidence,
        "provenance_quality": provenance_quality,
        "recency": recency,
        "contradiction_penalty": contradiction_penalty,
    }


@pbt_property(9, "Reranker contradiction monotonicity")
@given(
    semantic_similarity=_unit,
    graph_relevance=_unit,
    confidence=_unit,
    provenance_quality=_unit,
    recency=_unit,
    penalty=_positive_penalty,
)
def test_score_components_penalty_lowers_score_by_lambda(
    semantic_similarity: float,
    graph_relevance: float,
    confidence: float,
    provenance_quality: float,
    recency: float,
    penalty: float,
) -> None:
    """A positive contradiction penalty lowers the score by exactly lambda*p.

    Validates Requirements 17.1, 17.2, 17.3.
    """
    base_signals = dict(
        semantic_similarity=semantic_similarity,
        graph_relevance=graph_relevance,
        confidence=confidence,
        provenance_quality=provenance_quality,
        recency=recency,
    )
    clean = _components(**base_signals, contradiction_penalty=0.0)
    contradicted = _components(**base_signals, contradiction_penalty=penalty)

    clean_score = Reranker.score_components(clean, _WEIGHTS)
    contradicted_score = Reranker.score_components(contradicted, _WEIGHTS)

    # Exact algebraic relationship: only the -lambda*penalty term changes.
    assert contradicted_score == pytest.approx(clean_score - _WEIGHTS.lambda_ * penalty)
    # Monotonicity (Req 17.3): lambda = 0.30 > 0 and penalty > 0 ⇒ strictly lower.
    assert contradicted_score < clean_score


@pbt_property(9, "Reranker contradiction monotonicity")
@given(
    semantic_similarity=_unit,
    graph_relevance=_unit,
    confidence=_unit,
    provenance_quality=_unit,
    recency=_unit,
    mechanism=st.sampled_from(("contradicted_ids", "quarantined")),
)
def test_rerank_ranks_contradicted_item_strictly_lower(
    semantic_similarity: float,
    graph_relevance: float,
    confidence: float,
    provenance_quality: float,
    recency: float,
    mechanism: str,
) -> None:
    """In the full rerank path, an otherwise-identical contradicted item ranks lower.

    Two semantic hits carry identical pinned raw signals; one is flagged
    contradicted (via ``contradicted_ids`` or a ``quarantined`` status). The
    contradicted item must score strictly lower and sort after the clean one.

    Validates Requirements 17.1, 17.2, 17.3.
    """
    pinned = {
        "semantic_similarity": semantic_similarity,
        "graph_relevance": graph_relevance,
        "confidence": confidence,
        "provenance_quality": provenance_quality,
        "recency": recency,
    }
    metadata = {"mem_clean": dict(pinned), "mem_dirty": dict(pinned)}

    # The contradicted hit only differs in its contradiction status, never its
    # raw signals — its similarity/text/etc. mirror the clean hit exactly.
    dirty_status = "quarantined" if mechanism == "quarantined" else "accepted"
    semantic = [
        {"memory_id": "mem_clean", "memory_type": "claim", "status": "accepted",
         "similarity": semantic_similarity},
        {"memory_id": "mem_dirty", "memory_type": "claim", "status": dirty_status,
         "similarity": semantic_similarity},
    ]
    contradicted_ids = ["mem_dirty"] if mechanism == "contradicted_ids" else None

    reranker = Reranker(_WEIGHTS)
    ranked = reranker.rerank(
        semantic=semantic,
        contradicted_ids=contradicted_ids,
        metadata=metadata,
    )

    by_id = {item.memory_id: item for item in ranked}
    clean = by_id["mem_clean"]
    dirty = by_id["mem_dirty"]

    # The clean item has no penalty; the dirty item is penalized.
    assert clean.components["contradiction_penalty"] == 0.0
    assert dirty.components["contradiction_penalty"] > 0.0
    assert dirty.contradicted is True

    # Monotonicity (Req 17.3): contradicted scores strictly lower ...
    assert dirty.score < clean.score
    # ... and the exact gap is lambda * penalty (Req 17.1, 17.2).
    expected_gap = _WEIGHTS.lambda_ * dirty.components["contradiction_penalty"]
    assert clean.score - dirty.score == pytest.approx(expected_gap)
    # ... so the clean item sorts ahead of the contradicted one.
    assert ranked.index(clean) < ranked.index(dirty)
