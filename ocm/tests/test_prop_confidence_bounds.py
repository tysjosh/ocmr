"""Property 2: Confidence always in [0,1] (Feature: ontology-constrained-memory).

Validates Requirements 1.6, 1.9, 8.7.

The :class:`~ocm.ontology.models.Claim` and
:class:`~ocm.ontology.models.Assertion` models declare their ``confidence``
field as ``confloat(ge=0.0, le=1.0)``. These property tests assert the two
halves of Property 2:

* *Acceptance half* — any float in [0, 1] is accepted and round-trips
  unchanged onto the constructed model.
* *Rejection half* — any float strictly outside [0, 1] (including NaN, which is
  never within the closed bound) makes construction raise a pydantic
  ``ValidationError`` so that no model is produced.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from ocm.ontology.models import Assertion, Claim
from ocm.tests.markers import pbt_property

# A fixed, valid timestamp so the tests isolate the confidence constraint.
_TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_claim(confidence: float) -> Claim:
    return Claim(
        id="c1",
        text="the sky is blue",
        source_ref="doc:1",
        confidence=confidence,
        created_at=_TS,
    )


def _make_assertion(confidence: float) -> Assertion:
    return Assertion(
        id="a1",
        subject_id="s1",
        predicate="relates_to",
        object_id="o1",
        confidence=confidence,
        source_ref="doc:1",
        created_at=_TS,
    )


# Floats strictly inside the closed unit interval [0, 1].
_in_range = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)

# Floats outside [0, 1]. The ranges are bounded strictly away from the closed
# interval's edges (the largest value below 0 and the smallest value above 1),
# so even Hypothesis's boundary probes stay out of range without needing a
# filter. NaN and ±inf round out the non-finite cases that can never satisfy a
# closed bound.
_NEXT_ABOVE_ONE = math.nextafter(1.0, math.inf)  # smallest float strictly > 1.0
_NEXT_BELOW_ZERO = math.nextafter(0.0, -math.inf)  # largest float strictly < 0.0

_below = st.floats(
    min_value=-1e308, max_value=_NEXT_BELOW_ZERO, allow_nan=False, allow_infinity=False
)
_above = st.floats(
    min_value=_NEXT_ABOVE_ONE, max_value=1e308, allow_nan=False, allow_infinity=False
)
_out_of_range = st.one_of(
    _below,
    _above,
    st.just(float("nan")),
    st.just(float("inf")),
    st.just(float("-inf")),
)


@pbt_property(2, "Confidence always in [0,1]")
@given(confidence=_in_range)
def test_in_range_confidence_is_accepted(confidence: float) -> None:
    claim = _make_claim(confidence)
    assertion = _make_assertion(confidence)

    assert claim.confidence == confidence
    assert assertion.confidence == confidence
    # The constructed value provably lies within the closed unit interval.
    assert 0.0 <= claim.confidence <= 1.0
    assert 0.0 <= assertion.confidence <= 1.0


@pbt_property(2, "Confidence always in [0,1]")
@given(confidence=_out_of_range)
def test_out_of_range_confidence_is_rejected(confidence: float) -> None:
    # Guard the generators: every example must be outside the closed bound.
    assert math.isnan(confidence) or not (0.0 <= confidence <= 1.0)

    with pytest.raises(ValidationError):
        _make_claim(confidence)
    with pytest.raises(ValidationError):
        _make_assertion(confidence)
