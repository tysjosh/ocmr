"""Smoke tests for the property-based test harness itself (task 1.1).

These verify the tagging convention, the Hypothesis 100-iteration floor, and
the deterministic/offline settings fixture without depending on any module
that lands in a later task.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from ocm.tests.markers import (
    FEATURE_NAME,
    MIN_PROPERTY_ITERATIONS,
    pbt_property,
    property_tag,
)


def test_property_tag_format() -> None:
    assert property_tag(1, "Schema round-trip identity") == (
        "Feature: ontology-constrained-memory, "
        "Property 1: Schema round-trip identity"
    )
    assert FEATURE_NAME == "ontology-constrained-memory"


def test_hypothesis_profile_enforces_minimum_iterations() -> None:
    assert settings().max_examples >= MIN_PROPERTY_ITERATIONS


def test_deterministic_settings_kwargs(deterministic_settings_kwargs) -> None:
    assert deterministic_settings_kwargs == {
        "deterministic_test_mode": True,
        "chroma_mode": "memory",
        "extractor": "mock",
    }


def test_deterministic_settings_fixture(deterministic_settings) -> None:
    assert deterministic_settings.deterministic_test_mode is True
    assert deterministic_settings.chroma_mode == "memory"
    assert deterministic_settings.extractor == "mock"


@pbt_property(0, "harness self-check: tag and marker are applied")
@given(st.integers())
def test_pbt_property_decorator_tags_docstring(_n: int) -> None:
    assert test_pbt_property_decorator_tags_docstring.__doc__.startswith(
        "Feature: ontology-constrained-memory, Property 0:"
    )
