"""Shared property-based test tagging convention for OCM.

Every Hypothesis property test in this suite is tagged with a stable,
greppable label so the design's correctness properties map 1:1 onto the
tests that validate them. The convention is:

    Feature: ontology-constrained-memory, Property {n}: {text}

Use :func:`pbt_property` to both register a pytest marker and stamp the
canonical tag onto a test's docstring in one step::

    from ocm.tests.markers import pbt_property

    @pbt_property(1, "Schema round-trip identity")
    @given(...)
    def test_schema_round_trip(...):
        ...

The marker lets you select property tests with ``pytest -m property`` and
carries the property number/text as marker args for tooling, while the
docstring tag makes the link visible in test output and source grep.
"""

from __future__ import annotations

from typing import Callable, TypeVar

import pytest

#: Canonical feature name shared by every property tag.
FEATURE_NAME = "ontology-constrained-memory"

#: Minimum Hypothesis iterations every property test must run (Req 27, design
#: "Correctness Properties"). The ``ocm`` Hypothesis profile registered in
#: ``conftest.py`` enforces this as ``max_examples``.
MIN_PROPERTY_ITERATIONS = 100

F = TypeVar("F", bound=Callable[..., object])


def property_tag(n: int, text: str) -> str:
    """Return the canonical tag string for property ``n``.

    >>> property_tag(1, "Schema round-trip identity")
    'Feature: ontology-constrained-memory, Property 1: Schema round-trip identity'
    """
    return f"Feature: {FEATURE_NAME}, Property {n}: {text}"


def pbt_property(n: int, text: str) -> Callable[[F], F]:
    """Decorator: tag a test as validating correctness Property ``n``.

    Applies the ``property`` pytest marker (carrying ``n`` and ``text`` as
    args) and prepends the canonical :func:`property_tag` line to the test's
    docstring so the property link is visible in both grep and test reports.
    """

    tag = property_tag(n, text)

    def decorate(func: F) -> F:
        existing = func.__doc__
        func.__doc__ = f"{tag}\n\n{existing.strip()}" if existing else tag
        return pytest.mark.property(n, text)(func)

    return decorate
