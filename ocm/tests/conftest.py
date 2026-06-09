"""Shared pytest configuration and fixtures for the OCM test suite.

This module makes every later test **hermetic and offline** by default:

* A Hypothesis profile (``ocm``) pins a minimum of 100 examples per property
  and disables the deadline so property tests are stable in CI.
* ``deterministic_settings`` builds the canonical offline configuration
  (``deterministic_test_mode=True``, ``chroma_mode="memory"``,
  ``extractor="mock"``) so IDs are reproducible, the vector index is
  in-memory, and no network/API key is required (Req 27.2, 27.5, 13.6, 3.4).

The configuration module (``ocm.core.config.Settings``) and the dependency
container (``ocm.core.container.CoreContainer``) are created in later tasks.
To keep collection working today, every fixture imports those symbols
**lazily** inside its body: if a symbol is missing the fixture either falls
back to a plain namespace (settings) or skips the test (container/repository)
with a clear reason, rather than breaking collection of the whole suite.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import pytest
from hypothesis import HealthCheck, settings

from ocm.tests.markers import MIN_PROPERTY_ITERATIONS

# ---------------------------------------------------------------------------
# Hypothesis: minimum 100 iterations per property, deterministic & CI-friendly.
# ---------------------------------------------------------------------------
settings.register_profile(
    "ocm",
    max_examples=MIN_PROPERTY_ITERATIONS,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.load_profile("ocm")


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``property`` marker used by :func:`ocm.tests.markers.pbt_property`."""
    config.addinivalue_line(
        "markers",
        "property(n, text): tags a test as validating correctness Property n "
        "(Feature: ontology-constrained-memory, Property n: text).",
    )


# ---------------------------------------------------------------------------
# Canonical offline / deterministic configuration.
# ---------------------------------------------------------------------------
#: Keyword arguments that define the hermetic test configuration. Kept as a
#: plain dict so later tasks can pass them straight into ``Settings(**kwargs)``
#: or adapt them as the config model evolves.
DETERMINISTIC_SETTINGS_KWARGS: Dict[str, Any] = {
    "deterministic_test_mode": True,
    "chroma_mode": "memory",
    "extractor": "mock",
}


@pytest.fixture
def deterministic_settings_kwargs() -> Dict[str, Any]:
    """The canonical offline/deterministic settings as a kwargs dict."""
    return dict(DETERMINISTIC_SETTINGS_KWARGS)


@pytest.fixture
def deterministic_settings(deterministic_settings_kwargs: Dict[str, Any]) -> Any:
    """A deterministic, offline ``Settings`` object for hermetic tests.

    Instantiates ``ocm.core.config.Settings`` when it exists (task 1.3).
    Until then it returns a ``SimpleNamespace`` carrying the same fields so
    tests written against the attributes still work and collection never
    fails on a missing import.
    """
    try:
        from ocm.core.config import Settings  # type: ignore
    except Exception:
        return SimpleNamespace(**deterministic_settings_kwargs)
    return Settings(**deterministic_settings_kwargs)


@pytest.fixture
def in_memory_repository(deterministic_settings: Any) -> Any:
    """An in-memory ``StorageRepository`` for hermetic, offline tests.

    Skips until the repository layer (task 3.2) lands. The repository is
    constructed in SQLite ``:memory:`` mode so nothing touches disk.
    """
    try:
        from ocm.memory.sqlite_repository import SQLiteRepository  # type: ignore
    except Exception:
        pytest.skip("StorageRepository not implemented yet (task 3.2)")
    return SQLiteRepository(":memory:")


@pytest.fixture
def container(deterministic_settings: Any) -> Any:
    """A wired ``CoreContainer`` using the deterministic/offline settings.

    Skips until the dependency container (task 1.3 / API wiring) lands. When
    available it wires the Mock_Extractor, in-memory Chroma, and deterministic
    IDs from ``deterministic_settings`` so the whole pipeline is hermetic.
    """
    try:
        from ocm.core.container import CoreContainer  # type: ignore
    except Exception:
        pytest.skip("CoreContainer not implemented yet (task 1.3)")
    return CoreContainer(deterministic_settings)
