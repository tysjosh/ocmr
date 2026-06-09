"""FastAPI application factory for the OCM ``API_Service`` (Req 19.1).

:func:`create_app` builds the FastAPI application, wiring a single
:class:`~ocm.core.container.CoreContainer` onto ``app.state.container`` so the
endpoint dependency (:func:`ocm.app.api.routes.get_container`) can resolve it.
Tests pass a pre-built (deterministic, in-memory) container; production code
calls :func:`create_app` with no arguments and a default container is
constructed from :class:`~ocm.core.config.Settings`.

The five production endpoints live on the :data:`ocm.app.api.routes.router`. A
non-production ``routes_debug`` router (task 15.3) is mounted **only** when
``settings.deterministic_test_mode`` is set (or an explicit debug flag is
passed). It is imported defensively so the service still starts before that
router exists.

A module-level ``app = create_app()`` is exposed for ``uvicorn ocm.app.main:app``.

Requirements: 19.1, 28.1, 28.2.
"""

from __future__ import annotations

from fastapi import FastAPI

from ocm.app.api.routes import router as memory_router
from ocm.core.config import Settings
from ocm.core.container import CoreContainer


def create_app(
    container: CoreContainer | None = None,
    *,
    enable_debug_routes: bool | None = None,
) -> FastAPI:
    """Build and return the OCM FastAPI application (Req 19.1).

    Args:
        container: A pre-wired :class:`CoreContainer` (tests inject a
            deterministic, in-memory one). When omitted a default container is
            constructed from ``Settings()`` (offline-first defaults, Req 27.2).
        enable_debug_routes: Force-enable/disable the ``routes_debug`` router.
            When ``None`` (default) the debug router is mounted whenever
            ``settings.deterministic_test_mode`` is set.

    Returns:
        The configured :class:`fastapi.FastAPI` application with the container
        on ``app.state.container`` and the memory router included.
    """
    if container is None:
        container = CoreContainer(Settings())

    app = FastAPI(
        title="Ontology-Constrained Memory API",
        description=(
            "Ontology-constrained memory for agent reasoning: validated writes, "
            "evidence-packaged retrieval, and governance-aware conflict reporting."
        ),
        version="0.1.0",
    )
    app.state.container = container

    # Five production endpoints (Req 19.2–19.6).
    app.include_router(memory_router)

    # Non-production inspection endpoints (task 15.3), mounted only in debug /
    # deterministic-test mode. Imported defensively so the service still starts
    # if routes_debug has not landed yet.
    if enable_debug_routes is None:
        enable_debug_routes = bool(
            getattr(container.settings, "deterministic_test_mode", False)
        )
    if enable_debug_routes:
        _include_debug_routes(app)

    return app


def _include_debug_routes(app: FastAPI) -> None:
    """Include the ``routes_debug`` router when it is available (task 15.3)."""
    try:
        from ocm.app.api.routes_debug import router as debug_router
    except Exception:  # pragma: no cover - router not implemented yet (task 15.3)
        return
    app.include_router(debug_router)


# Module-level app for ``uvicorn ocm.app.main:app``.
app = create_app()
