"""Behavioral API tests for the five memory endpoints (task 15.4).

These go beyond the endpoint-shape checks in ``test_api_endpoints.py`` (task
15.2) and exercise the *behavior* of each route end to end against a
deterministic, in-memory :class:`~ocm.core.container.CoreContainer` wired into
:func:`ocm.app.main.create_app` via FastAPI's
:class:`~starlette.testclient.TestClient`:

* ``POST /memory/write`` returns accepted outcomes + a consistent summary, and a
  contradiction write produces a quarantined outcome (Req 28.1).
* ``POST /memory/query`` returns **both** symbolic (exact) and semantic
  (non-exact) results in ``retrieved_items`` plus supporting evidence
  (Req 28.2, 28.7).
* ``POST /memory/validate`` returns a verdict **without mutating state** — the
  Graph_Store edge count and the durable assertion rows are unchanged
  (Req 19.4).
* ``GET /memory/entity/{id}`` returns the entity payload, its type, and the
  assertions it participates in (Req 19.5).
* ``GET /memory/conflicts`` returns unresolved conflicts and quarantined
  candidates after a contradiction write (Req 19.6, 28.8).
* A service-start smoke test: :func:`create_app` builds and exposes the five
  routes, and a simple GET succeeds (Req 28.1).

The deterministic container uses an in-memory SQLite repo and in-memory vector
index, so the stateful tests never touch disk. Importing :mod:`ocm.app.main`
does run its module-level ``app = create_app()`` (default settings), which can
create ``ocm.db`` / ``.chroma`` on disk; the module-scoped cleanup fixture
removes any such artifacts after the tests run.

Requirements: 28.1, 28.2, 28.7, 28.8, 19.4, 19.5, 19.6.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ocm.app.main import create_app
from ocm.core.config import Settings
from ocm.core.container import CoreContainer

# Workspace root (…/ocmr) — where the default-settings module-level app would
# write its on-disk artifacts.
_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def _cleanup_disk_artifacts():
    """Remove any on-disk ``ocm.db`` / ``.chroma`` created at import time."""
    yield
    for name in ("ocm.db", ".chroma"):
        target = _ROOT / name
        try:
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif target.exists():
                target.unlink()
        except OSError:  # pragma: no cover - best-effort cleanup
            pass


def _build_client_and_container() -> tuple[TestClient, CoreContainer]:
    """A TestClient over an app wired to a deterministic, in-memory container."""
    settings = Settings(
        deterministic_test_mode=True, chroma_mode="memory", extractor="mock"
    )
    container = CoreContainer(settings)
    return TestClient(create_app(container)), container


@pytest.fixture
def client_and_container() -> tuple[TestClient, CoreContainer]:
    return _build_client_and_container()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _first_entity_id(container: CoreContainer, entity_type: str) -> str:
    """Return the first graph node id of ``entity_type`` (or fail the test)."""
    for node_id in container.graph.node_ids():
        if container.graph.get_entity_type(node_id) == entity_type:
            return node_id
    pytest.fail(f"no {entity_type} entity found in the graph")


def _accepted_predicates(write_body: dict) -> set[str]:
    return {o["candidate"]["predicate"] for o in write_body["accepted"]}


def _write(client: TestClient, text: str, source_ref: str) -> dict:
    resp = client.post(
        "/memory/write", json={"text": text, "source_ref": source_ref}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# --------------------------------------------------------------------------- #
# 1. POST /memory/write — accepted + summary (Req 28.1)
# --------------------------------------------------------------------------- #
def test_write_returns_accepted_and_consistent_summary(client_and_container):
    """A valid write returns accepted outcomes with a consistent summary (Req 28.1)."""
    client, _ = client_and_container
    body = _write(client, "Alice owns Project Orion. Bob is assigned to Task T1.", "src-1")

    # OWNS + ASSIGNED_TO are accepted (Req 28.1: write returns accepted results).
    assert {"OWNS", "ASSIGNED_TO"} <= _accepted_predicates(body)
    assert len(body["accepted"]) >= 2

    summary = body["summary"]
    # Summary counts mirror the outcome lists exactly.
    assert summary["num_accepted"] == len(body["accepted"])
    assert summary["num_superseded"] == len(body["superseded"])
    assert summary["num_quarantined"] == len(body["quarantined"])
    assert summary["num_rejected"] == len(body["rejected"])
    assert summary["num_candidates"] == (
        len(body["accepted"])
        + len(body["superseded"])
        + len(body["quarantined"])
        + len(body["rejected"])
    )


# --------------------------------------------------------------------------- #
# 1b. POST /memory/write — quarantined on contradiction (Req 28.1)
# --------------------------------------------------------------------------- #
def test_write_quarantines_status_contradiction(client_and_container):
    """A high-confidence contradiction is quarantined, not silently accepted (Req 28.1)."""
    client, _ = client_and_container
    _write(client, "Alice owns Project Orion. Bob is assigned to Task T1.", "src-1")
    # T1 becomes ``done`` via the completion event.
    _write(client, "Bob completed Task T1.", "src-2")
    # "not started" contradicts the accepted ``done`` status → quarantined.
    body = _write(client, "Task T1 is not started.", "src-3")

    assert len(body["quarantined"]) == 1
    assert body["quarantined"][0]["decision"] == "quarantined"
    assert body["summary"]["num_quarantined"] >= 1
    assert "status contradiction" in (body["quarantined"][0]["reason"] or "")


# --------------------------------------------------------------------------- #
# 2. POST /memory/query — symbolic + semantic results (Req 28.2, 28.7)
# --------------------------------------------------------------------------- #
def test_query_returns_symbolic_and_semantic_results(client_and_container):
    """Query returns both symbolic (exact) and semantic results + evidence (Req 28.2, 28.7)."""
    client, _ = client_and_container
    _write(client, "Alice owns Project Orion. Bob is assigned to Task T1.", "src-1")

    resp = client.post(
        "/memory/query", json={"query": "Who is assigned to Task T1?", "top_k": 10}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    items = body["retrieved_items"]
    assert items, "query returned no retrieved items"
    # Req 28.7 — the merged candidate set carries BOTH a symbolic (exact) hit
    # and at least one semantic (non-exact) hit.
    assert any(item["exact_match"] for item in items), "expected a symbolic exact match"
    assert any(not item["exact_match"] for item in items), "expected a semantic (non-exact) hit"

    # The symbolic hit is the ASSIGNED_TO assignment edge.
    assert any(item.get("predicate") == "ASSIGNED_TO" for item in items if item["exact_match"])

    # Req 28.2 / 28.8 — supporting evidence (ids + confidence) accompanies the answer.
    assert body["supporting_assertions"], "expected supporting assertions"
    for sa in body["supporting_assertions"]:
        assert sa["id"]
        assert 0.0 <= sa["confidence"] <= 1.0


# --------------------------------------------------------------------------- #
# 3. POST /memory/validate — no state mutation (Req 19.4)
# --------------------------------------------------------------------------- #
def test_validate_does_not_mutate_state(client_and_container):
    """Validate returns a verdict without writing to graph or storage (Req 19.4)."""
    client, container = client_and_container
    _write(client, "Alice owns Project Orion. Bob is assigned to Task T1.", "src-1")

    person_id = _first_entity_id(container, "Person")
    project_id = _first_entity_id(container, "Project")

    edges_before = container.graph.num_edges()
    nodes_before = container.graph.num_nodes()
    assertions_before = len(container.repo.list_assertions())

    resp = client.post(
        "/memory/validate",
        json={
            "candidate": {
                "subject_id": person_id,
                "predicate": "OWNS",
                "object_id": project_id,
                "confidence": 0.9,
                "source_ref": "src-validate",
            }
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert {"valid", "decision", "reason", "severity", "failed_check", "conflicting_ids"} <= set(body)
    assert body["decision"] in {"accept", "supersede", "quarantine", "reject"}

    # Nothing was committed: graph + durable storage are unchanged (Req 19.4).
    assert container.graph.num_edges() == edges_before
    assert container.graph.num_nodes() == nodes_before
    assert len(container.repo.list_assertions()) == assertions_before


# --------------------------------------------------------------------------- #
# 4. GET /memory/entity/{id} — entity + assertions (Req 19.5)
# --------------------------------------------------------------------------- #
def test_entity_returns_entity_and_assertions(client_and_container):
    """Entity endpoint returns the entity, its type, and its assertions (Req 19.5)."""
    client, container = client_and_container
    _write(client, "Alice owns Project Orion. Bob is assigned to Task T1.", "src-1")

    person_id = _first_entity_id(container, "Person")
    resp = client.get(f"/memory/entity/{person_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["entity_type"] == "Person"
    assert body["entity"]["id"] == person_id
    # Alice/Bob each participate in at least one accepted assertion.
    assert isinstance(body["assertions"], list)
    assert len(body["assertions"]) >= 1
    for assertion in body["assertions"]:
        assert person_id in (assertion["subject_id"], assertion["object_id"])


def test_entity_returns_404_for_unknown_entity(client_and_container):
    """Entity endpoint 404s for an unknown id (Req 19.5)."""
    client, _ = client_and_container
    resp = client.get("/memory/entity/nope-does-not-exist")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# 5. GET /memory/conflicts — unresolved + quarantined (Req 19.6, 28.8)
# --------------------------------------------------------------------------- #
def test_conflicts_returns_unresolved_and_quarantined(client_and_container):
    """Conflicts endpoint surfaces unresolved conflicts + quarantines (Req 19.6, 28.8)."""
    client, _ = client_and_container

    # No conflicts before any contradiction.
    empty = client.get("/memory/conflicts").json()
    assert empty["unresolved_conflicts"] == []
    assert empty["quarantined_candidates"] == []

    # Drive a status contradiction so a quarantine is persisted.
    _write(client, "Alice owns Project Orion. Bob is assigned to Task T1.", "src-1")
    _write(client, "Bob completed Task T1.", "src-2")
    _write(client, "Task T1 is not started.", "src-3")

    resp = client.get("/memory/conflicts")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert len(body["unresolved_conflicts"]) >= 1
    assert len(body["quarantined_candidates"]) >= 1

    # Req 28.8 — the curated conflict view carries ids, reason, severity, and
    # the conflicting ids it involves.
    conflict = body["unresolved_conflicts"][0]
    assert conflict["memory_id"]
    assert "status contradiction" in (conflict["reason"] or "")
    assert conflict["severity"]
    assert conflict["conflicting_ids"]

    # The raw persisted record mirrors the same unresolved status.
    record = body["quarantined_candidates"][0]
    assert record["status"] == "unresolved"


# --------------------------------------------------------------------------- #
# 6. Service-start smoke test (Req 28.1)
# --------------------------------------------------------------------------- #
def test_service_start_smoke():
    """create_app builds and exposes the five routes; a simple GET succeeds (Req 28.1)."""
    settings = Settings(
        deterministic_test_mode=True, chroma_mode="memory", extractor="mock"
    )
    app = create_app(CoreContainer(settings))

    paths = {getattr(route, "path", None) for route in app.routes}
    for expected in (
        "/memory/write",
        "/memory/query",
        "/memory/validate",
        "/memory/entity/{entity_id}",
        "/memory/conflicts",
    ):
        assert expected in paths, f"missing route {expected}"

    # The service answers a simple request (OpenAPI schema) once started.
    client = TestClient(app)
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert resp.json()["info"]["title"] == "Ontology-Constrained Memory API"

    # And a fresh service has no conflicts to report.
    conflicts = client.get("/memory/conflicts")
    assert conflicts.status_code == 200
    assert conflicts.json()["unresolved_conflicts"] == []
