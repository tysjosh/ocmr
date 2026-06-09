"""Endpoint-shape tests for the five memory routes (task 15.2).

These exercise :func:`ocm.app.main.create_app` end to end against a
deterministic, in-memory :class:`~ocm.core.container.CoreContainer` using
FastAPI's :class:`~starlette.testclient.TestClient`. They assert each endpoint
returns ``200`` with the expected response shape (Req 19.2–19.6, 28.1, 28.2,
28.7); deeper behavioral coverage belongs to task 15.4.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ocm.app.main import create_app
from ocm.core.config import Settings
from ocm.core.container import CoreContainer


@pytest.fixture
def client_and_container() -> tuple[TestClient, CoreContainer]:
    """A TestClient over an app wired to a deterministic, in-memory container."""
    settings = Settings(
        deterministic_test_mode=True, chroma_mode="memory", extractor="mock"
    )
    container = CoreContainer(settings)
    app = create_app(container)
    return TestClient(app), container


def test_write_endpoint_returns_outcome_lists_and_summary(client_and_container):
    """POST /memory/write returns the four outcome lists plus the summary (Req 19.2)."""
    client, _ = client_and_container
    resp = client.post(
        "/memory/write",
        json={"text": "Alice owns Project Orion.", "source_ref": "doc-1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    for key in ("accepted", "superseded", "quarantined", "rejected", "summary"):
        assert key in body
    summary = body["summary"]
    for key in (
        "num_candidates",
        "num_accepted",
        "num_quarantined",
        "num_rejected",
        "num_superseded",
    ):
        assert key in summary
    # The OWNS relation should be accepted.
    assert summary["num_accepted"] >= 1


def test_query_endpoint_returns_query_type_and_evidence_fields(client_and_container):
    """POST /memory/query returns query_type plus evidence fields (Req 19.3, 28.7)."""
    client, _ = client_and_container
    client.post(
        "/memory/write",
        json={"text": "Alice owns Project Orion.", "source_ref": "doc-1"},
    )
    resp = client.post("/memory/query", json={"query": "Who owns Project Orion?"})
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "query_type",
        "confidence",
        "supporting_assertions",
        "supporting_sources",
        "conflicts",
        "missing_information",
        "retrieved_items",
    ):
        assert key in body
    assert isinstance(body["query_type"], str)


def test_validate_endpoint_does_not_commit(client_and_container):
    """POST /memory/validate returns a verdict without writing (Req 19.4)."""
    client, container = client_and_container
    client.post(
        "/memory/write",
        json={"text": "Alice owns Project Orion.", "source_ref": "doc-1"},
    )
    # Find the existing Person and Project ids from the graph.
    person_id = next(
        (n for n in container.graph.node_ids() if container.graph.get_entity_type(n) == "Person"),
        None,
    )
    project_id = next(
        (n for n in container.graph.node_ids() if container.graph.get_entity_type(n) == "Project"),
        None,
    )
    assert person_id and project_id

    edges_before = container.graph.num_edges()
    resp = client.post(
        "/memory/validate",
        json={
            "candidate": {
                "subject_id": person_id,
                "predicate": "OWNS",
                "object_id": project_id,
                "confidence": 0.9,
                "source_ref": "doc-2",
            }
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(
        ["valid", "decision", "reason", "severity", "failed_check", "conflicting_ids"]
    ).issubset(body.keys())
    assert body["decision"] in {"accept", "supersede", "quarantine", "reject"}
    # No commit happened.
    assert container.graph.num_edges() == edges_before


def test_entity_endpoint_returns_entity_and_assertions(client_and_container):
    """GET /memory/entity/{id} returns the entity, type, and assertions (Req 19.5)."""
    client, container = client_and_container
    client.post(
        "/memory/write",
        json={"text": "Alice owns Project Orion.", "source_ref": "doc-1"},
    )
    person_id = next(
        n for n in container.graph.node_ids() if container.graph.get_entity_type(n) == "Person"
    )
    resp = client.get(f"/memory/entity/{person_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["entity_type"] == "Person"
    assert body["entity"]["id"] == person_id
    assert isinstance(body["assertions"], list)
    # Alice participates in the OWNS assertion.
    assert len(body["assertions"]) >= 1


def test_entity_endpoint_404_for_missing_entity(client_and_container):
    """GET /memory/entity/{id} returns 404 for an unknown entity (Req 19.5)."""
    client, _ = client_and_container
    resp = client.get("/memory/entity/does-not-exist")
    assert resp.status_code == 404


def test_conflicts_endpoint_returns_conflict_lists(client_and_container):
    """GET /memory/conflicts returns unresolved conflicts + quarantines (Req 19.6)."""
    client, _ = client_and_container
    resp = client.get("/memory/conflicts")
    assert resp.status_code == 200
    body = resp.json()
    assert "unresolved_conflicts" in body
    assert "quarantined_candidates" in body
    assert isinstance(body["unresolved_conflicts"], list)
    assert isinstance(body["quarantined_candidates"], list)
