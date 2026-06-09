"""Wiring tests for the LLM extractor path and real-embeddings selection.

These confirm the container selects the right components from configuration and
that the LLM extraction → governed write path works end-to-end with an injected
fake HTTP client (no network, no API key, no model download).
"""

from __future__ import annotations

import json

from ocm.core.container import CoreContainer
from ocm.evaluation import experiment as exp
from ocm.extraction.llm_extractor import LLMExtractor
from ocm.extraction.mock_extractor import MockExtractor
from ocm.retrieval.embeddings import DeterministicEmbeddingProvider, LocalEmbeddingProvider


# --------------------------------------------------------------------------- #
# Settings factory selection
# --------------------------------------------------------------------------- #
def test_make_settings_factory_offline_default():
    s = exp.make_settings_factory()()
    assert s.extractor == "mock"
    assert s.deterministic_test_mode is True
    assert s.sqlite_path == ":memory:" and s.chroma_mode == "memory"


def test_make_settings_factory_llm_and_local():
    s = exp.make_settings_factory(
        extractor="llm",
        embeddings="local",
        llm_base_url="https://api.example.com/v1",
        llm_api_key="sk-test",
        llm_model="gpt-4o-mini",
    )()
    assert s.extractor == "llm"
    assert s.llm_base_url == "https://api.example.com/v1"
    assert s.llm_model == "gpt-4o-mini"
    # Real embeddings imply a non-deterministic run, but storage stays hermetic.
    assert s.deterministic_test_mode is False
    assert s.sqlite_path == ":memory:" and s.chroma_mode == "memory"


# --------------------------------------------------------------------------- #
# Container component selection (no invocation -> fully offline)
# --------------------------------------------------------------------------- #
def test_container_selects_llm_extractor_and_local_embeddings():
    s = exp.make_settings_factory(
        extractor="llm", embeddings="local", llm_base_url="https://api.example.com/v1"
    )()
    c = CoreContainer(s)
    assert isinstance(c.extractor, LLMExtractor)
    # Real embeddings provider is selected (the model loads lazily on first use).
    assert isinstance(c.embeddings, LocalEmbeddingProvider)


def test_container_offline_default_selects_mock_and_deterministic():
    c = CoreContainer(exp.make_settings_factory()())
    assert isinstance(c.extractor, MockExtractor)
    assert isinstance(c.embeddings, DeterministicEmbeddingProvider)


# --------------------------------------------------------------------------- #
# End-to-end LLM extraction through the governed write path (fake client)
# --------------------------------------------------------------------------- #
def _fake_llm_client(payload):
    """Return a canned OpenAI-compatible response with a valid ExtractionResult."""
    content = json.dumps(
        {
            "entities": [
                {"type": "Person", "name": "Alice", "fields": {}},
                {"type": "Project", "name": "Orion", "fields": {}},
            ],
            "events": [],
            "claims": [{"text": "Alice owns Project Orion.", "confidence": 0.95}],
            "documents": [],
            "decisions": [],
            "relations": [
                {
                    "subject": "Alice",
                    "predicate": "OWNS",
                    "object": "Orion",
                    "confidence": 0.95,
                    "write_intent": "new_fact",
                }
            ],
        }
    )
    return {"choices": [{"message": {"content": content}}]}


def test_llm_extractor_path_writes_through_governed_pipeline():
    # Hermetic settings (deterministic embeddings + in-memory storage), but the
    # W1 stage is the real LLMExtractor wired to a fake transport.
    settings = exp.make_settings_factory(
        extractor="llm",
        embeddings="deterministic",
        llm_base_url="https://api.example.com/v1",
        llm_api_key="sk-test",
    )()
    llm = LLMExtractor(settings, client=_fake_llm_client)
    container = CoreContainer(settings, extractor=llm)

    result = container.write_pipeline.run("Alice owns Project Orion.", "src-1")

    accepted_preds = {o.candidate.predicate for o in result.accepted}
    assert "OWNS" in accepted_preds
    # The accepted OWNS edge is queryable through the retrieval pipeline.
    pkg = container.retrieval_pipeline.query("Who owns Project Orion?", top_k=5)
    assert pkg.answer == "Alice"


def test_llm_json_mode_toggle_controls_payload():
    """The JSON-mode setting governs whether response_format is sent."""
    base = exp.make_settings_factory(
        extractor="llm", embeddings="deterministic",
        llm_base_url="https://api.example.com/v1",
    )()
    on = LLMExtractor(base)._build_payload("text", "src")
    assert on["response_format"] == {"type": "json_object"}

    off_settings = exp.make_settings_factory(
        extractor="llm", embeddings="deterministic",
        llm_base_url="https://api.example.com/v1", llm_use_json_mode=False,
    )()
    off = LLMExtractor(off_settings)._build_payload("text", "src")
    assert "response_format" not in off


def test_local_qwen_style_config_wires_end_to_end():
    """A local OpenAI-compatible server config (e.g. Qwen2.5-32B-Instruct via
    vLLM) extracts through the governed pipeline with a fake transport."""
    settings = exp.make_settings_factory(
        extractor="llm",
        embeddings="deterministic",
        llm_base_url="http://localhost:8000/v1",
        llm_api_key="local",
        llm_model="Qwen/Qwen2.5-32B-Instruct",
    )()
    assert settings.llm_model == "Qwen/Qwen2.5-32B-Instruct"
    llm = LLMExtractor(settings, client=_fake_llm_client)
    container = CoreContainer(settings, extractor=llm)
    result = container.write_pipeline.run("Alice owns Project Orion.", "src-1")
    assert "OWNS" in {o.candidate.predicate for o in result.accepted}
