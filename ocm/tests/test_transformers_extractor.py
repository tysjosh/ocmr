"""Tests for the in-process (HF transformers) Qwen extractor.

Uses a fake ``complete`` callable so the extraction -> governed-write path is
exercised offline, without loading a real model.
"""

from __future__ import annotations

import json

import pytest

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.extraction.base import ExtractionError
from ocm.extraction.transformers_extractor import TransformersExtractor, _loads_lenient

_VALID_JSON = {
    "entities": [
        {"type": "Person", "name": "Alice", "fields": {}},
        {"type": "Project", "name": "Orion", "fields": {}},
    ],
    "events": [],
    "claims": [{"text": "Alice owns Project Orion.", "confidence": 0.95}],
    "documents": [],
    "decisions": [],
    "relations": [
        {"subject": "Alice", "predicate": "OWNS", "object": "Orion",
         "confidence": 0.95, "write_intent": "new_fact"},
    ],
}


def _fake_complete(_messages):
    # Qwen models often wrap JSON in a Markdown fence + prose; the parser must cope.
    return "Sure, here is the extraction:\n```json\n" + json.dumps(_VALID_JSON) + "\n```"


def test_loads_lenient_strips_fences_and_prose():
    raw = "blah\n```json\n{\"a\": 1}\n```\ntrailing"
    assert _loads_lenient(raw) == {"a": 1}


def test_loads_lenient_raises_without_json():
    with pytest.raises(ExtractionError):
        _loads_lenient("no json here")


def test_requires_model_or_complete():
    with pytest.raises(ValueError):
        TransformersExtractor()  # neither complete nor (model, tokenizer)


def test_transformers_extractor_parses_and_returns_result():
    ex = TransformersExtractor(complete=_fake_complete)
    result = ex.extract("Alice owns Project Orion.", "src-1")
    preds = {r["predicate"] for r in result.relations}
    assert "OWNS" in preds
    assert result.extractor_version == "qwen-transformers-v1"


def test_transformers_extractor_drives_governed_pipeline():
    settings = Settings(deterministic_test_mode=True, chroma_mode="memory")
    ex = TransformersExtractor(complete=_fake_complete)
    container = CoreContainer(settings, extractor=ex)

    r = container.write_pipeline.run("Alice owns Project Orion.", "src-1")
    assert "OWNS" in {o.candidate.predicate for o in r.accepted}
    assert container.retrieval_pipeline.query("Who owns Project Orion?", top_k=5).answer == "Alice"


def test_transformers_extractor_bad_output_is_rejected():
    settings = Settings(deterministic_test_mode=True, chroma_mode="memory")
    ex = TransformersExtractor(complete=lambda _m: "I cannot help with that.")
    container = CoreContainer(settings, extractor=ex)
    # Malformed (no JSON) -> graceful validation failure, empty result.
    r = container.write_pipeline.run("anything", "src-1")
    assert r.summary.num_candidates == 0 and r.accepted == []


def test_default_complete_uses_chat_template_and_generate():
    """A lightweight fake model/tokenizer exercises the default HF code path."""

    class FakeTok:
        eos_token_id = 0

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "PROMPT"

        def __call__(self, prompts, return_tensors=None):
            # Mimic a BatchEncoding with a .to() and dict access; input_ids
            # exposes a tensor-like .shape so the extractor can slice the prompt.
            class _Ids:
                shape = (1, 3)

            class BE(dict):
                def to(self, _device):
                    return self
            return BE(input_ids=_Ids())

        def decode(self, tokens, skip_special_tokens=True):
            return json.dumps(_VALID_JSON)

    class FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            # Return prompt(3) + new tokens; the extractor slices off the prompt.
            return [[1, 2, 3, 9, 9]]

    ex = TransformersExtractor(model=FakeModel(), tokenizer=FakeTok())
    result = ex.extract("Alice owns Project Orion.", "src-1")
    assert "OWNS" in {r["predicate"] for r in result.relations}
