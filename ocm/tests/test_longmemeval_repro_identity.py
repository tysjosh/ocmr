"""Regression tests for LongMemEval Arm-B reproducibility identity."""

from __future__ import annotations

import hashlib
import json

import pytest

from ocm.evaluation.benchmark import BenchmarkExample, Question, Session
from ocm.evaluation.datasets.longmemeval_adapter import _fingerprint_examples
from run_7f_local import CachedChat


def _example(answer: str) -> BenchmarkExample:
    return BenchmarkExample(
        id="qid-1",
        category="knowledge_update_e2e",
        sessions=[Session(session_id="s0", input="User likes tea.")],
        questions=[
            Question(
                query="What does the user like?",
                expected_answer_contains=[answer],
                expected_conflict=False,
            )
        ],
    )


def test_extracted_example_fingerprint_changes_with_replayed_state() -> None:
    first = _fingerprint_examples([_example("tea")])
    second = _fingerprint_examples([_example("coffee")])

    assert first != second
    assert _fingerprint_examples([_example("tea")]) == first


def test_prompt_cache_refuses_a_different_namespace(tmp_path) -> None:
    path = tmp_path / "chat_cache.json"
    calls = {"n": 0}

    def factory():
        def chat(prompt: str) -> str:
            calls["n"] += 1
            return f"response:{prompt}"

        return chat

    cache = CachedChat(path, factory, namespace={"model": "qwen", "tokens": 512})
    assert cache("extract this") == "response:extract this"
    cache.flush()

    reloaded = CachedChat(path, factory, namespace={"model": "qwen", "tokens": 512})
    assert reloaded("extract this") == "response:extract this"

    with pytest.raises(ValueError, match="different run identity"):
        CachedChat(path, factory, namespace={"model": "qwen", "tokens": 768})


def test_prompt_cache_legacy_md5_reuse_must_be_explicit(tmp_path) -> None:
    path = tmp_path / "legacy_cache.json"
    key = hashlib.md5("extract this".encode("utf-8")).hexdigest()
    path.write_text(json.dumps({key: "legacy response"}), encoding="utf-8")

    with pytest.raises(ValueError, match="predates identity metadata"):
        CachedChat(path, lambda: None, namespace={"model": "qwen"})

    cache = CachedChat(
        path,
        lambda: None,
        namespace={"model": "qwen"},
        legacy_keys=True,
    )
    assert cache.unversioned_cache is True
    assert cache("extract this") == "legacy response"
