"""An extraction cache must not be reused across extractors.

The cache key is ``sha256(source_ref || text)`` and contains nothing about the
model. ``TransformersExtractor.version`` is the constant ``"qwen-transformers-v1"``
regardless of which Qwen is loaded. So without a provenance check, a cache built
with a 14B model is served silently to a run configured for a 32B one, and no
downstream metric reveals it. That matters most once a cache is published for
others to reuse.
"""

from __future__ import annotations

import json

import pytest

from ocm.extraction.caching_extractor import CachingExtractor
from ocm.memory.contracts import ExtractionResult


class _Stub:
    """An extractor with a declared fingerprint."""

    def __init__(self, model: str, version: str = "qwen-transformers-v1") -> None:
        self.version = version
        self.fingerprint = {
            "version": version,
            "model": model,
            "max_new_tokens": 1024,
            "prompt_sha256": "abc123",
        }
        self.calls = 0

    def extract(self, text: str, source_ref: str) -> ExtractionResult:
        self.calls += 1
        return ExtractionResult(
            relations=[{"subject": "A", "predicate": "OWNS", "object": text}],
            extractor_version=self.version,
        )


def _write_cache(tmp_path, base) -> str:
    path = str(tmp_path / "cache.json")
    cache = CachingExtractor(base, cache_path=path)
    cache.extract("Alice owns Project Orion.", "s1")
    cache.save()
    return path


def test_cache_round_trips_for_the_same_extractor(tmp_path) -> None:
    base = _Stub("Qwen/Qwen2.5-14B-Instruct")
    path = _write_cache(tmp_path, base)

    reloaded = CachingExtractor(_Stub("Qwen/Qwen2.5-14B-Instruct"), cache_path=path)
    assert reloaded.stats["size"] == 1
    assert reloaded.unversioned_cache is False


def test_cache_from_a_different_model_is_refused(tmp_path) -> None:
    """The regression: silently serving 14B generations to a 32B run."""
    path = _write_cache(tmp_path, _Stub("Qwen/Qwen2.5-14B-Instruct"))

    with pytest.raises(ValueError) as excinfo:
        CachingExtractor(_Stub("Qwen/Qwen2.5-32B-Instruct"), cache_path=path)
    message = str(excinfo.value)
    assert "14B" in message and "32B" in message


def test_cache_from_a_different_prompt_is_refused(tmp_path) -> None:
    path = _write_cache(tmp_path, _Stub("Qwen/Qwen2.5-14B-Instruct"))
    other = _Stub("Qwen/Qwen2.5-14B-Instruct")
    other.fingerprint["prompt_sha256"] = "deadbeef"

    with pytest.raises(ValueError):
        CachingExtractor(other, cache_path=path)


def test_legacy_flat_cache_still_loads_but_is_flagged(tmp_path) -> None:
    """A cache is expensive to rebuild, so pre-fingerprint files stay usable."""
    path = tmp_path / "flat.json"
    payload = {
        "somekey": ExtractionResult(extractor_version="v1").model_dump(mode="json")
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    cache = CachingExtractor(_Stub("Qwen/Qwen2.5-14B-Instruct"), cache_path=str(path))
    assert cache.stats["size"] == 1
    assert cache.unversioned_cache is True


def test_resaving_a_legacy_cache_stamps_it(tmp_path) -> None:
    path = tmp_path / "flat.json"
    path.write_text(
        json.dumps({"k": ExtractionResult(extractor_version="v1").model_dump(mode="json")}),
        encoding="utf-8",
    )
    base = _Stub("Qwen/Qwen2.5-14B-Instruct")
    cache = CachingExtractor(base, cache_path=str(path))
    cache.save()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["__meta__"]["fingerprint"] == base.fingerprint
    assert payload["__meta__"]["format"] == 2

    # And it now round-trips under validation.
    assert CachingExtractor(base, cache_path=str(path)).unversioned_cache is False


def test_extractor_without_a_fingerprint_is_not_blocked(tmp_path) -> None:
    """Providers that declare nothing keep working, just without verification."""

    class _NoFingerprint:
        version = "mock-v1"

        def extract(self, text: str, source_ref: str) -> ExtractionResult:
            return ExtractionResult(extractor_version="mock-v1")

    path = str(tmp_path / "c.json")
    cache = CachingExtractor(_NoFingerprint(), cache_path=path)
    cache.extract("x", "s1")
    cache.save()
    assert CachingExtractor(_NoFingerprint(), cache_path=path).stats["size"] == 1


def test_transformers_extractor_fingerprint_includes_the_model() -> None:
    """The real extractor must expose an identity that separates model sizes."""
    from types import SimpleNamespace

    from ocm.extraction.transformers_extractor import TransformersExtractor

    def build(model_id: str) -> dict:
        extractor = TransformersExtractor(
            model=SimpleNamespace(config=SimpleNamespace(_name_or_path=model_id)),
            tokenizer=SimpleNamespace(),
            complete=lambda messages: "{}",
        )
        return extractor.fingerprint

    small, large = build("Qwen/Qwen2.5-14B-Instruct"), build("Qwen/Qwen2.5-32B-Instruct")
    assert small != large
    assert small["model"] == "Qwen/Qwen2.5-14B-Instruct"
    assert small["prompt_sha256"] == large["prompt_sha256"]  # same prompt


def test_strict_extractor_forwards_the_fingerprint() -> None:
    """The cache sits outside StrictExtractor, so identity must pass through."""
    from ocm.extraction.strict_extractor import StrictExtractor

    base = _Stub("Qwen/Qwen2.5-14B-Instruct")
    assert StrictExtractor(base).fingerprint == base.fingerprint
