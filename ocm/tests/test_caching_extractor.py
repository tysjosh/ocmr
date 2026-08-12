"""Tests for the memoizing CachingExtractor evaluation accelerator."""

from __future__ import annotations

from ocm.extraction.caching_extractor import CachingExtractor
from ocm.memory.contracts import ExtractionResult


class _CountingExtractor:
    """Base extractor that counts real calls and tags output per call."""

    version = "counting-1"

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, text: str, source_ref: str) -> ExtractionResult:
        self.calls += 1
        return ExtractionResult(
            entities=[{"type": "Person", "name": f"P{self.calls}", "fields": {}}],
            events=[], claims=[], documents=[], decisions=[], relations=[],
            extractor_version="counting-1",
        )


def test_repeated_inputs_hit_cache_and_skip_base():
    base = _CountingExtractor()
    ext = CachingExtractor(base)

    r1 = ext.extract("Alice owns Orion.", "ex1:s1")
    r2 = ext.extract("Alice owns Orion.", "ex1:s1")  # identical -> cache hit
    r3 = ext.extract("Bob owns Atlas.", "ex2:s1")    # new -> miss

    assert base.calls == 2  # only the two distinct inputs hit the base model
    assert ext.stats == {
        "hits": 1,
        "misses": 2,
        "size": 2,
        "failure_hits": 0,
        "distinct_failures": 0,
    }
    # Cached result equals the first (deterministic), and is a distinct object.
    assert r1.entities == r2.entities
    assert r1 is not r2
    assert r3.entities != r1.entities


def test_source_ref_is_part_of_the_key():
    base = _CountingExtractor()
    ext = CachingExtractor(base)
    ext.extract("same text", "refA")
    ext.extract("same text", "refB")  # different source_ref -> distinct
    assert base.calls == 2


def test_persistence_round_trip(tmp_path):
    path = str(tmp_path / "extract_cache.json")
    base = _CountingExtractor()
    ext = CachingExtractor(base, cache_path=path, autosave_every=1)
    ext.extract("persist me", "ex:s1")
    assert ext.misses == 1

    # A fresh wrapper over a fresh base loads the cache and serves from it.
    base2 = _CountingExtractor()
    ext2 = CachingExtractor(base2, cache_path=path)
    ext2.extract("persist me", "ex:s1")
    assert base2.calls == 0  # served from the persisted cache
    assert ext2.hits == 1
