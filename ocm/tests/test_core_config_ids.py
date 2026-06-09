"""Unit tests for OCM configuration defaults and deterministic ID generation.

These cover task 1.4:

* ``Settings`` offline-first defaults: the Mock_Extractor and local
  ``all-MiniLM-L6-v2`` embeddings run with no API key or network (Req 27.2),
  and both extractor and embedding implementations are config-selectable
  (Req 27.3).
* ``RerankWeights`` defaults match the design's reranker score function.
* ``IdGenerator`` determinism (Req 27.5): with ``deterministic=True`` two
  fresh generators reproduce the same entity_id sequence for the same inputs,
  while differing ``type``/``normalized_name``/``source_ref`` yield distinct
  IDs; with ``deterministic=False`` IDs are unique across calls.
"""

from __future__ import annotations

from ocm.core.config import RerankWeights, Settings
from ocm.core.ids import IdGenerator


# ---------------------------------------------------------------------------
# Settings defaults (Req 27.2, 27.3)
# ---------------------------------------------------------------------------
def test_settings_offline_defaults() -> None:
    """No config supplied -> fully offline Mock_Extractor + local embeddings."""
    settings = Settings()

    # Offline extractor by default (Req 27.2).
    assert settings.extractor == "mock"

    # Local MiniLM embedding model, run locally (Req 27.2, 27.3).
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedding_mode == "local"

    # Determinism is opt-in; production default is random IDs (Req 27.5).
    assert settings.deterministic_test_mode is False

    # Offline implies no API key / base URL required by default.
    assert settings.llm_api_key is None
    assert settings.llm_base_url is None


def test_settings_extractor_and_embedding_are_selectable() -> None:
    """Both extractor and embedding implementations are config-selectable (Req 27.3)."""
    settings = Settings(extractor="llm")
    assert settings.extractor == "llm"

    custom = Settings(embedding_model="custom/model")
    assert custom.embedding_model == "custom/model"


# ---------------------------------------------------------------------------
# RerankWeights defaults
# ---------------------------------------------------------------------------
def test_rerank_weights_defaults() -> None:
    """Default reranker weights match the design's score function."""
    weights = RerankWeights()

    assert weights.alpha == 0.40
    assert weights.beta == 0.25
    assert weights.gamma == 0.15
    assert weights.delta == 0.10
    assert weights.eta == 0.05
    assert weights.lambda_ == 0.30


def test_rerank_weights_lambda_alias() -> None:
    """``lambda_`` is exposed under the ``"lambda"`` alias for serialization."""
    weights = RerankWeights()
    assert weights.model_dump(by_alias=True)["lambda"] == 0.30


def test_settings_uses_default_rerank_weights() -> None:
    """A fresh ``Settings`` carries the default reranker weights."""
    settings = Settings()
    assert settings.rerank_weights == RerankWeights()


# ---------------------------------------------------------------------------
# Deterministic ID generation (Req 27.5)
# ---------------------------------------------------------------------------
ENTITY_INPUTS = [
    ("Person", "ada lovelace", "doc-1#0"),
    ("Person", "alan turing", "doc-1#1"),
    ("Concept", "analytical engine", "doc-2#0"),
]


def test_deterministic_entity_id_sequences_are_reproducible() -> None:
    """Two fresh deterministic generators yield identical entity_id sequences."""
    gen_a = IdGenerator(deterministic=True)
    gen_b = IdGenerator(deterministic=True)

    ids_a = [gen_a.entity_id(t, n, s) for t, n, s in ENTITY_INPUTS]
    ids_b = [gen_b.entity_id(t, n, s) for t, n, s in ENTITY_INPUTS]

    assert ids_a == ids_b


def test_deterministic_entity_id_uses_type_prefix() -> None:
    """Deterministic IDs carry a stable lowercase 3-char type prefix."""
    gen = IdGenerator(deterministic=True)
    eid = gen.entity_id("Person", "ada lovelace", "doc-1#0")
    assert eid.startswith("per_")


def test_deterministic_entity_id_varies_with_type() -> None:
    """Different entity type at the same counter position yields a different ID."""
    gen_a = IdGenerator(deterministic=True)
    gen_b = IdGenerator(deterministic=True)

    id_type_a = gen_a.entity_id("Person", "ada lovelace", "doc-1#0")
    id_type_b = gen_b.entity_id("Concept", "ada lovelace", "doc-1#0")

    assert id_type_a != id_type_b


def test_deterministic_entity_id_varies_with_normalized_name() -> None:
    """Different normalized_name at the same counter position yields a different ID."""
    gen_a = IdGenerator(deterministic=True)
    gen_b = IdGenerator(deterministic=True)

    id_name_a = gen_a.entity_id("Person", "ada lovelace", "doc-1#0")
    id_name_b = gen_b.entity_id("Person", "alan turing", "doc-1#0")

    assert id_name_a != id_name_b


def test_deterministic_entity_id_varies_with_source_ref() -> None:
    """Different source_ref at the same counter position yields a different ID."""
    gen_a = IdGenerator(deterministic=True)
    gen_b = IdGenerator(deterministic=True)

    id_src_a = gen_a.entity_id("Person", "ada lovelace", "doc-1#0")
    id_src_b = gen_b.entity_id("Person", "ada lovelace", "doc-2#0")

    assert id_src_a != id_src_b


def test_deterministic_counter_breaks_ties_for_identical_input() -> None:
    """Identical inputs within one run still get distinct IDs via the counter."""
    gen = IdGenerator(deterministic=True)
    first = gen.entity_id("Person", "ada lovelace", "doc-1#0")
    second = gen.entity_id("Person", "ada lovelace", "doc-1#0")
    assert first != second


def test_random_entity_ids_are_unique() -> None:
    """In random mode, repeated calls with identical input produce unique IDs."""
    gen = IdGenerator(deterministic=False)
    ids = [gen.entity_id("Person", "ada lovelace", "doc-1#0") for _ in range(100)]
    assert len(set(ids)) == len(ids)


def test_random_entity_ids_carry_type_prefix() -> None:
    """Random IDs still carry the stable type prefix."""
    gen = IdGenerator(deterministic=False)
    assert gen.entity_id("Person", "ada lovelace", "doc-1#0").startswith("per_")
