"""Configuration model for the Ontology-Constrained Memory (OCM) system.

`Settings` is a Pydantic v2 model that controls extractor/embedding selection,
storage paths, determinism, and the reranker/governance thresholds. With **no
configuration supplied** the defaults select the offline ``Mock_Extractor`` and
the local ``sentence-transformers/all-MiniLM-L6-v2`` embedding model so the
whole system runs fully offline (Req 27.2). Both the extractor and embedding
implementations are selectable via configuration (Req 27.3).

See the design's "Configuration Model" section for the canonical field set.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class RerankWeights(BaseModel):
    """Weights for the retrieval reranker score function.

    ``score = alpha*semantic_similarity + beta*graph_relevance
              + gamma*confidence + delta*provenance_quality
              + eta*recency - lambda_*contradiction_penalty``

    Defaults come directly from the design (alpha 0.40, beta 0.25, gamma 0.15,
    delta 0.10, eta 0.05, lambda 0.30). ``lambda`` is a Python keyword, so the
    field is named ``lambda_`` with the alias ``"lambda"`` for serialization.
    """

    alpha: float = 0.40  # semantic_similarity
    beta: float = 0.25  # graph_relevance
    gamma: float = 0.15  # confidence
    delta: float = 0.10  # provenance_quality
    eta: float = 0.05  # recency
    lambda_: float = Field(default=0.30, alias="lambda")  # contradiction_penalty

    model_config = {"populate_by_name": True}


class Settings(BaseModel):
    """Top-level OCM configuration (environment + optional file override).

    Offline-first defaults (Req 27.2): ``extractor="mock"`` and local
    embeddings, requiring no API key or network access.
    """

    # --- Extraction selection (Req 27.3) -----------------------------------
    extractor: Literal["mock", "llm"] = "mock"
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    # Send ``response_format={"type":"json_object"}`` to the LLM endpoint. Keep
    # on for servers that support JSON mode (OpenAI, vLLM, recent Ollama); turn
    # off for local servers that reject the field — the prompt still requests
    # JSON-only output.
    llm_use_json_mode: bool = True

    # --- Embedding selection (Req 27.3, 13.x) ------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_mode: Literal["local"] = "local"

    # --- Storage paths ------------------------------------------------------
    sqlite_path: str = "ocm.db"
    chroma_mode: Literal["persistent", "memory"] = "persistent"
    chroma_path: str = ".chroma"

    # --- Determinism (Req 27.5) --------------------------------------------
    deterministic_test_mode: bool = False

    # --- Retrieval / governance knobs --------------------------------------
    rerank_weights: RerankWeights = Field(default_factory=RerankWeights)
    contradiction_high_confidence: float = 0.8
    decision_evidence_floor: int = 1

    # --- Supersession admissibility (Algorithm 1, line 7) ------------------
    # A ``correction`` may supersede an accepted high-confidence assertion only
    # when it is more confident than the incumbent by a margin ``delta`` and
    # carries at least ``e_min`` units of supporting evidence; otherwise the
    # conflict is quarantined. ``supersede_margin`` is delta in
    # ``c(a) - c(a_old) > delta`` (default 0.1, per the paper's Algorithm 1: a
    # correction must beat the incumbent by 0.1 to supersede, else quarantine).
    # ``supersede_evidence_min`` is e_min, operationalized as an integer count of
    # evidence units (a present ``source_ref`` plus accepted ``EVIDENCE_FOR``
    # edges into the subject), not a [0,1] completeness fraction.
    supersede_margin: float = 0.1
    supersede_evidence_min: int = 1

    # --- Write-time governance ablation switches (paper §IV-D) -------------
    # Enabled by default (full OCMR). Disabling one realizes a write-time
    # ablation: ``enable_schema_validation`` gates W5 typed-schema checks;
    # ``enable_contradiction_gate`` gates W7/C7 so contradictions are no longer
    # blocked/quarantined at write time. Retrieval-time ablations (provenance,
    # hybrid routing) are expressed via the baseline StrategyToggles.
    enable_schema_validation: bool = True
    enable_contradiction_gate: bool = True
