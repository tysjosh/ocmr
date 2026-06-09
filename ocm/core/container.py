"""Dependency container wiring the whole OCM system (Req 19.1, 27.2, 27.3, 11.8).

``CoreContainer`` constructs and holds every wired component the API service
(``ocm/app/api/``, task 15.2) and the agent (``ocm/agent/``, task 16.x) depend
on, so endpoints and tools stay decoupled from construction. A single container
is built per process (or per test) and FastAPI dependencies resolve from it.

What it wires (in order)
------------------------
* **Settings / IDs / logger** — :class:`~ocm.core.config.Settings`,
  :class:`~ocm.core.ids.IdGenerator` (deterministic per
  ``settings.deterministic_test_mode``), and a :class:`~ocm.core.logging.ResearchLogger`.
* **Storage** — a :class:`~ocm.memory.repository.StorageRepository` (default
  :class:`~ocm.memory.sqlite_repository.SQLiteRepository`; injectable for tests).
* **Graph** — rebuilt from the repository **on startup** via
  :func:`~ocm.memory.graph_store.rebuild_graph` so the in-memory accepted-only
  projection matches durable storage after a restart (Req 11.8).
* **Embeddings + vector index** — a swappable
  :class:`~ocm.retrieval.embeddings.EmbeddingProvider` and the Chroma-backed
  :class:`~ocm.retrieval.vector_index.VectorIndex` (graph-aware for assertion
  embedding text).
* **Governance stores** — :class:`~ocm.memory.quarantine_store.QuarantineStore`
  and :class:`~ocm.memory.provenance_tracker.ProvenanceTracker`.
* **Extractor** — :class:`~ocm.extraction.mock_extractor.MockExtractor` by
  default, or :class:`~ocm.extraction.llm_extractor.LLMExtractor` when
  ``settings.extractor == "llm"`` (Req 27.3).
* **Write pipeline (W1–W8)** and **retrieval pipeline (R0–R4)**, fully wired.

Offline-first defaults (Req 27.2)
---------------------------------
With no configuration the container selects the offline ``Mock_Extractor`` and
local embeddings, requiring no API key or network access. Both the extractor and
the embedding provider are **selectable** (Req 27.3): ``settings.extractor``
chooses mock vs LLM, and ``settings.embedding_mode`` / ``deterministic_test_mode``
choose the embedding provider.

Hermetic / deterministic operation
-----------------------------------
``LocalEmbeddingProvider`` needs the heavy ``sentence-transformers`` package,
which may be unavailable. When ``settings.deterministic_test_mode`` is set the
container uses the dependency-free
:class:`~ocm.retrieval.embeddings.DeterministicEmbeddingProvider` instead, and —
unless a repository is injected — backs storage with an in-memory SQLite
database. Combined with ``settings.chroma_mode == "memory"`` (and the vector
index's pure-Python fallback when ``chromadb`` is absent) this makes the whole
system run fully offline for tests and the research demo.

Requirements: 11.8, 19.1, 27.2, 27.3.
"""

from __future__ import annotations

from ocm.core.config import Settings
from ocm.core.ids import IdGenerator
from ocm.core.logging import ResearchLogger
from ocm.extraction.llm_extractor import LLMExtractor
from ocm.extraction.mock_extractor import MockExtractor
from ocm.memory.assertion_builder import AssertionBuilder
from ocm.memory.commit_manager import CommitManager
from ocm.memory.graph_store import GraphStore, rebuild_graph
from ocm.memory.provenance_tracker import ProvenanceTracker
from ocm.memory.quarantine_store import QuarantineStore
from ocm.memory.repository import StorageRepository
from ocm.memory.sqlite_repository import SQLiteRepository
from ocm.memory.write_pipeline import WritePipeline
from ocm.resolution.entity_resolver import EntityResolver
from ocm.resolution.normalizer import Normalizer
from ocm.retrieval.embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingProvider,
    LocalEmbeddingProvider,
)
from ocm.retrieval.evidence_packager import EvidencePackager
from ocm.retrieval.query_classifier import QueryClassifier
from ocm.retrieval.reranker import Reranker
from ocm.retrieval.retrieval_pipeline import RetrievalPipeline
from ocm.retrieval.semantic_retriever import SemanticRetriever
from ocm.retrieval.symbolic_retriever import SymbolicRetriever
from ocm.retrieval.vector_index import VectorIndex
from ocm.validation.constraints import ConstraintValidator
from ocm.validation.schema_validator import SchemaValidator


class CoreContainer:
    """Constructs and holds the wired OCM components (Req 19.1)."""

    def __init__(
        self,
        settings: Settings,
        repo: StorageRepository | None = None,
        extractor: object | None = None,
        embeddings: "EmbeddingProvider | None" = None,
    ) -> None:
        """Build the full component graph from ``settings``.

        Args:
            settings: The OCM :class:`Settings`. Offline-first defaults select
                the mock extractor and local embeddings (Req 27.2); both are
                selectable via configuration (Req 27.3).
            repo: Optional :class:`StorageRepository` to inject (tests pass an
                in-memory ``SQLiteRepository(":memory:")``). When omitted a
                default repository is constructed: an in-memory SQLite database
                in ``deterministic_test_mode`` (hermetic), otherwise
                ``SQLiteRepository(settings.sqlite_path)``.
            extractor: Optional pre-built W1 extractor to inject (e.g. an
                :class:`LLMExtractor` wired with a fake HTTP client for offline
                tests). When omitted the extractor is selected from
                ``settings.extractor`` (Req 27.3).
            embeddings: Optional pre-built :class:`EmbeddingProvider` to inject.
                Lets an expensive provider (e.g. a real sentence-transformers
                model) be **loaded once and shared** across many containers
                (the multi-seed/ablation experiment harness). When omitted the
                provider is selected from ``settings`` (Req 27.3).
        """
        self.settings = settings

        # --- core services ------------------------------------------------
        self.ids = IdGenerator(deterministic=settings.deterministic_test_mode)
        self.research_logger = ResearchLogger()

        # --- storage ------------------------------------------------------
        self.repo: StorageRepository = repo or self._build_default_repo(settings)

        # --- graph: rebuild from durable storage on startup (Req 11.8) ----
        self.graph: GraphStore = rebuild_graph(self.repo)

        # --- embeddings + vector index (provider injectable, Req 27.3) ----
        self.embeddings: EmbeddingProvider = (
            embeddings if embeddings is not None else self._build_embedding_provider(settings)
        )
        self.vector_index = VectorIndex(
            self.embeddings,
            chroma_mode=settings.chroma_mode,
            chroma_path=settings.chroma_path,
            graph=self.graph,
        )

        # --- governance stores --------------------------------------------
        self.quarantine_store = QuarantineStore(self.repo, self.ids)
        self.provenance_tracker = ProvenanceTracker(self.repo, self.ids)

        # --- extractor (selectable, Req 27.3; injectable for tests) -------
        self.extractor = extractor if extractor is not None else self._build_extractor(settings)

        # --- write-pipeline stages (W2–W8) --------------------------------
        self.normalizer = Normalizer()
        self.resolver = EntityResolver()
        self.assertion_builder = AssertionBuilder()
        self.schema_validator = SchemaValidator()
        # ConstraintValidator binds the real Contradiction_Checker (W7) by
        # default using these settings, so the contradiction gate (C7) runs.
        self.constraint_validator = ConstraintValidator(settings)
        self.commit_manager = CommitManager(
            repo=self.repo,
            graph=self.graph,
            ids=self.ids,
            quarantine_store=self.quarantine_store,
            provenance_tracker=self.provenance_tracker,
            # Embed accepted assertions into the Vector_Index (Req 13.5).
            embed_hook=self.vector_index.embed_assertion,
            # Re-tag superseded assertions in the Vector_Index (Req 10.5, 16.2).
            status_hook=self.vector_index.set_status,
        )

        # --- write pipeline (W1–W8) ---------------------------------------
        self.write_pipeline = WritePipeline(
            extractor=self.extractor,
            normalizer=self.normalizer,
            resolver=self.resolver,
            assertion_builder=self.assertion_builder,
            schema_validator=self.schema_validator,
            constraint_validator=self.constraint_validator,
            commit_manager=self.commit_manager,
            repo=self.repo,
            graph=self.graph,
            ids=self.ids,
            provenance_tracker=self.provenance_tracker,
            quarantine_store=self.quarantine_store,
            # Embed accepted claims / documents / events (Req 16.6).
            memory_embed_hook=self.vector_index.embed_memory,
            research_logger=self.research_logger,
            settings=settings,
        )

        # --- retrieval-pipeline stages (R0–R4) ----------------------------
        self.query_classifier = QueryClassifier()
        self.symbolic_retriever = SymbolicRetriever()
        self.semantic_retriever = SemanticRetriever(self.vector_index)
        self.reranker = Reranker(settings.rerank_weights)
        self.evidence_packager = EvidencePackager()

        # --- retrieval pipeline (R0–R4) -----------------------------------
        self.retrieval_pipeline = RetrievalPipeline(
            classifier=self.query_classifier,
            symbolic_retriever=self.symbolic_retriever,
            semantic_retriever=self.semantic_retriever,
            reranker=self.reranker,
            evidence_packager=self.evidence_packager,
            graph=self.graph,
            provenance_tracker=self.provenance_tracker,
            quarantine_store=self.quarantine_store,
            research_logger=self.research_logger,
            settings=settings,
            ids=self.ids,
        )

    # ------------------------------------------------------------------ #
    # Component selection helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_default_repo(settings: Settings) -> StorageRepository:
        """Build the default repository when none is injected.

        Uses an in-memory SQLite database in ``deterministic_test_mode`` so
        tests stay hermetic and never touch disk; otherwise persists to
        ``settings.sqlite_path``.
        """
        if settings.deterministic_test_mode:
            return SQLiteRepository(":memory:")
        return SQLiteRepository(settings.sqlite_path)

    @staticmethod
    def _build_embedding_provider(settings: Settings) -> EmbeddingProvider:
        """Select the embedding provider (Req 27.3, offline-first Req 27.2).

        In ``deterministic_test_mode`` the dependency-free, offline
        :class:`DeterministicEmbeddingProvider` is used so the system runs
        hermetically without ``sentence-transformers``. Otherwise the default
        local provider (``settings.embedding_model``) is used.
        """
        if settings.deterministic_test_mode:
            return DeterministicEmbeddingProvider()
        return LocalEmbeddingProvider(settings.embedding_model)

    @staticmethod
    def _build_extractor(settings: Settings):
        """Select the W1 extractor (Req 27.3; offline default Req 27.2)."""
        if settings.extractor == "llm":
            return LLMExtractor(settings)
        return MockExtractor()
