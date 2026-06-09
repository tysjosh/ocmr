"""Retrieval Pipeline — orchestrates R0→R4 (Req 18.x, 25.2).

The :class:`RetrievalPipeline` runs the read path end to end:

    R0 classify → R1 symbolic → R2 semantic → R3 rerank → R4 package

and returns an :class:`~ocm.retrieval.evidence_packager.EvidencePackage`. It is
the object behind ``POST /memory/query`` (task 15.2) and the agent's
``MemoryTool.query`` (task 16.x). The pipeline performs no ranking or packaging
logic itself — it wires the five stages together and records one structured
research-log record per query (Req 25.2).

Per-query research log (Req 25.2)
---------------------------------
On every :meth:`query`, when a :class:`~ocm.core.logging.ResearchLogger` is
configured, the pipeline emits a ``query`` record with ``query_id``,
``query_type`` (from R0), ``symbolic_results_count`` (R1), ``semantic_results_count``
(R2), ``top_k_ids`` (the highest-scored ``memory_id``s from R3),
``conflicts_returned`` (from the R4 package), and ``latency_ms``.

Contradiction signal
---------------------
When a :class:`~ocm.memory.quarantine_store.QuarantineStore` is wired, the ids
that unresolved quarantine records conflict with are passed to the Reranker as
``contradicted_ids`` so accepted items in an open conflict are penalized
(Req 17.3), and quarantined items surface as conflicts in the package (Req 18.4).

Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 25.2.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

from ocm.retrieval.evidence_packager import EvidencePackage, EvidencePackager


class RetrievalPipeline:
    """Wires R0→R4 and returns an :class:`EvidencePackage` per query."""

    def __init__(
        self,
        classifier: Any,
        symbolic_retriever: Any,
        semantic_retriever: Any,
        reranker: Any,
        evidence_packager: EvidencePackager,
        graph: Any,
        provenance_tracker: Any,
        quarantine_store: Any | None = None,
        research_logger: Any | None = None,
        settings: Any | None = None,
        ids: Any | None = None,
    ) -> None:
        """Wire the retrieval stages and their backing stores.

        Args:
            classifier: R0 ``QueryClassifier`` (``classify(query)``).
            symbolic_retriever: R1 ``SymbolicRetriever`` (``retrieve(cls, graph)``).
            semantic_retriever: R2 ``SemanticRetriever``
                (``retrieve(query, cls, top_k, include_conflicts)``).
            reranker: R3 ``Reranker`` (``rerank(symbolic, semantic, weights, ...)``).
            evidence_packager: R4 ``EvidencePackager`` (``package(...)``).
            graph: The ``Graph_Store`` R1 reads and R4 resolves names from.
            provenance_tracker: The ``Provenance_Tracker`` R4 reads sources from.
            quarantine_store: Optional ``Quarantine_Store`` used to derive
                contradiction signals and augment conflicts.
            research_logger: Optional ``ResearchLogger`` for per-query logs (Req 25.2).
            settings: Optional ``Settings`` supplying ``rerank_weights``.
            ids: Optional ``IdGenerator`` for deterministic ``query_id``s.
        """
        self.classifier = classifier
        self.symbolic_retriever = symbolic_retriever
        self.semantic_retriever = semantic_retriever
        self.reranker = reranker
        self.evidence_packager = evidence_packager
        self.graph = graph
        self.provenance_tracker = provenance_tracker
        self.quarantine_store = quarantine_store
        self.research_logger = research_logger
        self.settings = settings
        self.ids = ids

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        include_conflicts: bool = False,
    ) -> EvidencePackage:
        """Run R0→R4 for ``query_text`` and return the evidence package.

        Args:
            query_text: The natural-language query.
            top_k: Number of nearest semantic items to retrieve (R2) and the
                size of the ``top_k_ids`` slice recorded in the research log.
            include_conflicts: Force inclusion of quarantined items in R2 even
                for a non-conflict query.

        Returns:
            The assembled :class:`EvidencePackage` (Req 18.1).
        """
        start = time.perf_counter()

        # R0 — classify.
        classification = self.classifier.classify(query_text)

        # R1 — symbolic retrieval over the accepted-only graph.
        symbolic = self.symbolic_retriever.retrieve(classification, self.graph)

        # R2 — semantic retrieval (status/conflict-aware).
        semantic = self.semantic_retriever.retrieve(
            query_text,
            classification,
            top_k=top_k,
            include_conflicts=include_conflicts,
        )

        # Contradiction signal: ids that unresolved quarantine records conflict
        # with (so accepted items in an open conflict are penalized, Req 17.3).
        contradicted_ids = self._contradicted_ids()

        # R3 — rerank into a single, ordered candidate set.
        weights = getattr(self.settings, "rerank_weights", None) if self.settings else None
        ranked = self.reranker.rerank(
            symbolic,
            semantic,
            weights=weights,
            contradicted_ids=contradicted_ids,
        )

        # R4 — package into an EvidencePackage.
        package = self.evidence_packager.package(
            query_text,
            classification,
            ranked,
            graph=self.graph,
            provenance_tracker=self.provenance_tracker,
            quarantine_store=self.quarantine_store,
        )

        latency_ms = (time.perf_counter() - start) * 1000.0

        # Per-query research log (Req 25.2).
        self._log_query(
            query_text=query_text,
            classification=classification,
            symbolic_count=len(symbolic),
            semantic_count=len(semantic),
            ranked=ranked,
            top_k=top_k,
            conflicts_returned=len(package.conflicts),
            latency_ms=latency_ms,
        )

        return package

    # Alias matching the design's ``RetrievalPipeline.run`` terminology.
    run = query

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _contradicted_ids(self) -> set[str]:
        """Collect ids that unresolved quarantine records conflict with."""
        if self.quarantine_store is None:
            return set()
        contradicted: set[str] = set()
        try:
            records = self.quarantine_store.list("unresolved")
        except Exception:  # pragma: no cover - defensive
            records = []
        for record in records or []:
            contradicted.update(getattr(record, "conflicting_ids", []) or [])
        return contradicted

    def _make_query_id(self, query_text: str) -> str:
        """Build a query id, deterministic when an ``IdGenerator`` is wired."""
        if self.ids is not None:
            try:
                return self.ids.generic_id("qry", query_text)
            except Exception:  # pragma: no cover - defensive
                pass
        digest = hashlib.sha1(query_text.encode("utf-8")).hexdigest()[:12]
        return f"qry_{digest}"

    def _log_query(
        self,
        *,
        query_text: str,
        classification: Any,
        symbolic_count: int,
        semantic_count: int,
        ranked: list[Any],
        top_k: int,
        conflicts_returned: int,
        latency_ms: float,
    ) -> None:
        """Emit the per-query research-log record (Req 25.2)."""
        if self.research_logger is None:
            return
        top_k_ids = [item.memory_id for item in ranked[: max(0, top_k)]]
        self.research_logger.log_query(
            query_id=self._make_query_id(query_text),
            query_type=getattr(classification, "query_type", "open_ended"),
            symbolic_results_count=symbolic_count,
            semantic_results_count=semantic_count,
            top_k_ids=top_k_ids,
            conflicts_returned=conflicts_returned,
            latency_ms=latency_ms,
        )
