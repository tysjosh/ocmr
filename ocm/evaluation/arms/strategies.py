"""Baseline strategy abstraction for the evaluation harness (Req 22).

This module implements the design's "Baselines as Configurable Strategy Objects"
contract: every baseline (B0–B4) is the **same** :class:`MemoryStrategy`
differing only by a set of feature :class:`StrategyToggles`, so each baseline is
a clean *ablation* of the full OCM system rather than a separate codebase.

Design mapping (the toggle matrix, Req 22.1–22.5)
-------------------------------------------------
``StrategyToggles`` carries the seven switches from the design table:

* ``use_ontology``      — schema + relation registry + constraints
* ``use_graph``         — symbolic retrieval over the NetworkX Graph_Store (R1)
* ``use_vectors``       — semantic retrieval over the Chroma Vector_Index (R2)
* ``use_contradiction`` — the W7 contradiction gate / contradiction penalty (C7, R3)
* ``use_quarantine``    — surface quarantined items as conflicts vs. treat them
  as ordinary accepted results ("accept-anyway")
* ``use_provenance``    — provenance recording + provenance-aware reranking and
  ``supporting_sources`` in the evidence package
* ``use_answer_policy`` — render the package through the P1–P5 Answer_Policy

How the ablation is realised (pragmatic stance)
-----------------------------------------------
All baselines **write** through the same wired, governed
:class:`~ocm.memory.write_pipeline.WritePipeline` on the
:class:`~ocm.core.container.CoreContainer` (the graph, assertion store, and
vector index are all populated). The ablation is then applied at **retrieval
composition** time by the strategy's own ``query`` orchestration, which reuses
the container's already-wired R0–R4 components but selects *which* of them feed
the result and *whether* governance signals (contradictions, quarantine,
provenance) are honoured:

* ``use_graph`` / ``use_vectors`` select whether the Symbolic Retriever (R1)
  and/or the Semantic Retriever (R2) feed the Reranker (R3) — this is how B0
  becomes "vectors only" and B1 becomes "graph/symbolic only".
* ``use_contradiction`` decides whether contradiction penalties are applied in
  R3 (governed) or suppressed (accept-anyway).
* ``use_quarantine`` decides whether quarantined items surface as **conflicts**
  in the evidence package (governed, B3/B4) or are folded in as ordinary
  results (ungoverned, B0–B2).
* ``use_provenance`` decides whether the Provenance_Tracker populates
  ``supporting_sources`` and whether provenance quality contributes to ranking.
* ``use_answer_policy`` decides whether the package is rendered through the
  Answer_Policy (B4) — its rendered text is placed on ``package.answer``.

This keeps every baseline a true ablation of one shared implementation while
remaining fully testable over a deterministic in-memory container.

Requirements: 22.1, 22.2, 22.3, 22.4, 22.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ocm.core.container import CoreContainer
from ocm.memory.write_pipeline import WriteResult
from ocm.ontology.enums import WriteIntent
from ocm.retrieval.evidence_packager import EvidencePackage

# The Answer_Policy is built by a parallel task (16.2). Import it defensively so
# this module — and the B4 baseline — load even before that task lands. When it
# is unavailable B4 falls back to a small built-in renderer (see ``_render``).
try:  # pragma: no cover - exercised once task 16.2 lands.
    from ocm.agent.answer_policy import AnswerPolicy as _AnswerPolicy
except Exception:  # pragma: no cover - parallel task may not exist yet.
    _AnswerPolicy = None  # type: ignore[assignment]


@dataclass(frozen=True)
class StrategyToggles:
    """Feature switches that turn the shared system into a specific baseline.

    The fields mirror the design's B0–B4 toggle matrix (Req 22). All default to
    the fully-governed configuration (everything on except the answer policy),
    so a baseline is expressed by turning features *off*.
    """

    use_ontology: bool = True
    use_graph: bool = True
    use_vectors: bool = True
    use_contradiction: bool = True
    use_quarantine: bool = True
    use_provenance: bool = True
    use_answer_policy: bool = False
    #: When False, the evidence package is built without the Graph_Store, so no
    #: graph-assisted (structural) answer is derived and the result is read only
    #: from retrieved text — modelling a vanilla RAG system (the RAG-only
    #: baseline). Defaults True so B0–B4 keep their structural answer derivation.
    use_structured_answer: bool = True
    #: When True, contradictions are detected **at retrieval time** by scanning
    #: accepted memory for single-valued-relation conflicts (rather than gated at
    #: write time). Models the "filter at read time" alternative to write-time
    #: governance. Independent of ``use_contradiction`` (which reads write-time
    #: quarantine records); used by the retrieval-time-contradiction-filter
    #: baseline, which leaves the write gate off.
    use_read_time_filter: bool = False

    # -- convenience aliases -------------------------------------------- #
    @property
    def use_symbolic(self) -> bool:
        """Alias for :attr:`use_graph` (symbolic retrieval, R1)."""
        return self.use_graph

    @property
    def use_semantic(self) -> bool:
        """Alias for :attr:`use_vectors` (semantic retrieval, R2)."""
        return self.use_vectors

    @property
    def use_contradiction_gate(self) -> bool:
        """Alias for :attr:`use_contradiction` (W7 gate / C7)."""
        return self.use_contradiction


class MemoryStrategy:
    """A baseline as a configurable strategy over a wired ``CoreContainer``.

    Exposes the uniform interface the Baseline_Runner (task 17.3) drives:

    * :meth:`write` — ingest text into governed memory.
    * :meth:`query` — retrieve an :class:`EvidencePackage`, composed and
      governed according to the strategy's :class:`StrategyToggles`.

    The strategy never reconstructs the pipeline; it reuses the container's
    already-wired R0–R4 components and applies the ablation by selecting which
    feed the result and which governance signals are honoured.
    """

    def __init__(
        self,
        name: str,
        container: CoreContainer,
        toggles: StrategyToggles,
        *,
        high_stakes: bool = False,
    ) -> None:
        """Create a strategy.

        Args:
            name: Baseline name (e.g. ``"B3"``) used in research logs/metrics.
            container: The wired :class:`CoreContainer` backing this strategy.
            toggles: The feature switches defining this baseline.
            high_stakes: Whether the Answer_Policy should treat answers as
                high-stakes / decision-support (forces provenance inclusion,
                P4) when ``use_answer_policy`` is set.
        """
        self.name = name
        self.container = container
        self.toggles = toggles
        self.high_stakes = high_stakes
        # Build the Answer_Policy lazily/defensively for B4.
        self._answer_policy = self._build_answer_policy() if toggles.use_answer_policy else None

    # ================================================================== #
    # Write
    # ================================================================== #
    def write(
        self,
        text: str,
        source_ref: str,
        write_intent: str | WriteIntent | None = None,
    ) -> WriteResult:
        """Ingest ``text`` into governed memory via the shared Write_Pipeline.

        All baselines write through the same governed pipeline so the graph and
        vector index are populated identically; the baselines differ only in how
        :meth:`query` composes and governs retrieval. Returns the pipeline's
        :class:`WriteResult` (accepted / superseded / quarantined / rejected
        outcome lists plus the rolled-up summary).
        """
        return self.container.write_pipeline.run(
            text, source_ref, write_intent=write_intent
        )

    # ================================================================== #
    # Query
    # ================================================================== #
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        include_conflicts: bool = False,
    ) -> EvidencePackage:
        """Retrieve an :class:`EvidencePackage` under this baseline's toggles.

        The flow reuses the container's wired stages (R0 classify → R1 symbolic
        → R2 semantic → R3 rerank → R4 package), but:

        * R1 hits are dropped when ``use_graph`` is off (vectors-only baselines).
        * R2 hits are dropped when ``use_vectors`` is off (graph-only baselines).
        * Contradiction penalties (R3) are applied only when ``use_contradiction``
          is on; otherwise contradicted ids are not signalled.
        * Quarantined items surface as conflicts only when ``use_quarantine`` is
          on; an ungoverned baseline instead folds them in as ordinary results.
        * Provenance populates ``supporting_sources`` only when ``use_provenance``
          is on.
        * For B4 (``use_answer_policy``), the package is rendered through the
          Answer_Policy and the text is placed on ``package.answer``.
        """
        c = self.container
        t = self.toggles

        # R0 — classify (always cheap; drives retriever routing).
        classification = c.query_classifier.classify(query_text)

        # R1 — symbolic retrieval (graph) — only when enabled.
        symbolic: list[Any] = []
        if t.use_graph:
            symbolic = c.symbolic_retriever.retrieve(classification, c.graph)

        # R2 — semantic retrieval (vectors) — only when enabled. For ungoverned
        # baselines (no quarantine governance) include quarantined items so they
        # behave like ordinary accepted memory ("accept-anyway").
        semantic: list[Any] = []
        if t.use_vectors:
            want_conflicts = include_conflicts or not t.use_quarantine
            semantic = c.semantic_retriever.retrieve(
                query_text,
                classification,
                top_k=top_k,
                include_conflicts=want_conflicts,
            )

        # Contradiction signal. Two governance regimes:
        # * write-time (``use_contradiction``): honour the write gate's quarantine
        #   records (B3/B4).
        # * read-time (``use_read_time_filter``): detect single-valued-relation
        #   conflicts live in accepted memory at query time, since an arm with the
        #   write gate off has no quarantine records (the retrieval-time filter
        #   baseline).
        if t.use_contradiction:
            contradicted_ids = self._contradicted_ids()
        elif t.use_read_time_filter:
            contradicted_ids = self._read_time_contradicted_ids()
        else:
            contradicted_ids = set()

        # R3 — rerank the selected hits into one ordered candidate set.
        weights = getattr(c.settings, "rerank_weights", None)
        ranked = c.reranker.rerank(
            symbolic,
            semantic,
            weights=weights,
            contradicted_ids=contradicted_ids,
        )

        # R4 — package. Provenance and quarantine awareness are gated by toggles.
        # A RAG-only arm (``use_structured_answer`` off) is packaged without the
        # graph so the answer is read only from retrieved text.
        provenance_tracker = c.provenance_tracker if t.use_provenance else None
        quarantine_store = c.quarantine_store if t.use_quarantine else None
        answer_graph = c.graph if t.use_structured_answer else None
        package = c.evidence_packager.package(
            query_text,
            classification,
            ranked,
            graph=answer_graph,
            provenance_tracker=provenance_tracker,
            quarantine_store=quarantine_store,
        )

        # B4 — render through the Answer_Policy (P1–P5).
        if t.use_answer_policy:
            rendered = self._render(package)
            if rendered is not None:
                package.answer = rendered

        return package

    # ================================================================== #
    # Internals
    # ================================================================== #
    def _contradicted_ids(self) -> set[str]:
        """Ids that unresolved quarantine records conflict with (R3 signal)."""
        store = self.container.quarantine_store
        contradicted: set[str] = set()
        try:
            records = store.list("unresolved")
        except Exception:  # pragma: no cover - defensive
            records = []
        for record in records or []:
            contradicted.update(getattr(record, "conflicting_ids", []) or [])
        return contradicted

    def _read_time_contradicted_ids(self) -> set[str]:
        """Detect single-valued-relation conflicts in accepted memory at read time.

        The retrieval-time-contradiction-filter baseline leaves the write gate
        off, so durable memory accumulates mutually contradictory accepted state
        and there are no quarantine records to consult. This scans the accepted
        assertions and, for every single-valued relation (cardinality 1:1 or m:1,
        e.g. ``HAS_STATUS`` / ``ASSIGNED_TO``) with two or more **distinct**
        accepted objects for the same subject, marks **all** assertions in that
        conflicting group as contradicted. The reranker then excludes them from
        confident support and the packager surfaces them as conflicts — the
        read-time analogue of write-time quarantining. Because no single side can
        be declared the winner after the fact, the whole group is flagged (the
        defensible conservative choice for a read-time filter).
        """
        from ocm.ontology.relations import RELATION_SIGNATURES, Cardinality

        single_valued = {Cardinality.ONE_TO_ONE, Cardinality.M_TO_ONE}
        try:
            accepted = list(self.container.repo.list_assertions("accepted"))
        except Exception:  # pragma: no cover - defensive
            return set()
        # (subject, predicate) -> {object_id: [assertion_id, ...]}
        groups: dict[tuple[str, str], dict[str, list[str]]] = {}
        for a in accepted:
            sig = RELATION_SIGNATURES.get(a.predicate)
            if sig is None or sig.cardinality not in single_valued:
                continue
            by_obj = groups.setdefault((a.subject_id, a.predicate), {})
            by_obj.setdefault(a.object_id, []).append(a.id)
        contradicted: set[str] = set()
        for by_obj in groups.values():
            if len(by_obj) > 1:  # >= 2 distinct objects for one subject => conflict
                for aids in by_obj.values():
                    contradicted.update(aids)
        return contradicted

    def _build_answer_policy(self) -> Optional[Any]:
        """Instantiate the Answer_Policy if available (defensive, task 16.2)."""
        if _AnswerPolicy is None:
            return None
        try:  # pragma: no cover - depends on parallel task's constructor.
            return _AnswerPolicy()
        except Exception:  # pragma: no cover - defensive
            return None

    def _render(self, package: EvidencePackage) -> Optional[str]:
        """Render ``package`` to an answer string for B4.

        Prefers the real Answer_Policy (``render(pkg, high_stakes) -> str``);
        falls back to a small built-in renderer when the policy is unavailable
        so B4 still produces a rendered answer before task 16.2 lands.
        """
        if self._answer_policy is not None:
            for attempt in (
                lambda: self._answer_policy.render(package, self.high_stakes),
                lambda: self._answer_policy.render(package, high_stakes=self.high_stakes),
                lambda: self._answer_policy.render(package),
            ):
                try:
                    return attempt()
                except TypeError:
                    continue
                except Exception:  # pragma: no cover - defensive
                    break
        return self._fallback_render(package)

    def _fallback_render(self, package: EvidencePackage) -> str:
        """Minimal P1–P5-flavoured renderer used when no Answer_Policy exists.

        P1: prefer accepted high-confidence support; P2/P3: surface conflicts
        separately; P5: state missing evidence; P4: note provenance when
        high-stakes.
        """
        parts: list[str] = []
        if package.answer:
            parts.append(str(package.answer))
        elif package.supporting_assertions:
            top = package.supporting_assertions[0]
            parts.append(
                f"Best supported by assertion {top.id} "
                f"(confidence {top.confidence:.2f})."
            )
        else:
            parts.append("No accepted supporting evidence was found.")

        if package.conflicts:
            parts.append(
                f"Unresolved conflict(s) detected: {len(package.conflicts)} "
                "(reported separately, not merged)."
            )
        if self.high_stakes and package.supporting_sources:
            parts.append(
                f"Provenance: {len(package.supporting_sources)} supporting source(s)."
            )
        if package.missing_information:
            parts.append("Missing: " + "; ".join(package.missing_information))
        return " ".join(parts)
