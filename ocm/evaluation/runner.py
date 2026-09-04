"""Baseline_Runner: execute B0–B3 against the benchmark (Req 22.6, 25.3, 28.9).

The :class:`BaselineRunner` drives the evaluation harness end to end. For each
baseline (B0–B3 by default, per Req 22.6 / 28.9) it builds a **fresh**
:class:`~ocm.core.container.CoreContainer` and
:class:`~ocm.evaluation.arms.strategies.MemoryStrategy` so the baselines never share
governed-memory state, then replays every
:class:`~ocm.evaluation.benchmark.BenchmarkExample`:

1. Each ``session`` is ingested as a write with ``source_ref`` qualified by the
   example id (``f"{example.id}:{session.session_id}"``) so provenance is unique
   per example.
2. Each ``question`` is then answered with ``strategy.query`` and scored
   pragmatically (does the answer / retrieved evidence contain the expected
   answer tokens, and does conflict-surfacing match ``expected_conflict``).

For every (baseline, example, question) the runner:

* appends a structured **result record** (the shape the Metrics_Reporter, task
  17.4, consumes) to the list returned by :meth:`run`, and
* emits a **benchmark research log** via
  :meth:`~ocm.core.logging.ResearchLogger.log_benchmark`
  (``baseline_name``, ``answer``, ``retrieved_ids``, ``conflicts``,
  ``expected_conflict``, ``score``, ``latency_ms``) (Req 25.3).

Scoring here is intentionally lightweight: the formal evaluation metrics
(hit@k, factual precision/recall, conflict-surfacing rate, …) are computed by
the Metrics_Reporter from these records; the runner only needs to produce
records carrying the required fields.

Requirements: 22.6, 25.3, 28.9.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.core.logging import ResearchLogger
from ocm.evaluation.arms import (
    DEFAULT_RUN_BASELINES,
    baseline_settings_overrides,
    build_baseline,
)
from ocm.evaluation.benchmark import BenchmarkExample
from ocm.evaluation.arms import MemoryStrategy
from ocm.retrieval.evidence_packager import EvidencePackage

#: Default retrieval depth used per question (matches the baseline tests).
DEFAULT_TOP_K: int = 10


def load_benchmark(path: str | Path) -> list[BenchmarkExample]:
    """Load benchmark examples from a JSONL file (one example per line).

    Blank lines are skipped. Each line is parsed and validated into a
    :class:`BenchmarkExample` (the inverse of
    :func:`ocm.evaluation.benchmark.write_jsonl`).
    """
    in_path = Path(path)
    examples: list[BenchmarkExample] = []
    with in_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            examples.append(BenchmarkExample.model_validate(json.loads(line)))
    return examples


def _default_settings() -> Settings:
    """Deterministic, offline settings so runs are hermetic and reproducible."""
    return Settings(
        deterministic_test_mode=True,
        chroma_mode="memory",
        extractor="mock",
    )


class BaselineRunner:
    """Executes baselines against the benchmark, scoring and logging each run.

    A single :class:`ResearchLogger` accumulates the per-benchmark-example
    benchmark logs across every baseline (Req 25.3); access it via
    :attr:`logger` or :meth:`benchmark_records`.
    """

    def __init__(
        self,
        *,
        logger: Optional[ResearchLogger] = None,
        settings_factory: Any = _default_settings,
        top_k: int = DEFAULT_TOP_K,
        token_counter: Optional[Any] = None,
    ) -> None:
        """Create a runner.

        Args:
            logger: Optional :class:`ResearchLogger` to record benchmark logs
                into. When omitted a fresh in-memory logger is created.
            settings_factory: Zero-arg callable returning the :class:`Settings`
                used to build each fresh container. Defaults to deterministic,
                offline settings so the run is reproducible.
            top_k: Retrieval depth passed to ``strategy.query`` per question.
            token_counter: Optional ``callable(str) -> int`` used to measure
                context size for the efficiency table's token-overhead column. A
                model tokenizer's encoder gives true token counts for the LLM
                run; when omitted a whitespace split is used (a ratio-preserving
                proxy that keeps the offline reference deterministic).
        """
        self.logger = logger or ResearchLogger()
        self._settings_factory = settings_factory
        self.top_k = top_k
        self._token_counter = token_counter

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        examples: list[BenchmarkExample],
        baselines: Iterable[str] = DEFAULT_RUN_BASELINES,
    ) -> list[dict]:
        """Run ``baselines`` over ``examples`` and return per-question records.

        For each baseline a **fresh** container + strategy is built (so no
        memory state leaks between baselines), every example's sessions are
        ingested, and every question is queried and scored. Returns one result
        record per (baseline, example, question) — the records the
        Metrics_Reporter (task 17.4) consumes — and emits a benchmark research
        log per record (Req 25.3).
        """
        records: list[dict] = []
        for baseline_name in baselines:
            settings = self._settings_factory().model_copy(
                update=baseline_settings_overrides(baseline_name)
            )
            container = CoreContainer(settings)
            strategy = build_baseline(baseline_name, container)
            for example in examples:
                wc = self._ingest_sessions(strategy, example)
                for q_index, question in enumerate(example.questions):
                    record = self._run_question(
                        baseline_name, strategy, example, q_index, question,
                        write_quarantined=wc["quarantined"],
                    )
                    records.append(record)
        return records

    def benchmark_records(self) -> list[dict]:
        """Return only the benchmark-kind records emitted to the logger."""
        return [r for r in self.logger.records if r.get("kind") == "benchmark"]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _ingest_sessions(
        self, strategy: MemoryStrategy, example: BenchmarkExample
    ) -> dict[str, int]:
        """Write every session of ``example`` through the governed pipeline.

        Returns a dict of write-outcome counts summed across the example's
        sessions: ``candidates``, ``accepted``, ``superseded``, ``quarantined``,
        ``rejected``. These per-arm write-side counts let the harness show
        whether an ablation actually changes what the governed write path admits
        (divergence) or merely fires a different gate for the same net outcome
        (redundancy), rather than leaving identical decisive metrics unexplained.
        """
        counts = {
            "candidates": 0,
            "accepted": 0,
            "superseded": 0,
            "quarantined": 0,
            "rejected": 0,
            "write_ms": 0.0,
            "write_calls": 0,
        }
        for session in example.sessions:
            source_ref = f"{example.id}:{session.session_id}"
            _t0 = time.perf_counter()
            result = strategy.write(session.input, source_ref)
            counts["write_ms"] += (time.perf_counter() - _t0) * 1000.0
            counts["write_calls"] += 1
            summary = result.summary
            counts["candidates"] += int(getattr(summary, "num_candidates", 0) or 0)
            counts["accepted"] += int(getattr(summary, "num_accepted", 0) or 0)
            counts["superseded"] += int(getattr(summary, "num_superseded", 0) or 0)
            counts["quarantined"] += int(getattr(summary, "num_quarantined", 0) or 0)
            counts["rejected"] += int(getattr(summary, "num_rejected", 0) or 0)
        return counts

    def _run_question(
        self,
        baseline_name: str,
        strategy: MemoryStrategy,
        example: BenchmarkExample,
        q_index: int,
        question: Any,
        write_quarantined: int = 0,
    ) -> dict:
        """Query one question, score it, log it, and build a result record."""
        start = time.perf_counter()
        package = strategy.query(question.query, top_k=self.top_k)
        latency_ms = (time.perf_counter() - start) * 1000.0

        retrieved_ids = [item.memory_id for item in package.retrieved_items]
        supporting_ids = [sa.id for sa in package.supporting_assertions]
        conflict_ids = [
            c.memory_id for c in package.conflicts if c.memory_id is not None
        ]
        conflict_surfaced = bool(package.conflicts)
        answer = package.answer or ""

        # Durable-write constraint-violation exposure (decisive metric, paper
        # §IV-B). All arms share the governed write path, so the *durable store*
        # is identical; governance differs in what each surfaces at answer time.
        # We count a response as exposing a constraint violation when it surfaces
        # a constraint-violating durable item — one whose status is not
        # ``accepted`` (e.g. quarantined/superseded) or that is contradicted —
        # WITHOUT flagging it as a conflict. Governed baselines (B3) exclude or
        # flag such state; ungoverned baselines (B0–B2) fold it in unflagged.
        flagged_conflicts = set(conflict_ids)
        surfaced_violation = any(
            (
                str(getattr(item, "status", "accepted")) != "accepted"
                or bool(getattr(item, "contradicted", False))
            )
            and getattr(item, "memory_id", None) not in flagged_conflicts
            for item in package.retrieved_items
        )

        answer_score = self._answer_score(question, package)
        conflict_correct = conflict_surfaced == bool(question.expected_conflict)
        # Task success = answering / plan completion only (paper §IV-B). It is
        # the fraction of expected answer tokens recalled, decoupled from
        # conflict-surfacing — that is measured independently by the
        # contradiction-rate metric, and false-quarantine by the tau-sweep.
        # Bundling conflict-surfacing into task success previously double-counted
        # conflict behavior and penalized the governed system for surfacing
        # (even correctly), so the three decisive metrics were not independent.
        score = answer_score
        # Calibration signals: the package's confidence and whether the answer
        # was fully correct (all expected tokens present).
        answer_correct = answer_score >= 1.0

        # Per-benchmark-example research log (Req 25.3).
        self.logger.log_benchmark(
            baseline_name=baseline_name,
            answer=answer,
            retrieved_ids=retrieved_ids,
            conflicts=conflict_ids,
            expected_conflict=bool(question.expected_conflict),
            score=score,
            latency_ms=latency_ms,
        )

        return {
            "baseline_name": baseline_name,
            "example_id": example.id,
            "category": example.category,
            "intensity": getattr(example, "intensity", None),
            "question_index": q_index,
            "query": question.query,
            "answer": answer,
            "retrieved_ids": retrieved_ids,
            "supporting_ids": supporting_ids,
            "supporting_source_count": len(package.supporting_sources),
            "conflict_ids": conflict_ids,
            "conflict_surfaced": conflict_surfaced,
            "expected_conflict": bool(question.expected_conflict),
            "expected_answer_contains": list(question.expected_answer_contains),
            "expected_supporting_ids": list(question.expected_supporting_ids or []),
            "answer_score": answer_score,
            "answer_correct": answer_correct,
            "surfaced_violation": bool(surfaced_violation),
            "context_tokens": self._context_tokens(package),
            "package_confidence": float(package.confidence),
            "write_quarantined": int(write_quarantined),
            "conflict_correct": conflict_correct,
            "score": score,
            "latency_ms": latency_ms,
        }

    @staticmethod
    def _answer_score(question: Any, package: EvidencePackage) -> float:
        """Fraction of expected answer tokens found in the answer/evidence.

        Builds a case-insensitive haystack from the rendered answer plus the
        retrieved items' text and entity ids, then returns the share of
        ``expected_answer_contains`` tokens present. Returns ``1.0`` when no
        tokens are expected.
        """
        expected = list(question.expected_answer_contains or [])
        if not expected:
            return 1.0
        haystack = BaselineRunner._haystack(package).lower()
        hits = sum(1 for token in expected if str(token).lower() in haystack)
        return hits / len(expected)

    def _context_tokens(self, package: EvidencePackage) -> int:
        """Token count of the context a method would feed downstream.

        A proxy for prompt/context size used by the efficiency table's token-
        overhead column (paper Table V): the rendered answer plus the text of
        every retrieved item, surfaced conflict, and provenance source the
        package carries. Governed arms add provenance/conflict annotations, so
        their context is larger — this quantifies that overhead. Uses the
        injected ``token_counter`` (a real model tokenizer for the LLM run) when
        available, else a whitespace split.
        """
        parts: list[str] = []
        if package.answer:
            parts.append(str(package.answer))
        for item in package.retrieved_items:
            if getattr(item, "text", None):
                parts.append(str(item.text))
        for conflict in package.conflicts:
            for value in (getattr(conflict, "text", None), getattr(conflict, "reason", None)):
                if value:
                    parts.append(str(value))
        for source in package.supporting_sources:
            ref = getattr(source, "source_ref", None)
            if ref:
                parts.append(str(ref))
        text = " ".join(parts)
        if self._token_counter is not None:
            try:
                return int(len(self._token_counter(text)))
            except TypeError:
                # Counter may return an int directly rather than a sequence.
                return int(self._token_counter(text))
            except Exception:  # pragma: no cover - defensive: fall back to proxy
                pass
        return len(text.split())

    @staticmethod
    def _haystack(package: EvidencePackage) -> str:
        """Concatenate the answer + retrieved-item text/ids for token matching."""
        parts: list[str] = []
        if package.answer:
            parts.append(str(package.answer))
        for item in package.retrieved_items:
            for value in (
                item.text,
                item.predicate,
                item.subject_id,
                item.object_id,
                item.memory_id,
            ):
                if value:
                    parts.append(str(value))
        for conflict in package.conflicts:
            if conflict.text:
                parts.append(str(conflict.text))
        return " ".join(parts)
