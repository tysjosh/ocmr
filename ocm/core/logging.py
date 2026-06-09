"""Structured research logging for OCM (Req 25).

`ResearchLogger` writes **append-only JSONL** records at three granularities so
experiments are traceable and reproducible:

* :meth:`log_write` — one record per write operation (Req 25.1).
* :meth:`log_query` — one record per query operation (Req 25.2).
* :meth:`log_benchmark` — one record per benchmark example evaluation (Req 25.3).

Records are keyed by ids (``input_id`` / ``query_id`` / ``baseline_name``) so
writes, queries, and benchmark evaluations can be joined for analysis. The
logger appends one JSON object per line; if no path is configured it buffers
records in memory (useful for tests).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class ResearchLogger:
    """Append-only JSONL logger for writes, queries, and benchmark runs."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        #: In-memory mirror of every record emitted, primarily for tests and
        #: when no ``path`` is configured.
        self.records: List[Dict[str, Any]] = []
        if self.path:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)

    # -- internal -----------------------------------------------------------
    def _emit(self, kind: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """Stamp, store, and (optionally) persist a single JSONL record."""
        enriched = {
            "kind": kind,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        self.records.append(enriched)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(enriched, default=str) + "\n")
        return enriched

    # -- per-write (Req 25.1) ----------------------------------------------
    def log_write(
        self,
        *,
        input_id: str,
        source_ref: str,
        number_of_candidates: int,
        number_accepted: int,
        number_quarantined: int,
        number_rejected: int,
        validation_failures: int,
        contradiction_failures: int,
        latency_ms: float,
        token_count_if_llm_used: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Record a completed write operation (Req 25.1)."""
        return self._emit(
            "write",
            {
                "input_id": input_id,
                "source_ref": source_ref,
                "number_of_candidates": number_of_candidates,
                "number_accepted": number_accepted,
                "number_quarantined": number_quarantined,
                "number_rejected": number_rejected,
                "validation_failures": validation_failures,
                "contradiction_failures": contradiction_failures,
                "latency_ms": latency_ms,
                "token_count_if_llm_used": token_count_if_llm_used,
            },
        )

    # -- per-query (Req 25.2) ----------------------------------------------
    def log_query(
        self,
        *,
        query_id: str,
        query_type: str,
        symbolic_results_count: int,
        semantic_results_count: int,
        top_k_ids: List[str],
        conflicts_returned: int,
        latency_ms: float,
        token_count_if_llm_used: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Record a completed query operation (Req 25.2)."""
        return self._emit(
            "query",
            {
                "query_id": query_id,
                "query_type": query_type,
                "symbolic_results_count": symbolic_results_count,
                "semantic_results_count": semantic_results_count,
                "top_k_ids": list(top_k_ids),
                "conflicts_returned": conflicts_returned,
                "latency_ms": latency_ms,
                "token_count_if_llm_used": token_count_if_llm_used,
            },
        )

    # -- per-benchmark-example (Req 25.3) ----------------------------------
    def log_benchmark(
        self,
        *,
        baseline_name: str,
        answer: str,
        retrieved_ids: List[str],
        conflicts: List[Any],
        expected_conflict: bool,
        score: float,
        latency_ms: float,
    ) -> Dict[str, Any]:
        """Record a single benchmark example evaluation (Req 25.3)."""
        return self._emit(
            "benchmark",
            {
                "baseline_name": baseline_name,
                "answer": answer,
                "retrieved_ids": list(retrieved_ids),
                "conflicts": list(conflicts),
                "expected_conflict": expected_conflict,
                "score": score,
                "latency_ms": latency_ms,
            },
        )
