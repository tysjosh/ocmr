"""Metrics_Reporter — the full evaluation metric suite (Req 24).

This module turns the per-(baseline, example, question) result records produced
by the Baseline_Runner (``ocm/evaluation/runner.py``) into the four metric
families the research claim is assessed on, plus deltas of every baseline
against **B0** (Req 24.5).

Result record contract
-----------------------
The reporter is *tolerant*: a record may be a plain ``dict`` or any object with
attributes (e.g. a dataclass / pydantic ``ResultRecord``). It is read through
:func:`_get`, which falls back across attribute and item access, so the runner
and the reporter can evolve independently. The fields the reporter consumes
(all optional, defaulted) are:

* ``baseline_name`` — e.g. ``"B0"`` … ``"B4"`` (records are grouped by this).
* ``example_id`` / ``category`` — provenance + per-category agent proxies.
* ``query`` — the question text (reporting only).
* ``answer`` — the system's rendered answer (``str`` or ``None``).
* ``retrieved_ids`` — ordered list of retrieved memory ids (rank-1 first).
* ``conflicts`` — surfaced conflicts as a count or a list.
* ``expected_answer_contains`` — tokens that must appear in a correct answer.
* ``expected_conflict`` — whether the question is a known conflict.
* ``expected_supporting_ids`` — gold supporting memory ids (or ``None``).
* ``score`` / ``latency_ms`` — runner-provided score and latency.

Some Req 24.3 write-time metrics need write-side signals that are not present in
question-level records. Those are computed from optional fields when the runner
supplies them (``expected_invalid_write`` / ``invalid_write_detected``,
``quarantined`` / ``expected_quarantine``, ``entity_resolution_correct``) and
otherwise reported as ``None`` with an explanatory note rather than a misleading
zero.

Output shape
------------
:meth:`MetricsReporter.compute` returns::

    {
      "B0": {"retrieval": {...}, "answer": {...}, "write_time": {...},
             "agent": {...}, "counts": {...}},
      "B1": {...}, ...,
      "comparisons_vs_B0": {"B1": {"<metric>_delta": value, ...}, ...},
      "_meta": {"baselines": [...], "total_records": N, "b0_present": bool,
                "notes": [...]},
    }

:meth:`MetricsReporter.report` renders a readable summary table from the same
data.

Requirements: 24.1, 24.2, 24.3, 24.4, 24.5.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

# The canonical comparison baseline (Req 24.5).
B0 = "B0"

# Category whose per-record correctness proxies long-horizon plan success.
_PLANNING_CATEGORY = "multi_step_planning_entity_consistency"

# Token names the runner might use for an LLM token count (best-effort).
_TOKEN_FIELDS: tuple[str, ...] = (
    "token_count",
    "token_count_if_llm_used",
    "tokens",
    "token_overhead",
)

# Key metrics carried into the B0 comparison (flattened group.metric paths).
_COMPARISON_METRICS: tuple[str, ...] = (
    "retrieval.hit@1",
    "retrieval.hit@3",
    "retrieval.hit@5",
    "retrieval.supporting_evidence_precision",
    "retrieval.supporting_evidence_recall",
    "answer.factual_precision",
    "answer.factual_recall",
    "answer.conflict_surfacing_rate",
    "answer.memory_induced_hallucination_rate",
    "answer.contradiction_rate_per_100_responses",
    "write_time.contradiction_detection_precision",
    "write_time.contradiction_detection_recall",
    "agent.long_horizon_plan_success",
    "agent.answer_quality",
    "agent.latency_overhead",
    "agent.token_overhead",
)


# --------------------------------------------------------------------------- #
# Tolerant field access
# --------------------------------------------------------------------------- #
def _get(record: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict-like or attribute-bearing ``record``.

    Tries mapping access first, then attribute access, returning ``default``
    when neither is present. This keeps the reporter agnostic to whether the
    runner emits dicts or a typed ``ResultRecord``.
    """
    if isinstance(record, Mapping):
        return record.get(key, default)
    if hasattr(record, key):
        return getattr(record, key)
    # Some records expose a ``.get`` without being a Mapping subclass.
    getter = getattr(record, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return default


def _conflict_count(record: Any) -> int:
    """Number of surfaced conflicts, tolerant of count-or-list ``conflicts``."""
    conflicts = _get(record, "conflicts", 0)
    if conflicts is None:
        return 0
    if isinstance(conflicts, bool):
        return 1 if conflicts else 0
    if isinstance(conflicts, int):
        return max(conflicts, 0)
    try:
        return len(conflicts)  # list / tuple / set
    except TypeError:
        return 0


def _as_list(value: Any) -> list[Any]:
    """Coerce a possibly-``None`` / scalar value into a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _answer_text(record: Any) -> str:
    """The answer string for a record (``""`` when absent/None)."""
    answer = _get(record, "answer", "")
    return answer if isinstance(answer, str) else ("" if answer is None else str(answer))


def _has_answer(record: Any) -> bool:
    """Whether the record produced a non-empty answer."""
    return bool(_answer_text(record).strip())


def _is_answer_correct(record: Any) -> bool:
    """A record is correct when every expected token appears in the answer.

    Matching is case-insensitive substring containment against
    ``expected_answer_contains``. A question with no expected tokens is treated
    as correct iff it produced an answer (nothing to get wrong).
    """
    expected = [str(t) for t in _as_list(_get(record, "expected_answer_contains", []))]
    answer = _answer_text(record).lower()
    if not expected:
        return _has_answer(record)
    return all(token.lower() in answer for token in expected if token)


def _token_count(record: Any) -> Optional[float]:
    """Best-effort token count for a record, or ``None`` if unavailable."""
    for field in _TOKEN_FIELDS:
        value = _get(record, field, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _mean(values: Sequence[float]) -> Optional[float]:
    """Arithmetic mean, or ``None`` for an empty sequence."""
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    """Safe ratio, ``None`` when the denominator is zero."""
    return numerator / denominator if denominator else None


# --------------------------------------------------------------------------- #
# Metrics_Reporter
# --------------------------------------------------------------------------- #
class MetricsReporter:
    """Computes the Req 24 metric suite and B0 comparisons over result records."""

    def compute(self, results: Iterable[Any]) -> dict[str, Any]:
        """Compute all metrics grouped by ``baseline_name`` (Req 24.1–24.5).

        Args:
            results: An iterable of per-question result records (dicts or typed
                records). Records missing ``baseline_name`` are grouped under
                ``"unknown"``.

        Returns:
            The structured metrics dict documented in the module docstring,
            including ``comparisons_vs_B0`` and a ``_meta`` block.
        """
        records = list(results)
        grouped: dict[str, list[Any]] = {}
        for record in records:
            name = _get(record, "baseline_name", "unknown") or "unknown"
            grouped.setdefault(str(name), []).append(record)

        notes: list[str] = []
        report: dict[str, Any] = {}
        for baseline in sorted(grouped):
            report[baseline] = self._compute_for_baseline(grouped[baseline], notes)

        b0_present = B0 in report
        report["comparisons_vs_B0"] = self._compare_to_b0(report, b0_present, notes)
        report["_meta"] = {
            "baselines": sorted(grouped),
            "total_records": len(records),
            "b0_present": b0_present,
            "notes": _dedupe(notes),
        }
        return report

    # ------------------------------------------------------------------ #
    # Per-baseline computation
    # ------------------------------------------------------------------ #
    def _compute_for_baseline(self, records: list[Any], notes: list[str]) -> dict[str, Any]:
        retrieval = self._retrieval_metrics(records, notes)
        answer = self._answer_metrics(records)
        write_time = self._write_time_metrics(records, notes)
        agent = self._agent_metrics(records)
        counts = {
            "questions": len(records),
            "answered": sum(1 for r in records if _has_answer(r)),
            "correct": sum(1 for r in records if _is_answer_correct(r)),
            "expected_conflicts": sum(1 for r in records if _get(r, "expected_conflict", False)),
            "with_supporting_ids": sum(
                1 for r in records if _as_list(_get(r, "expected_supporting_ids", None))
            ),
        }
        return {
            "retrieval": retrieval,
            "answer": answer,
            "write_time": write_time,
            "agent": agent,
            "counts": counts,
        }

    # -- Req 24.1: retrieval -------------------------------------------- #
    def _retrieval_metrics(self, records: list[Any], notes: list[str]) -> dict[str, Any]:
        scored = [r for r in records if _as_list(_get(r, "expected_supporting_ids", None))]
        if not scored:
            notes.append(
                "retrieval: no records carried expected_supporting_ids; "
                "hit@k and supporting-evidence precision/recall are None."
            )
            return {
                "hit@1": None,
                "hit@3": None,
                "hit@5": None,
                "supporting_evidence_precision": None,
                "supporting_evidence_recall": None,
            }

        hits = {1: [], 3: [], 5: []}
        precisions: list[float] = []
        recalls: list[float] = []
        for record in scored:
            expected = {str(x) for x in _as_list(_get(record, "expected_supporting_ids", None))}
            retrieved = [str(x) for x in _as_list(_get(record, "retrieved_ids", []))]
            retrieved_set = set(retrieved)
            relevant_retrieved = len(retrieved_set & expected)
            precisions.append((relevant_retrieved / len(retrieved_set)) if retrieved_set else 0.0)
            recalls.append((relevant_retrieved / len(expected)) if expected else 0.0)
            for k in hits:
                top_k = set(retrieved[:k])
                hits[k].append(1.0 if top_k & expected else 0.0)

        return {
            "hit@1": _mean(hits[1]),
            "hit@3": _mean(hits[3]),
            "hit@5": _mean(hits[5]),
            "supporting_evidence_precision": _mean(precisions),
            "supporting_evidence_recall": _mean(recalls),
        }

    # -- Req 24.2: answer ----------------------------------------------- #
    def _answer_metrics(self, records: list[Any]) -> dict[str, Any]:
        total = len(records)
        answered = sum(1 for r in records if _has_answer(r))
        correct = sum(1 for r in records if _is_answer_correct(r))

        # factual precision: of the answers produced, how many are correct.
        # factual recall: of all questions, how many got a correct answer.
        factual_precision = _ratio(correct, answered)
        factual_recall = _ratio(correct, total)

        # conflict surfacing: of expected-conflict questions, fraction surfaced.
        expected_conflict_records = [r for r in records if _get(r, "expected_conflict", False)]
        surfaced = sum(1 for r in expected_conflict_records if _conflict_count(r) > 0)
        conflict_surfacing_rate = _ratio(surfaced, len(expected_conflict_records))

        # A response is a "contradiction response" when the question is a known
        # conflict, the system still asserted an answer, and it did NOT surface
        # the conflict — i.e. it spoke confidently over contradictory memory.
        contradiction_responses = sum(
            1
            for r in expected_conflict_records
            if _has_answer(r) and _conflict_count(r) == 0
        )
        contradiction_rate_per_100 = (
            (contradiction_responses / total) * 100.0 if total else None
        )

        # Memory-induced hallucination: a non-empty answer that is incorrect and
        # carries no surfaced conflict to flag the uncertainty.
        hallucinations = sum(
            1
            for r in records
            if _has_answer(r) and not _is_answer_correct(r) and _conflict_count(r) == 0
        )
        hallucination_rate = _ratio(hallucinations, total)

        # Calibration: ECE / Brier over the package confidence vs correctness,
        # when the runner supplies the optional ``package_confidence`` field.
        cal_records = [
            r for r in records if _get(r, "package_confidence", None) is not None
        ]
        if cal_records:
            from ocm.evaluation import stats as _stats

            confidences = [float(_get(r, "package_confidence", 0.0)) for r in cal_records]
            correct = [bool(_is_answer_correct(r)) for r in cal_records]
            ece = _stats.expected_calibration_error(confidences, correct)
            brier = _stats.brier_score(confidences, correct)
        else:
            ece = None
            brier = None

        return {
            "factual_precision": factual_precision,
            "factual_recall": factual_recall,
            "contradiction_rate_per_100_responses": contradiction_rate_per_100,
            "conflict_surfacing_rate": conflict_surfacing_rate,
            "memory_induced_hallucination_rate": hallucination_rate,
            "expected_calibration_error": ece,
            "brier_score": brier,
        }

    # -- Req 24.3: write-time ------------------------------------------- #
    def _write_time_metrics(self, records: list[Any], notes: list[str]) -> dict[str, Any]:
        # Contradiction detection precision/recall treat a surfaced conflict as
        # the positive prediction and expected_conflict as the gold label.
        tp = fp = fn = 0
        for record in records:
            predicted = _conflict_count(record) > 0
            actual = bool(_get(record, "expected_conflict", False))
            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and actual:
                fn += 1
        contradiction_precision = _ratio(tp, tp + fp)
        contradiction_recall = _ratio(tp, tp + fn)

        # The remaining write-time metrics need write-side signals. Compute them
        # only from optional runner fields; otherwise report None + a note.
        invalid_rate = self._optional_rate(
            records,
            condition="invalid_write_detected",
            population="expected_invalid_write",
        )
        if invalid_rate is None:
            notes.append(
                "write_time: invalid_write_detection_rate needs write-side fields "
                "(expected_invalid_write / invalid_write_detected); reported None."
            )

        false_quarantine = self._optional_false_rate(
            records,
            flag="quarantined",
            expected="expected_quarantine",
        )
        if false_quarantine is None:
            notes.append(
                "write_time: false_quarantine_rate needs write-side fields "
                "(quarantined / expected_quarantine); reported None."
            )

        er_records = [
            r for r in records if _get(r, "entity_resolution_correct", None) is not None
        ]
        entity_resolution_accuracy: Optional[float]
        if er_records:
            entity_resolution_accuracy = _mean(
                [1.0 if _get(r, "entity_resolution_correct", False) else 0.0 for r in er_records]
            )
        else:
            entity_resolution_accuracy = None
            notes.append(
                "write_time: entity_resolution_accuracy needs the optional "
                "entity_resolution_correct field; reported None."
            )

        return {
            "invalid_write_detection_rate": invalid_rate,
            "false_quarantine_rate": false_quarantine,
            "contradiction_detection_precision": contradiction_precision,
            "contradiction_detection_recall": contradiction_recall,
            "entity_resolution_accuracy": entity_resolution_accuracy,
        }

    @staticmethod
    def _optional_rate(
        records: list[Any], *, condition: str, population: str
    ) -> Optional[float]:
        """Detection rate over records flagged by ``population``.

        Returns the fraction of ``population``-true records where ``condition``
        is also true, or ``None`` when no record carries the ``population`` flag.
        """
        pop = [r for r in records if _get(r, population, None) is not None]
        positives = [r for r in pop if _get(r, population, False)]
        if not positives:
            # No labelled invalid writes available to score against.
            if not any(_get(r, condition, None) is not None for r in records):
                return None
            return None
        detected = sum(1 for r in positives if _get(r, condition, False))
        return detected / len(positives)

    @staticmethod
    def _optional_false_rate(
        records: list[Any], *, flag: str, expected: str
    ) -> Optional[float]:
        """False-positive rate of ``flag`` against ``expected`` gold labels.

        Of the records that were *not* expected to be flagged, the fraction that
        nonetheless carried ``flag``. ``None`` when neither field is present.
        """
        labelled = [
            r
            for r in records
            if _get(r, flag, None) is not None or _get(r, expected, None) is not None
        ]
        if not labelled:
            return None
        negatives = [r for r in labelled if not _get(r, expected, False)]
        if not negatives:
            return None
        false_flags = sum(1 for r in negatives if _get(r, flag, False))
        return false_flags / len(negatives)

    # -- Req 24.4: agent ------------------------------------------------ #
    def _agent_metrics(self, records: list[Any]) -> dict[str, Any]:
        # Long-horizon plan success: correctness proxy on the multi-step
        # planning category (entity-consistency across turns).
        planning = [r for r in records if _get(r, "category", None) == _PLANNING_CATEGORY]
        long_horizon = _mean(
            [1.0 if _is_answer_correct(r) else 0.0 for r in planning]
        )

        # Overall answer-quality proxy (per-baseline correctness rate).
        answer_quality = _mean([1.0 if _is_answer_correct(r) else 0.0 for r in records])

        # Latency / token means feed the B0 overhead deltas (Req 24.5).
        latencies = [
            float(_get(r, "latency_ms", 0.0) or 0.0)
            for r in records
            if isinstance(_get(r, "latency_ms", None), (int, float))
            and not isinstance(_get(r, "latency_ms", None), bool)
        ]
        mean_latency = _mean(latencies)
        token_values = [tc for tc in (_token_count(r) for r in records) if tc is not None]
        mean_tokens = _mean(token_values)

        return {
            # Correction turns after an injected error require multi-turn agent
            # transcripts the question-level records do not carry.
            "correction_turns_after_injected_error": None,
            "long_horizon_plan_success": long_horizon,
            "answer_quality": answer_quality,
            "mean_latency_ms": mean_latency,
            # Overheads are baseline-relative; filled per baseline as the raw
            # mean here and converted to a delta vs B0 in comparisons.
            "latency_overhead": mean_latency,
            "token_overhead": mean_tokens,
        }

    # ------------------------------------------------------------------ #
    # Req 24.5: comparison vs B0
    # ------------------------------------------------------------------ #
    def _compare_to_b0(
        self, report: dict[str, Any], b0_present: bool, notes: list[str]
    ) -> dict[str, Any]:
        if not b0_present:
            notes.append("comparison: B0 missing from results; deltas omitted.")
            return {}
        b0_metrics = report[B0]
        comparisons: dict[str, Any] = {}
        for baseline, metrics in report.items():
            if baseline in (B0, "comparisons_vs_B0", "_meta"):
                continue
            deltas: dict[str, Any] = {}
            for path in _COMPARISON_METRICS:
                base = _lookup(b0_metrics, path)
                value = _lookup(metrics, path)
                deltas[f"{path.split('.')[-1]}_delta"] = _delta(value, base)
            comparisons[baseline] = deltas
        return comparisons

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def report(self, results: Iterable[Any]) -> str:
        """Render a human-readable summary table of the computed metrics."""
        data = self.compute(results)
        baselines = [b for b in data["_meta"]["baselines"] if b in data]
        lines: list[str] = []
        lines.append("Metrics_Reporter summary (Req 24)")
        lines.append(f"  records: {data['_meta']['total_records']}  baselines: {', '.join(baselines) or '-'}")
        lines.append("")

        rows = [
            ("hit@1", "retrieval.hit@1"),
            ("hit@3", "retrieval.hit@3"),
            ("hit@5", "retrieval.hit@5"),
            ("supp_prec", "retrieval.supporting_evidence_precision"),
            ("supp_rec", "retrieval.supporting_evidence_recall"),
            ("fact_prec", "answer.factual_precision"),
            ("fact_rec", "answer.factual_recall"),
            ("conflict_surf", "answer.conflict_surfacing_rate"),
            ("halluc_rate", "answer.memory_induced_hallucination_rate"),
            ("contra/100", "answer.contradiction_rate_per_100_responses"),
            ("contra_prec", "write_time.contradiction_detection_precision"),
            ("contra_rec", "write_time.contradiction_detection_recall"),
            ("plan_success", "agent.long_horizon_plan_success"),
            ("answer_qual", "agent.answer_quality"),
            ("latency_ms", "agent.mean_latency_ms"),
        ]

        header = f"  {'metric':<16}" + "".join(f"{b:>12}" for b in baselines)
        lines.append(header)
        lines.append("  " + "-" * (16 + 12 * len(baselines)))
        for label, path in rows:
            cells = "".join(f"{_fmt(_lookup(data[b], path)):>12}" for b in baselines)
            lines.append(f"  {label:<16}{cells}")

        comparisons = data.get("comparisons_vs_B0", {})
        if comparisons:
            lines.append("")
            lines.append("  Deltas vs B0 (key metrics):")
            for baseline, deltas in comparisons.items():
                key_keys = (
                    "conflict_surfacing_rate_delta",
                    "memory_induced_hallucination_rate_delta",
                    "factual_precision_delta",
                    "hit@5_delta",
                )
                summary = ", ".join(
                    f"{k.replace('_delta', '')}={_fmt(deltas.get(k))}" for k in key_keys
                )
                lines.append(f"    {baseline}: {summary}")
        elif not data["_meta"]["b0_present"]:
            lines.append("")
            lines.append("  (B0 absent from results — no comparison computed.)")

        notes = data["_meta"]["notes"]
        if notes:
            lines.append("")
            lines.append("  Notes:")
            for note in notes:
                lines.append(f"    - {note}")

        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Small helpers used by compute/report
# --------------------------------------------------------------------------- #
def _lookup(metrics: Mapping[str, Any], path: str) -> Any:
    """Read a ``group.metric`` dotted path from a per-baseline metrics dict."""
    node: Any = metrics
    for part in path.split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)
    return node


def _delta(value: Any, base: Any) -> Optional[float]:
    """``value - base`` when both are numbers, else ``None``."""
    if isinstance(value, (int, float)) and isinstance(base, (int, float)):
        if isinstance(value, bool) or isinstance(base, bool):
            return None
        return float(value) - float(base)
    return None


def _fmt(value: Any) -> str:
    """Format a metric value for the report table."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return str(value)


def _dedupe(items: list[str]) -> list[str]:
    """Order-preserving de-duplication of note strings."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
