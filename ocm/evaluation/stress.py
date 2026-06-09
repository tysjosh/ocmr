"""Stress benchmark with controlled perturbation classes (paper §IV-C).

Constructs four controlled perturbation classes, each replayed across multi-
session trajectories with increasing perturbation **intensity** (low / medium /
high), to probe persistent-memory brittleness (Tables IX and VIII):

1. ``conflicting_updates``      — mutually conflicting single-valued updates
   (reassignments / status flips) that the governance gate must surface.
2. ``alias_ambiguity``          — the same entity referred to under several
   alias surface forms with partial context (drives entity-resolution F1 and
   false-merge rate).
3. ``temporal_overlap``         — overlapping / cyclic ``PRECEDES`` orderings.
4. ``confidence_inflation``     — high-confidence assertions / final decisions
   asserted without supporting evidence.

Generation is fully seeded so the dataset is byte-identical for a fixed seed.
Session text uses only patterns the offline ``Mock_Extractor`` recognizes, so
the stress suite runs end-to-end with no network or API key.
"""

from __future__ import annotations

import random
from typing import Optional

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.evaluation.benchmark import BenchmarkExample, Question, Session

#: Perturbation classes (paper §IV-C).
PERTURBATION_CLASSES: tuple[str, ...] = (
    "conflicting_updates",
    "alias_ambiguity",
    "temporal_overlap",
    "confidence_inflation",
)

#: Intensity regimes (paper Table IX) and the perturbation count each implies.
INTENSITY_LEVELS: tuple[str, ...] = ("low", "medium", "high")
_INTENSITY_COUNT: dict[str, int] = {"low": 1, "medium": 3, "high": 5}

DEFAULT_SEED: int = 4242

_NAMES = (
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi",
    "Ivan", "Judy", "Mallory", "Niaj", "Olivia", "Peggy", "Trent", "Victor",
)
_TASKS = tuple(f"T{i}" for i in range(1, 30))
_EVENTS = (
    "Kickoff", "Design", "Build", "Review", "Launch", "Retro", "Audit", "Demo",
)
_PROJECTS = ("Orion", "Apollo", "Hermes", "Atlas", "Titan", "Nova", "Vega")
_PER_CLASS = 30  # trajectories per (class, intensity)


def _conflicting_updates(rng: random.Random, idx: int, intensity: str) -> BenchmarkExample:
    """A task reassigned (and status-flipped) repeatedly — single-valued conflicts."""
    n = _INTENSITY_COUNT[intensity]
    task = rng.choice(_TASKS)
    people = rng.sample(_NAMES, k=min(n + 1, len(_NAMES)))
    sessions = [Session(session_id="s0", input=f"{people[0]} is assigned to Task {task}.")]
    for i in range(1, n + 1):
        sessions.append(
            Session(session_id=f"s{i}", input=f"{people[i]} is assigned to Task {task}.")
        )
    return BenchmarkExample(
        id=f"stress-conflict-{intensity}-{idx:03d}",
        category="stress_conflicting_updates",
        intensity=intensity,
        sessions=sessions,
        questions=[
            Question(
                query=f"Who is assigned to Task {task}?",
                expected_answer_contains=[people[0]],
                expected_conflict=True,
            )
        ],
    )


def _alias_ambiguity(rng: random.Random, idx: int, intensity: str) -> BenchmarkExample:
    """One person under several alias surface forms with partial context."""
    n = _INTENSITY_COUNT[intensity]
    base = rng.choice(_NAMES)
    project = rng.choice(_PROJECTS)
    # Alias surface forms of increasing ambiguity (initials, truncations).
    aliases = [base, f"{base[0]}.", base[:3], base.lower()][: n + 1]
    sessions = [Session(session_id="s0", input=f"{aliases[0]} owns Project {project}.")]
    for i, alias in enumerate(aliases[1:], start=1):
        sessions.append(
            Session(session_id=f"s{i}", input=f"{alias} is assigned to Task T{i}.")
        )
    return BenchmarkExample(
        id=f"stress-alias-{intensity}-{idx:03d}",
        category="stress_alias_ambiguity",
        intensity=intensity,
        sessions=sessions,
        questions=[
            Question(
                query=f"Who owns Project {project}?",
                expected_answer_contains=[base],
                expected_conflict=False,
            )
        ],
        # All alias surface forms should resolve to a single canonical person.
        gold_entity_groups={f"person::{base}": aliases},
    )


def _temporal_overlap(rng: random.Random, idx: int, intensity: str) -> BenchmarkExample:
    """Event orderings that introduce overlapping / cyclic PRECEDES edges."""
    n = _INTENSITY_COUNT[intensity]
    events = rng.sample(_EVENTS, k=min(n + 1, len(_EVENTS)))
    sessions: list[Session] = []
    for i in range(len(events) - 1):
        sessions.append(
            Session(
                session_id=f"s{i}",
                input=f"Event {events[i]} precedes Event {events[i + 1]}.",
            )
        )
    # Close a cycle back to the first event (the contradiction to surface).
    sessions.append(
        Session(
            session_id=f"s{len(events)}",
            input=f"Event {events[-1]} precedes Event {events[0]}.",
        )
    )
    return BenchmarkExample(
        id=f"stress-temporal-{intensity}-{idx:03d}",
        category="stress_temporal_overlap",
        intensity=intensity,
        sessions=sessions,
        questions=[
            Question(
                query=f"Does the ordering of {events[0]} and {events[-1]} form a cycle?",
                expected_answer_contains=[events[0]],
                expected_conflict=True,
            )
        ],
    )


def _confidence_inflation(rng: random.Random, idx: int, intensity: str) -> BenchmarkExample:
    """Final decisions asserted without supporting evidence (C8 should quarantine)."""
    n = _INTENSITY_COUNT[intensity]
    projects = rng.sample(_PROJECTS, k=min(n, len(_PROJECTS)))
    sessions = [
        Session(
            session_id=f"s{i}",
            input=f"We finalized the decision to launch Project {proj}.",
        )
        for i, proj in enumerate(projects)
    ]
    target = projects[0]
    return BenchmarkExample(
        id=f"stress-confidence-{intensity}-{idx:03d}",
        category="stress_confidence_inflation",
        intensity=intensity,
        sessions=sessions,
        questions=[
            Question(
                query=f"What was decided about Project {target}?",
                expected_answer_contains=["launch"],
                expected_conflict=True,  # unsupported final decision -> quarantined
            )
        ],
    )


_GENERATORS = {
    "conflicting_updates": _conflicting_updates,
    "alias_ambiguity": _alias_ambiguity,
    "temporal_overlap": _temporal_overlap,
    "confidence_inflation": _confidence_inflation,
}


def generate_stress_examples(
    seed: int = DEFAULT_SEED,
    intensities: tuple[str, ...] = INTENSITY_LEVELS,
    per_class: int = _PER_CLASS,
) -> list[BenchmarkExample]:
    """Generate the full seeded stress suite across classes and intensities."""
    rng = random.Random(seed)
    examples: list[BenchmarkExample] = []
    for intensity in intensities:
        for cls in PERTURBATION_CLASSES:
            gen = _GENERATORS[cls]
            for idx in range(per_class):
                examples.append(gen(rng, idx, intensity))
    return examples


# --------------------------------------------------------------------------- #
# Entity-resolution stress evaluation (paper Table VIII)
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    return " ".join(str(text).strip().casefold().split())


def evaluate_entity_resolution(
    examples: list[BenchmarkExample],
    settings_factory=None,
    *,
    extractor=None,
    embeddings=None,
) -> dict[str, float]:
    """Replay alias examples and score entity-resolution quality.

    For each alias example, all gold surface forms should resolve to **one**
    Person node. We measure, over all gold mention-pairs:

    * **F1** of the "same-entity" pair classification (did the system co-locate
      mentions that should be merged?),
    * **false-merge rate** — the share of distinct-entity pairs (across
      different gold groups in the same example) wrongly merged.

    Returns ``{"entity_resolution_f1", "false_merge_rate", "n_examples"}``.
    """
    from ocm.resolution.entity_resolver import normalize_name  # local import

    if settings_factory is None:
        def settings_factory() -> Settings:
            return Settings(
                deterministic_test_mode=True, chroma_mode="memory", extractor="mock"
            )

    tp = fp = fn = 0
    distinct_pairs = 0
    false_merges = 0
    n_examples = 0

    alias_examples = [e for e in examples if e.gold_entity_groups]
    for example in alias_examples:
        n_examples += 1
        container = CoreContainer(
            settings_factory(), extractor=extractor, embeddings=embeddings
        )
        for session in example.sessions:
            container.write_pipeline.run(
                session.input, f"{example.id}:{session.session_id}"
            )

        # Map each gold surface form to the Person node id it resolved to.
        graph = container.graph
        person_ids = [n for n in graph.node_ids() if graph.get_entity_type(n) == "Person"]
        # Build a lookup from normalized person label -> node id.
        label_to_node: dict[str, str] = {}
        for nid in person_ids:
            payload = graph.get_entity_payload(nid) or {}
            for key in ("name", "title"):
                val = payload.get(key)
                if val:
                    label_to_node.setdefault(_norm(val), nid)
            label_to_node.setdefault(_norm(nid), nid)

        def resolve(surface: str) -> Optional[str]:
            # The surface form's normalized name keys the resolver's node id.
            key = _norm(normalize_name(surface))
            if key in label_to_node:
                return label_to_node[key]
            # Fall back to a direct normalized-name match.
            return label_to_node.get(_norm(surface))

        groups = example.gold_entity_groups or {}
        # Same-entity pairs (within each gold group) -> should be merged.
        for surfaces in groups.values():
            resolved = [resolve(s) for s in surfaces]
            for i in range(len(resolved)):
                for j in range(i + 1, len(resolved)):
                    ri, rj = resolved[i], resolved[j]
                    if ri is not None and rj is not None and ri == rj:
                        tp += 1
                    else:
                        fn += 1
        # Distinct-entity pairs (across gold groups) -> should NOT be merged.
        group_list = list(groups.values())
        for gi in range(len(group_list)):
            for gj in range(gi + 1, len(group_list)):
                for si in group_list[gi]:
                    for sj in group_list[gj]:
                        distinct_pairs += 1
                        ri, rj = resolve(si), resolve(sj)
                        if ri is not None and rj is not None and ri == rj:
                            fp += 1
                            false_merges += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    false_merge_rate = (false_merges / distinct_pairs) if distinct_pairs else 0.0
    return {
        "entity_resolution_f1": f1,
        "entity_resolution_precision": precision,
        "entity_resolution_recall": recall,
        "false_merge_rate": false_merge_rate,
        "n_examples": float(n_examples),
    }
