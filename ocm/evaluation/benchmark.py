"""Seeded, reproducible evaluation Benchmark_Generator (Req 23).

`Benchmark_Generator` produces a JSONL dataset spanning the six reasoning
categories required by the spec. Each example is a multi-session memory
scenario followed by one or more questions whose expected answers, conflict
flag, and (optionally) supporting memory IDs let the evaluation harness score
baselines B0-B4 consistently.

Design contract (see design.md "Benchmark_Generator"):

* Each example has ``id``, ``category``, ``sessions`` (each ``{session_id,
  input}``), and ``questions`` (each ``{query, expected_answer_contains,
  expected_conflict, expected_supporting_ids?}``) (Req 23.1, 23.6).
* Six categories, at least 25 examples each and at least 150 total
  (Req 23.2, 23.3).
* Six hand-authored anchor examples covering the Task T1 conflict, the
  Joseph/Pharaoh case, project owner conflict, inactive assignee, final
  decision without evidence, and a temporal cycle (Req 23.4).
* A single seeded ``random.Random(seed)`` drives all sampling so the dataset is
  byte-identical across runs for a fixed seed (Req 23.5).

Session ``input`` strings are written so the deterministic ``Mock_Extractor``
recognizes them: ``"X owns Project Y"`` (OWNS), ``"X is assigned to Task Y"``
(ASSIGNED_TO), ``"X completed Task Y"`` (done + completion Event),
``"Task Y is not started"`` (status), ``"We decided to ..."`` (Decision), and
``http(s)://`` URLs (Document). This keeps the benchmark fully offline and
reproducible end to end.

Requirements: 23.1, 23.2, 23.3, 23.4, 23.5, 23.6.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

# --- Configuration ----------------------------------------------------------

#: Fixed default seed used by the configured benchmark build (Req 23.5).
DEFAULT_SEED: int = 1337

#: Number of programmatically generated examples per category. Six categories
#: at 25 each yields 150 generated examples; the six anchors push the total to
#: 156, satisfying "at least 25 per category and at least 150 total"
#: (Req 23.3).
PER_CATEGORY: int = 25

#: The six reasoning categories (Req 23.2), in a fixed order so generation is
#: deterministic.
CATEGORIES: tuple[str, ...] = (
    "longitudinal_factual_qa",
    "multi_step_planning_entity_consistency",
    "contradiction_heavy_update_stream",
    "temporal_reasoning_ordered_events",
    "entity_resolution_ambiguity",
    "evidence_required_decisions",
)

#: Mapping from the paper's four headline scenario classes (Tables II/IX) to the
#: implementation's six categories. The paper reports four scenario classes
#: (Recall, Contradiction-heavy, Temporal, Planning); the harness additionally
#: generates entity-resolution and evidence categories, which the paper reports
#: separately (entity resolution in Table VIII; evidence via the C8/governance
#: results). When reproducing the paper's per-scenario breakdown, group the
#: per-category metrics by this mapping.
#:
#: Scale note: the paper's protocol is 4 classes x 120 trajectories = 480. The
#: harness scale is governed by ``per_category`` (default :data:`PER_CATEGORY`),
#: so the paper's 120/class is reproduced with ``per_category=120`` over the four
#: headline categories; the default 25/class is a faster offline reference. The
#: per-class count and category set are configuration, not hard-coded results.
PAPER_SCENARIO_CLASSES: dict[str, str] = {
    "Recall": "longitudinal_factual_qa",
    "Contradiction-heavy": "contradiction_heavy_update_stream",
    "Temporal": "temporal_reasoning_ordered_events",
    "Planning": "multi_step_planning_entity_consistency",
}

# --- Fixed vocabularies (sampled with a seeded RNG) -------------------------

_NAMES: tuple[str, ...] = (
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi",
    "Ivan", "Judy", "Mallory", "Niaj", "Olivia", "Peggy", "Trent", "Victor",
    "Walter", "Yvonne", "Sybil", "Rupert",
)

_PROJECTS: tuple[str, ...] = (
    "Orion", "Apollo", "Hermes", "Atlas", "Titan", "Nova", "Vega", "Lyra",
    "Draco", "Phoenix", "Cygnus", "Pegasus", "Aquila", "Corvus",
)

_TASKS: tuple[str, ...] = (
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10",
    "T11", "T12", "T13", "T14", "T15", "T16",
)

#: (status phrase used in a sentence, canonical token expected in the answer).
_STATUS_PHRASES: tuple[tuple[str, str], ...] = (
    ("not started", "todo"),
    ("in progress", "in_progress"),
    ("blocked", "blocked"),
    ("cancelled", "cancelled"),
)

_DECISION_VERBS: tuple[str, ...] = ("launch", "pause", "rename", "expand")


# --- Data models (Req 23.1, 23.6) -------------------------------------------


class Session(BaseModel):
    """A single ingestion turn in a benchmark example.

    ``input`` is unstructured text fed to the write pipeline; the harness uses
    ``session_id`` (qualified by the example id) as the ``source_ref`` for
    provenance.
    """

    session_id: str
    input: str


class Question(BaseModel):
    """A question posed against the memory built from an example's sessions."""

    query: str
    expected_answer_contains: list[str]
    expected_conflict: bool
    #: Optional expected supporting memory facts for retrieval scoring
    #: (Req 23.6). Omitted from JSONL when absent.
    expected_supporting_ids: Optional[list[str]] = None


class BenchmarkExample(BaseModel):
    """One benchmark scenario: a category, ingestion sessions, and questions."""

    id: str
    category: str
    sessions: list[Session]
    questions: list[Question] = Field(default_factory=list)
    #: Optional stress-test perturbation intensity ("low"/"medium"/"high").
    intensity: Optional[str] = None
    #: Optional gold entity-resolution groups for alias stress scenarios:
    #: ``{"canonical_id": ["mention surface form", ...], ...}`` used to score
    #: entity-resolution F1 and false-merge rate (paper Table VIII).
    gold_entity_groups: Optional[dict[str, list[str]]] = None


# --- Generator --------------------------------------------------------------


class BenchmarkGenerator:
    """Produces the seeded, reproducible benchmark dataset (Req 23.5).

    A single ``random.Random(seed)`` drives every sample, and categories and
    indices are iterated in a fixed order, so :meth:`generate` returns an
    identical list across runs for a fixed seed.
    """

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self.seed = seed

    # -- public API ---------------------------------------------------------
    def generate(self, per_category: int = PER_CATEGORY) -> list[BenchmarkExample]:
        """Return all benchmark examples (generated + anchors), deterministic.

        Generated examples come first, grouped by category in ``CATEGORIES``
        order, followed by the six hand-authored anchors (Req 23.4).
        ``per_category`` controls how many examples are generated per category
        (defaults to :data:`PER_CATEGORY`); experiments use a smaller value for
        faster multi-seed sweeps.
        """
        rng = random.Random(self.seed)
        examples: list[BenchmarkExample] = []

        builders = {
            "longitudinal_factual_qa": self._gen_longitudinal,
            "multi_step_planning_entity_consistency": self._gen_planning,
            "contradiction_heavy_update_stream": self._gen_contradiction,
            "temporal_reasoning_ordered_events": self._gen_temporal,
            "entity_resolution_ambiguity": self._gen_entity_resolution,
            "evidence_required_decisions": self._gen_evidence,
        }

        for category in CATEGORIES:
            builder = builders[category]
            for i in range(per_category):
                examples.append(builder(rng, i))

        examples.extend(self._anchors())
        return examples

    # -- per-category generators -------------------------------------------
    def _gen_longitudinal(self, rng: random.Random, i: int) -> BenchmarkExample:
        """Longitudinal factual QA: facts established over turns, recalled later."""
        owner = rng.choice(_NAMES)
        project = rng.choice(_PROJECTS)
        assignee = rng.choice(_NAMES)
        task = rng.choice(_TASKS)
        return BenchmarkExample(
            id=f"longitudinal-{i:03d}",
            category="longitudinal_factual_qa",
            sessions=[
                Session(session_id="s1", input=f"{owner} owns Project {project}."),
                Session(session_id="s2", input=f"{assignee} is assigned to Task {task}."),
            ],
            questions=[
                Question(
                    query=f"Who owns Project {project}?",
                    expected_answer_contains=[owner],
                    expected_conflict=False,
                ),
                Question(
                    query=f"Who is assigned to Task {task}?",
                    expected_answer_contains=[assignee],
                    expected_conflict=False,
                ),
            ],
        )

    def _gen_planning(self, rng: random.Random, i: int) -> BenchmarkExample:
        """Multi-step planning requiring entity consistency across turns."""
        owner = rng.choice(_NAMES)
        project = rng.choice(_PROJECTS)
        assignee = rng.choice(_NAMES)
        task = rng.choice(_TASKS)
        phrase, status_token = rng.choice(_STATUS_PHRASES)
        return BenchmarkExample(
            id=f"planning-{i:03d}",
            category="multi_step_planning_entity_consistency",
            sessions=[
                Session(session_id="s1", input=f"{owner} owns Project {project}."),
                Session(session_id="s2", input=f"{assignee} is assigned to Task {task}."),
                Session(session_id="s3", input=f"Task {task} is {phrase}."),
            ],
            questions=[
                Question(
                    query=f"What is the status of Task {task}?",
                    expected_answer_contains=[status_token],
                    expected_conflict=False,
                ),
                Question(
                    query=f"Who is assigned to Task {task} on Project {project}?",
                    expected_answer_contains=[assignee],
                    expected_conflict=False,
                ),
            ],
        )

    def _gen_contradiction(self, rng: random.Random, i: int) -> BenchmarkExample:
        """Contradiction-heavy update stream: a later turn contradicts an earlier one."""
        person = rng.choice(_NAMES)
        task = rng.choice(_TASKS)
        return BenchmarkExample(
            id=f"contradiction-{i:03d}",
            category="contradiction_heavy_update_stream",
            sessions=[
                Session(
                    session_id="s1",
                    input=f"{person} is assigned to Task {task} and {person} completed Task {task}.",
                ),
                Session(
                    session_id="s2",
                    input=f"Actually, Task {task} has not been started yet.",
                ),
            ],
            questions=[
                Question(
                    query=f"What is the current status of Task {task}?",
                    expected_answer_contains=["done"],
                    expected_conflict=True,
                ),
            ],
        )

    def _gen_temporal(self, rng: random.Random, i: int) -> BenchmarkExample:
        """Temporal reasoning over ordered completion events."""
        person = rng.choice(_NAMES)
        first, second = rng.sample(_TASKS, 2)
        return BenchmarkExample(
            id=f"temporal-{i:03d}",
            category="temporal_reasoning_ordered_events",
            sessions=[
                Session(session_id="s1", input=f"{person} completed Task {first}."),
                Session(session_id="s2", input=f"{person} completed Task {second}."),
            ],
            questions=[
                Question(
                    query=f"Which task was completed first, Task {first} or Task {second}?",
                    expected_answer_contains=[first],
                    expected_conflict=False,
                ),
            ],
        )

    def _gen_entity_resolution(self, rng: random.Random, i: int) -> BenchmarkExample:
        """Entity resolution ambiguity: a shared name across distinct facts."""
        person = rng.choice(_NAMES)
        project = rng.choice(_PROJECTS)
        task = rng.choice(_TASKS)
        return BenchmarkExample(
            id=f"entity-resolution-{i:03d}",
            category="entity_resolution_ambiguity",
            sessions=[
                Session(session_id="s1", input=f"{person} owns Project {project}."),
                Session(session_id="s2", input=f"{person} is assigned to Task {task}."),
            ],
            questions=[
                Question(
                    query=f"Who owns Project {project}?",
                    expected_answer_contains=[person],
                    expected_conflict=False,
                ),
            ],
        )

    def _gen_evidence(self, rng: random.Random, i: int) -> BenchmarkExample:
        """Evidence-required decisions: a decision backed by a referenced document."""
        project = rng.choice(_PROJECTS)
        verb = rng.choice(_DECISION_VERBS)
        slug = project.lower()
        return BenchmarkExample(
            id=f"evidence-{i:03d}",
            category="evidence_required_decisions",
            sessions=[
                Session(session_id="s1", input=f"We decided to {verb} Project {project}."),
                Session(
                    session_id="s2",
                    input=f"See https://docs.example.com/{slug} for the rationale.",
                ),
            ],
            questions=[
                Question(
                    query=f"What was decided about Project {project}?",
                    expected_answer_contains=[verb],
                    expected_conflict=False,
                ),
            ],
        )

    # -- hand-authored anchors (Req 23.4) ----------------------------------
    def _anchors(self) -> list[BenchmarkExample]:
        """The six curated anchor examples, injected verbatim (Req 23.4)."""
        return [
            # 1. Task T1 conflict (done vs. not-started).
            BenchmarkExample(
                id="anchor-task-t1-conflict",
                category="contradiction_heavy_update_stream",
                sessions=[
                    Session(session_id="s1", input="Bob is assigned to Task T1 and Bob completed Task T1."),
                    Session(session_id="s2", input="Actually, Task T1 has not been started yet."),
                ],
                questions=[
                    Question(
                        query="What is the current status of Task T1?",
                        expected_answer_contains=["done"],
                        expected_conflict=True,
                        expected_supporting_ids=["ast_t1_done", "ast_t1_notstarted"],
                    ),
                ],
            ),
            # 2. Joseph / Pharaoh case (longitudinal multi-hop recall).
            BenchmarkExample(
                id="anchor-joseph-pharaoh",
                category="longitudinal_factual_qa",
                sessions=[
                    Session(session_id="s1", input="Joseph owns Project Pharaoh."),
                    Session(session_id="s2", input="Joseph is assigned to Task Dream."),
                    Session(session_id="s3", input="Joseph completed Task Dream."),
                ],
                questions=[
                    Question(
                        query="Who owns Project Pharaoh?",
                        expected_answer_contains=["Joseph"],
                        expected_conflict=False,
                        expected_supporting_ids=["ast_joseph_owns_pharaoh"],
                    ),
                    Question(
                        query="What is the status of Task Dream?",
                        expected_answer_contains=["done"],
                        expected_conflict=False,
                        expected_supporting_ids=["ast_dream_done"],
                    ),
                ],
            ),
            # 3. Project owner conflict (two owners asserted).
            BenchmarkExample(
                id="anchor-project-owner-conflict",
                category="contradiction_heavy_update_stream",
                sessions=[
                    Session(session_id="s1", input="Alice owns Project Orion."),
                    Session(session_id="s2", input="Actually, Carol owns Project Orion."),
                ],
                questions=[
                    Question(
                        query="Who owns Project Orion?",
                        expected_answer_contains=["Alice"],
                        expected_conflict=True,
                        expected_supporting_ids=["ast_alice_owns_orion", "ast_carol_owns_orion"],
                    ),
                ],
            ),
            # 4. Inactive assignee (assigning a task to an inactive person).
            BenchmarkExample(
                id="anchor-inactive-assignee",
                category="multi_step_planning_entity_consistency",
                sessions=[
                    Session(session_id="s1", input="Mallory is inactive."),
                    Session(session_id="s2", input="Mallory is assigned to Task T7."),
                ],
                questions=[
                    Question(
                        query="Who is assigned to Task T7?",
                        expected_answer_contains=["Mallory"],
                        expected_conflict=True,
                        expected_supporting_ids=["ast_t7_assigned_mallory"],
                    ),
                ],
            ),
            # 5. Final decision without supporting evidence.
            BenchmarkExample(
                id="anchor-final-decision-no-evidence",
                category="evidence_required_decisions",
                sessions=[
                    Session(session_id="s1", input="We finalized the decision to cancel Project Atlas."),
                ],
                questions=[
                    Question(
                        query="What was decided about Project Atlas?",
                        expected_answer_contains=["cancel"],
                        expected_conflict=True,
                        expected_supporting_ids=["dec_cancel_atlas"],
                    ),
                ],
            ),
            # 6. Temporal cycle (events ordered into a contradiction).
            BenchmarkExample(
                id="anchor-temporal-cycle",
                category="temporal_reasoning_ordered_events",
                sessions=[
                    Session(session_id="s1", input="Event Kickoff precedes Event Review."),
                    Session(session_id="s2", input="Event Review precedes Event Kickoff."),
                ],
                questions=[
                    Question(
                        query="Does the event ordering form a cycle between Kickoff and Review?",
                        expected_answer_contains=["cycle"],
                        expected_conflict=True,
                        expected_supporting_ids=["evt_kickoff", "evt_review"],
                    ),
                ],
            ),
        ]


# --- JSONL serialization ----------------------------------------------------


def write_jsonl(examples: list[BenchmarkExample], path: str | Path) -> None:
    """Write ``examples`` to ``path`` as JSONL, one example per line.

    ``None`` ``expected_supporting_ids`` fields are omitted so the output
    matches the design's schema (the field appears only where present). Output
    is byte-identical for identical input (Req 23.5).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for example in examples:
            payload = example.model_dump(exclude_none=True)
            fh.write(json.dumps(payload, ensure_ascii=False))
            fh.write("\n")


def generate_jsonl(path: str | Path, seed: int = DEFAULT_SEED) -> list[BenchmarkExample]:
    """Generate the benchmark for ``seed`` and write it to ``path`` as JSONL.

    Returns the generated examples so callers can use them in-memory too.
    """
    examples = BenchmarkGenerator(seed=seed).generate()
    write_jsonl(examples, path)
    return examples
