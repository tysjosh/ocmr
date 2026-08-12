"""The RAHGM evaluation corpus (paper §3.2).

1,500 candidate writes in 50 multi-session analytic scenarios of 30 temporally
ordered writes each, balanced across three write classes:

* 500 **routine** compatible updates (gold: ``accept``);
* 500 **authoritative corrections** or temporal supersessions (gold: ``supersede``);
* 500 **ambiguous or consequential conflicts** (gold: ``review``, with a
  malformed/prohibited subset whose gold is ``reject``).

Within each class, cases vary the eight axes of §3.2 — entity-alias ambiguity,
schema cardinality, source authority, temporal consistency, evidentiary support,
contradiction type, consequence, and reversibility. Poisoned or unsupported
evidence appears in 20% of scenarios so admission-time integrity is tested rather
than retrieval quality alone.

Ground truth is **derivable by construction**: every write is instantiated from a
template that allocates its *own* target entity and, where the template needs one,
its own incumbent assertion. Because no two writes contend for the same
``(subject, predicate)`` pair, the correct transition for each write follows
directly from the state the generator installed — a correction always has exactly
one recoverable incumbent to displace, and a routine write never displaces
anything. No model or annotator is needed to know the right answer, which is what
makes the routing metrics objective (Req 9.7).

Scenarios — not individual writes — are partitioned into training (25),
development (10), canary (5), and test (10). Entity ids are namespaced by
scenario, so no fact or alias can appear in more than one partition (Req 9.5).

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Iterator, Sequence

from ocm.governance.policy import Tier
from ocm.ontology.enums import WriteIntent

#: Corpus dimensions from §3.2.
N_SCENARIOS = 50
WRITES_PER_SCENARIO = 30
TOTAL_WRITES = N_SCENARIOS * WRITES_PER_SCENARIO  # 1500

#: Per-scenario write-class counts, giving the 500/500/500 global balance.
PER_CLASS_PER_SCENARIO = WRITES_PER_SCENARIO // 3  # 10

#: Fraction of scenarios that contain poisoned or unsupported evidence (§3.2).
POISONED_SCENARIO_FRACTION = 0.20

#: Partition sizes in scenarios (§3.2).
PARTITION_SIZES: dict[str, int] = {"train": 25, "dev": 10, "canary": 5, "test": 10}

#: Default corpus seed, matching the repository's benchmark convention.
DEFAULT_SEED = 1337

#: Corpus epoch; every write timestamp is derived from it deterministically.
EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


class WriteClass(str, Enum):
    """The three balanced write classes of §3.2."""

    routine = "routine"
    correction = "correction"
    conflict = "conflict"


class Partition(str, Enum):
    """Scenario partitions."""

    train = "train"
    dev = "dev"
    canary = "canary"
    test = "test"


#: The eight case-variation axes of §3.2 (Req 9.3).
PERTURBATION_AXES: tuple[str, ...] = (
    "entity_alias_ambiguity",
    "schema_cardinality",
    "source_authority",
    "temporal_consistency",
    "evidentiary_support",
    "contradiction_type",
    "consequence",
    "reversibility",
)

#: Source-authority pools, keyed to the governance rubric's schemes.
AUTHORITATIVE_SOURCES = ("system-of-record", "analyst", "verified")
ROUTINE_SOURCES = ("tool", "document", "observation")
WEAK_SOURCES = ("inferred", "unverified", "untrusted")

#: Authority values by scheme, mirroring :mod:`ocm.governance.features` so the
#: corpus and the router agree on what "authoritative" means.
AUTHORITY_BY_SCHEME: dict[str, float] = {
    "system-of-record": 0.98,
    "analyst": 0.95,
    "verified": 0.92,
    # Deliberately just below the 0.90 authoritative floor, so a case can carry
    # high authority without satisfying ``h(u)``. This is what lets the
    # discriminating-authority pair vary ``a`` while holding ``h`` fixed.
    "corroborated": 0.85,
    "tool": 0.75,
    "document": 0.70,
    "observation": 0.50,
    "inferred": 0.35,
    "unverified": 0.25,
    "untrusted": 0.10,
    "": 0.0,
}

#: The single-valued predicates a correction or cardinality conflict can displace.
SINGLE_VALUED_PREDICATES = ("ASSIGNED_TO", "HAS_STATUS", "HAS_VALUE")


# --------------------------------------------------------------------------- #
# Corpus records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SeedEntity:
    """One entity installed as incumbent state before a scenario's writes run."""

    entity_id: str
    entity_type: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class SeedAssertion:
    """One accepted incumbent assertion, representing prior durable memory."""

    assertion_id: str
    subject_id: str
    predicate: str
    object_id: str
    confidence: float
    source_ref: str
    created_at: datetime
    valid_from: datetime | None = None


@dataclass(frozen=True)
class CandidateWrite:
    """One candidate write ``u = (x, e, t, E, s, o)`` with its ground truth.

    Attributes:
        write_id: Globally unique, stable id.
        scenario_id: Owning scenario.
        index: Position in the scenario's temporal order (0-based).
        write_class: Which of the three balanced classes this case belongs to.
        template: The generating template, for per-family breakdowns.
        subject_id / predicate / object_id: The proposed assertion ``x``.
        confidence: The model's scalar confidence, consumed by the C3 baseline.
        source_ref: The source ``s``; its scheme drives the authority rubric.
        write_intent: The requested operation ``o``.
        valid_from / valid_to: The write's temporal extent ``t``.
        consequence / reversibility / authority: Preregistered rubric values
            ``q``, ``v``, ``a`` for this case.
        alias_ambiguous: Whether the subject alias fails to resolve uniquely.
        poisoned_evidence: Whether the cited evidence does not support the claim.
        gold_transition: The correct transition — the reference standard.
        consequential: Whether an incorrect transition here would be consequential.
        minimum_evidence: The least evidence that resolves the case (§3.2).
        perturbations: Which of :data:`PERTURBATION_AXES` this case exercises.
        creates_violation: Whether committing this write autonomously would leave
            invalid durable state.
        expected_object_after: The object that should be current for
            ``(subject, predicate)`` after a correct replay, or ``None`` when the
            write should not change the current value. Drives Experiment 4.
        incumbent_id: The assertion this write is constructed to displace, if any.
    """

    write_id: str
    scenario_id: str
    index: int
    write_class: WriteClass
    template: str
    subject_id: str
    predicate: str
    object_id: str
    confidence: float
    source_ref: str
    write_intent: WriteIntent
    valid_from: datetime | None
    valid_to: datetime | None
    consequence: float
    reversibility: float
    authority: float
    alias_ambiguous: bool
    poisoned_evidence: bool
    gold_transition: Tier
    consequential: bool
    minimum_evidence: str
    perturbations: tuple[str, ...]
    creates_violation: bool = False
    expected_object_after: str | None = None
    incumbent_id: str | None = None
    chain_id: str | None = None
    chain_position: int | None = None

    @property
    def contended(self) -> bool:
        """Whether this write shares its target with other writes in a chain."""
        return self.chain_id is not None

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "write_id": self.write_id,
            "scenario_id": self.scenario_id,
            "index": self.index,
            "write_class": self.write_class.value,
            "template": self.template,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "confidence": self.confidence,
            "source_ref": self.source_ref,
            "write_intent": self.write_intent.value,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "consequence": self.consequence,
            "reversibility": self.reversibility,
            "authority": self.authority,
            "alias_ambiguous": self.alias_ambiguous,
            "poisoned_evidence": self.poisoned_evidence,
            "gold_transition": self.gold_transition.value,
            "consequential": self.consequential,
            "minimum_evidence": self.minimum_evidence,
            "perturbations": list(self.perturbations),
            "creates_violation": self.creates_violation,
            "expected_object_after": self.expected_object_after,
            "incumbent_id": self.incumbent_id,
            "chain_id": self.chain_id,
            "chain_position": self.chain_position,
        }


@dataclass(frozen=True)
class ScenarioQuestion:
    """A downstream analytic question answered from the final memory state (§4.5).

    Attributes:
        query: The natural-language question, for reporting.
        subject_id / predicate: The assertion whose current object answers it.
        gold_object_id: The correct current object after a fully correct replay.
        stale_object_id: The object returned if a valid correction was never
            released — the stale-value-propagation signal.
        requires_evidence: Whether a correct answer requires supported memory.
    """

    query: str
    subject_id: str
    predicate: str
    gold_object_id: str
    stale_object_id: str | None = None
    requires_evidence: bool = False


@dataclass(frozen=True)
class Scenario:
    """One multi-session analytic scenario."""

    scenario_id: str
    index: int
    partition: Partition
    poisoned: bool
    entities: tuple[SeedEntity, ...]
    incumbents: tuple[SeedAssertion, ...]
    writes: tuple[CandidateWrite, ...]
    questions: tuple[ScenarioQuestion, ...]

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable view."""
        return {
            "scenario_id": self.scenario_id,
            "index": self.index,
            "partition": self.partition.value,
            "poisoned": self.poisoned,
            "n_entities": len(self.entities),
            "n_incumbents": len(self.incumbents),
            "writes": [w.as_dict() for w in self.writes],
            "questions": [
                {
                    "query": q.query,
                    "subject_id": q.subject_id,
                    "predicate": q.predicate,
                    "gold_object_id": q.gold_object_id,
                    "stale_object_id": q.stale_object_id,
                    "requires_evidence": q.requires_evidence,
                }
                for q in self.questions
            ],
        }


@dataclass(frozen=True)
class RahgmCorpus:
    """The full corpus with partition accessors."""

    scenarios: tuple[Scenario, ...]
    seed: int

    def __iter__(self) -> Iterator[Scenario]:
        return iter(self.scenarios)

    def __len__(self) -> int:
        return len(self.scenarios)

    def partition(self, name: Partition | str) -> tuple[Scenario, ...]:
        """Scenarios in one partition."""
        name = Partition(name)
        return tuple(s for s in self.scenarios if s.partition is name)

    @property
    def writes(self) -> tuple[CandidateWrite, ...]:
        """Every candidate write, in scenario then temporal order."""
        return tuple(w for s in self.scenarios for w in s.writes)

    def writes_in(self, name: Partition | str) -> tuple[CandidateWrite, ...]:
        """Every candidate write in one partition."""
        return tuple(w for s in self.partition(name) for w in s.writes)

    def class_counts(self) -> dict[str, int]:
        """Write counts per class, for the §3.2 balance check."""
        counts = {cls.value: 0 for cls in WriteClass}
        for write in self.writes:
            counts[write.write_class.value] += 1
        return counts

    def gold_counts(self) -> dict[str, int]:
        """Write counts per gold transition."""
        counts: dict[str, int] = {tier.value: 0 for tier in Tier}
        for write in self.writes:
            counts[write.gold_transition.value] += 1
        return counts

    def template_counts(self) -> dict[str, int]:
        """Write counts per template family."""
        counts: dict[str, int] = {}
        for write in self.writes:
            counts[write.template] = counts.get(write.template, 0) + 1
        return dict(sorted(counts.items()))

    def perturbation_coverage(self) -> dict[str, int]:
        """How many writes exercise each of the eight variation axes (Req 9.3)."""
        counts = {axis: 0 for axis in PERTURBATION_AXES}
        for write in self.writes:
            for axis in write.perturbations:
                if axis in counts:
                    counts[axis] += 1
        return counts

    def chain_counts(self) -> dict[str, int]:
        """Contention-chain statistics, for the order-sensitivity check (§3.1)."""
        chains = {w.chain_id for w in self.writes if w.chain_id}
        contended = [w for w in self.writes if w.contended]
        return {
            "n_chains": len(chains),
            "n_contended_writes": len(contended),
            "n_independent_writes": len(self.writes) - len(contended),
            "contended_fraction": (
                len(contended) / len(self.writes) if self.writes else 0.0
            ),
        }

    def summary(self) -> dict[str, Any]:
        """A compact, checkable description of the corpus."""
        poisoned = sum(1 for s in self.scenarios if s.poisoned)
        return {
            "chain_counts": self.chain_counts(),
            "seed": self.seed,
            "n_scenarios": len(self.scenarios),
            "n_writes": len(self.writes),
            "writes_per_scenario": (
                len(self.scenarios[0].writes) if self.scenarios else 0
            ),
            "class_counts": self.class_counts(),
            "gold_counts": self.gold_counts(),
            "template_counts": self.template_counts(),
            "poisoned_scenarios": poisoned,
            "poisoned_fraction": poisoned / len(self.scenarios) if self.scenarios else 0.0,
            "partition_sizes": {p.value: len(self.partition(p)) for p in Partition},
            "partition_write_counts": {
                p.value: len(self.writes_in(p)) for p in Partition
            },
            "perturbation_coverage": self.perturbation_coverage(),
        }


# --------------------------------------------------------------------------- #
# Entity pool
# --------------------------------------------------------------------------- #
class EntityPool:
    """Allocates scenario-namespaced entities and incumbent assertions on demand.

    Templates request exactly the entities they need, so each write gets its own
    target and its own incumbent. Nothing is shared between writes, which is what
    makes the gold transition a property of the template rather than of write
    ordering (Req 9.7). Ids are namespaced by scenario, guaranteeing partition
    disjointness (Req 9.5).
    """

    def __init__(self, scenario_id: str) -> None:
        self.scenario_id = scenario_id
        self.index = int(scenario_id[2:])
        self.base = EPOCH + timedelta(days=self.index * 7)
        self.entities: list[SeedEntity] = []
        self.incumbents: list[SeedAssertion] = []
        self._counters: dict[str, int] = {}
        self._install_status_values()

    # -- allocation --------------------------------------------------------
    def _next(self, kind: str) -> int:
        value = self._counters.get(kind, 0)
        self._counters[kind] = value + 1
        return value

    def _add(self, entity_type: str, prefix: str, payload: dict[str, Any]) -> str:
        index = self._next(prefix)
        entity_id = f"{self.scenario_id}-{prefix}-{index}"
        payload = {**payload, "id": entity_id}
        self.entities.append(SeedEntity(entity_id, entity_type, payload))
        return entity_id

    def _install_status_values(self) -> None:
        """StatusValue nodes are shared vocabulary, not scenario-specific state."""
        for status in (
            "todo",
            "in_progress",
            "blocked",
            "done",
            "cancelled",
            "active",
            "inactive",
            "completed",
            "draft",
            "final",
        ):
            sid = f"status:{status}"
            self.entities.append(
                SeedEntity(sid, "StatusValue", {"id": sid, "value": status, "name": status})
            )

    # -- typed allocators --------------------------------------------------
    def person(self, name: str | None = None, status: str = "active") -> str:
        index = self._counters.get("per", 0)
        return self._add(
            "Person",
            "per",
            {"name": name or f"{self.scenario_id} Person {index}", "status": status},
        )

    def organization(self) -> str:
        index = self._counters.get("org", 0)
        return self._add(
            "Organization",
            "org",
            {"name": f"{self.scenario_id} Org {index}", "type": "agency", "status": "active"},
        )

    def project(self) -> str:
        index = self._counters.get("prj", 0)
        return self._add(
            "Project",
            "prj",
            {"name": f"{self.scenario_id} Project {index}", "goal": "assess", "status": "active"},
        )

    def task(self, status: str = "in_progress") -> str:
        index = self._counters.get("tsk", 0)
        return self._add(
            "Task",
            "tsk",
            {"title": f"{self.scenario_id} Task {index}", "status": status, "priority": "medium"},
        )

    def document(self) -> str:
        index = self._counters.get("doc", 0)
        return self._add(
            "Document",
            "doc",
            {
                "title": f"{self.scenario_id} Document {index}",
                "path_or_url": f"https://example.invalid/{self.scenario_id}/doc{index}",
            },
        )

    def decision(self, status: str = "draft") -> str:
        index = self._counters.get("dec", 0)
        return self._add(
            "Decision",
            "dec",
            {"summary": f"{self.scenario_id} decision {index}", "status": status},
        )

    def event(self) -> str:
        index = self._counters.get("evt", 0)
        return self._add(
            "Event",
            "evt",
            {"type": "completion", "description": f"{self.scenario_id} event {index}"},
        )

    def slot(self) -> str:
        index = self._counters.get("slt", 0)
        return self._add("Slot", "slt", {"name": f"{self.scenario_id}-slot-{index}"})

    def slot_value(self, label: str) -> str:
        index = self._counters.get("val", 0)
        return self._add("SlotValue", "val", {"value": label, "name": label})

    # -- incumbents --------------------------------------------------------
    def incumbent(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        *,
        confidence: float = 0.88,
        source: str = "system-of-record",
        age_hours: int = 0,
    ) -> str:
        """Install one accepted incumbent assertion and return its id."""
        index = self._next("ast")
        assertion_id = f"{self.scenario_id}-ast-{index}"
        stamp = self.base + timedelta(hours=age_hours)
        self.incumbents.append(
            SeedAssertion(
                assertion_id=assertion_id,
                subject_id=subject_id,
                predicate=predicate,
                object_id=object_id,
                confidence=confidence,
                source_ref=f"{source}:{self.scenario_id}:baseline",
                created_at=stamp,
                valid_from=stamp,
            )
        )
        return assertion_id


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
class CorpusGenerator:
    """Deterministic generator for the §3.2 evaluation corpus.

    A single :class:`random.Random` seeded per scenario drives every choice, so a
    scenario's content depends only on ``(seed, scenario_index)``.
    """

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        """Create a generator."""
        self.seed = seed

    # -- public API --------------------------------------------------------
    def generate(
        self,
        *,
        n_scenarios: int = N_SCENARIOS,
        writes_per_scenario: int = WRITES_PER_SCENARIO,
    ) -> RahgmCorpus:
        """Generate the corpus.

        Args:
            n_scenarios: Number of scenarios (50 in the paper).
            writes_per_scenario: Writes per scenario (30 in the paper). Must be
                divisible by three so the class balance is exact.

        Returns:
            The :class:`RahgmCorpus`.
        """
        if writes_per_scenario % 3 != 0:
            raise ValueError("writes_per_scenario must be divisible by 3 for class balance")

        partitions = self._assign_partitions(n_scenarios)
        poisoned = self._assign_poisoned(n_scenarios)

        scenarios = tuple(
            self._scenario(
                index=index,
                partition=partitions[index],
                poisoned=poisoned[index],
                writes_per_scenario=writes_per_scenario,
            )
            for index in range(n_scenarios)
        )
        return RahgmCorpus(scenarios=scenarios, seed=self.seed)

    # -- partitioning ------------------------------------------------------
    @staticmethod
    def _assign_partitions(n_scenarios: int) -> list[Partition]:
        """Assign scenarios to partitions by contiguous index blocks (Req 9.5).

        Contiguous blocks keep the mapping trivially auditable, and because entity
        ids are namespaced by scenario there is no leakage a shuffle would prevent.
        """
        if n_scenarios == N_SCENARIOS:
            sizes = dict(PARTITION_SIZES)
        else:
            total = sum(PARTITION_SIZES.values())
            sizes = {
                name: max(1, round(count * n_scenarios / total))
                for name, count in PARTITION_SIZES.items()
            }
            sizes["train"] = max(1, sizes["train"] + (n_scenarios - sum(sizes.values())))

        out: list[Partition] = []
        for name in ("train", "dev", "canary", "test"):
            out.extend([Partition(name)] * sizes[name])
        out = out[:n_scenarios]
        while len(out) < n_scenarios:
            out.append(Partition.test)
        return out

    def _assign_poisoned(self, n_scenarios: int) -> list[bool]:
        """Mark exactly 20% of scenarios as containing poisoned evidence.

        Poisoned scenarios are spread evenly across the index range so every
        partition receives a proportional share.
        """
        n_poisoned = int(round(n_scenarios * POISONED_SCENARIO_FRACTION))
        if n_poisoned == 0:
            return [False] * n_scenarios
        stride = n_scenarios / n_poisoned
        marked = {int(round(i * stride)) % n_scenarios for i in range(n_poisoned)}
        candidate = n_scenarios - 1
        while len(marked) < n_poisoned and candidate >= 0:
            marked.add(candidate)
            candidate -= 1
        return [index in marked for index in range(n_scenarios)]

    # -- one scenario ------------------------------------------------------
    def _scenario(
        self,
        *,
        index: int,
        partition: Partition,
        poisoned: bool,
        writes_per_scenario: int,
    ) -> Scenario:
        """Build one scenario: entities, incumbents, writes, and questions."""
        scenario_id = f"sc{index:03d}"
        rng = random.Random((self.seed * 100003) + index)
        pool = EntityPool(scenario_id)
        builder = ScenarioBuilder(pool=pool, rng=rng, poisoned_scenario=poisoned)

        per_class = writes_per_scenario // 3
        units = builder.build_units(per_class, scenario_index=index)

        # Shuffle *units* rather than writes, so a contention chain keeps its
        # internal temporal order while its position in the scenario varies. Class
        # therefore does not correlate with temporal position, but a chain's
        # causal structure survives.
        rng.shuffle(units)
        drafted = [write for unit in units for write in unit]
        writes = tuple(
            _finalize(write, position, scenario_id, pool)
            for position, write in enumerate(drafted)
        )

        return Scenario(
            scenario_id=scenario_id,
            index=index,
            partition=partition,
            poisoned=poisoned,
            entities=tuple(pool.entities),
            incumbents=tuple(pool.incumbents),
            writes=writes,
            questions=build_questions(writes, pool),
        )


def _finalize(
    write: CandidateWrite, position: int, scenario_id: str, pool: EntityPool
) -> CandidateWrite:
    """Assign a write its final temporal slot, id, and timestamp.

    Timestamps are placed strictly after every incumbent so an authoritative
    correction is genuinely the newer fact; the intentionally *undated* conflict
    template keeps its ``None`` timestamp.
    """
    stamp = (
        None
        if write.valid_from is None
        else pool.base + timedelta(days=1, hours=position + 1)
    )
    return CandidateWrite(
        **{
            **write.__dict__,
            "index": position,
            "write_id": f"{scenario_id}-w{position:02d}",
            "valid_from": stamp,
        }
    )


# --------------------------------------------------------------------------- #
# Template families
# --------------------------------------------------------------------------- #
class ScenarioBuilder:
    """Builds the three write classes for one scenario.

    Each template allocates its own target entity — and, where the case requires
    one, its own incumbent assertion — so exactly one transition is correct for
    every write. The gold label is a property of the template, not a judgment call.
    """

    def __init__(
        self,
        *,
        pool: EntityPool,
        rng: random.Random,
        poisoned_scenario: bool,
    ) -> None:
        self.pool = pool
        self.rng = rng
        self.poisoned_scenario = poisoned_scenario
        self.scenario_id = pool.scenario_id
        self._counter = 0
        self._chains = 0

    # -- unit assembly -----------------------------------------------------
    def build_units(
        self, per_class: int, *, scenario_index: int
    ) -> list[list[CandidateWrite]]:
        """Build the scenario's writes grouped into order-sensitive units.

        A *unit* is either a single independent write or a **contention chain** — a
        run of writes on the same ``(subject, predicate)`` target whose correct
        transitions depend on the state the earlier writes left behind. Chains are
        what make replay order matter: a wrong transition at position 0 changes
        the decision the system faces at position 2, exactly as §3.1 describes.

        Each chain and each discriminating pair consumes a fixed class budget, so
        the 1:1:1 class balance of §3.2 is preserved exactly:

        * 2 chains, each 1 routine + 1 correction + 1 conflict;
        * 3 discriminating pairs, each 1 correction + 1 conflict;
        * the remainder as independent writes.
        """
        n_chains = 2 if per_class >= 5 else 0
        n_pairs = 3 if per_class >= 8 else 0

        units: list[list[CandidateWrite]] = []

        chain_builders = (
            self._chain_status_cascade,
            self._chain_slot_revision,
            self._chain_assignment_churn,
        )
        for i in range(n_chains):
            # Rotate chain types by scenario so all three appear across the corpus.
            builder = chain_builders[(scenario_index + i) % len(chain_builders)]
            units.append(builder())

        pair_builders = (
            self._pair_consequence,
            self._pair_authority,
            self._pair_reversibility,
        )
        for i in range(n_pairs):
            units.extend([w] for w in pair_builders[i]())

        remaining_routine = per_class - n_chains
        remaining_correction = per_class - n_chains - n_pairs
        remaining_conflict = per_class - n_chains - n_pairs

        units.extend([w] for w in self.routine_writes(remaining_routine))
        units.extend([w] for w in self.correction_writes(remaining_correction))

        # A poisoned scenario must actually contain a poisoned write. The weak
        # authority conflict is the template that carries unsupported evidence, so
        # it is drafted explicitly rather than left to the rotation, which might
        # not select it in a given scenario.
        if self.poisoned_scenario and remaining_conflict > 0:
            units.append([self._conflict_weak_authority()])
            remaining_conflict -= 1

        # The scenario index offsets the template rotation so that families which
        # fire only once or twice per scenario — the malformed/prohibited rejects —
        # are still all represented across the corpus.
        units.extend(
            [w]
            for w in self.conflict_writes(remaining_conflict, offset=scenario_index)
        )
        return units

    # -- contention chains -------------------------------------------------
    def _next_chain_id(self, kind: str) -> str:
        """A stable id for the next chain."""
        self._chains += 1
        return f"{self.scenario_id}-chain{self._chains}-{kind}"

    def _chain_status_cascade(self) -> list[CandidateWrite]:
        """A Task status chain whose middle write, if committed, corrupts the next.

        Gold transitions are the correct trajectory:

        0. ``todo -> in_progress`` — first status, accept.
        1. ``in_progress -> done`` with no completion Event — C4 fails, review.
        2. ``in_progress -> cancelled`` from an authoritative source — supersede,
           because ``in_progress -> cancelled`` is a permitted transition.

        The cascade is in position 2. On the correct trajectory the incumbent
        status is ``in_progress``, so cancelling is legal. A system that wrongly
        committed position 1 holds ``done`` instead — a terminal status — so its
        own C10 check now fails and it faces a materially harder decision. The
        error at position 1 propagates into the routing decision at position 2.
        """
        chain_id = self._next_chain_id("status")
        task = self.pool.task(status="todo")
        return [
            self._build(
                write_class=WriteClass.routine,
                template="chain_status_first",
                subject_id=task,
                predicate="HAS_STATUS",
                object_id="status:in_progress",
                confidence=self.rng.uniform(0.88, 0.96),
                source=self.rng.choice(ROUTINE_SOURCES),
                intent=WriteIntent.update,
                consequence=0.40,
                reversibility=0.85,
                gold=Tier.accept,
                minimum_evidence="none: first status for the task",
                perturbations=("temporal_consistency",),
                expected_object_after="status:in_progress",
                chain_id=chain_id,
                chain_position=0,
            ),
            self._build(
                write_class=WriteClass.conflict,
                template="chain_status_unsupported_done",
                subject_id=task,
                predicate="HAS_STATUS",
                object_id="status:done",
                confidence=self.rng.uniform(0.85, 0.95),
                source=self.rng.choice(ROUTINE_SOURCES),
                intent=WriteIntent.update,
                consequence=0.70,
                reversibility=0.50,
                gold=Tier.review,
                minimum_evidence="a completion Event related to the task by RESULTS_IN",
                perturbations=("evidentiary_support", "consequence"),
                creates_violation=True,
                chain_id=chain_id,
                chain_position=1,
            ),
            self._build(
                write_class=WriteClass.correction,
                template="chain_status_authoritative_cancel",
                subject_id=task,
                predicate="HAS_STATUS",
                object_id="status:cancelled",
                confidence=self.rng.uniform(0.92, 0.98),
                source="analyst",
                intent=WriteIntent.update,
                consequence=0.60,
                reversibility=0.90,
                gold=Tier.supersede,
                minimum_evidence="analyst cancellation record postdating the current status",
                perturbations=("temporal_consistency", "source_authority"),
                expected_object_after="status:cancelled",
                chain_id=chain_id,
                chain_position=2,
            ),
        ]

    def _chain_slot_revision(self) -> list[CandidateWrite]:
        """A Slot value chain whose final write must not be allowed to land.

        0. first value — accept.
        1. authoritative correction — supersede.
        2. weakly attributed contradiction — review.

        This is an **answer cascade**, the complement of the status chain's routing
        cascade. The reviewable write is last, so committing it corrupts the
        current value that the scenario's downstream question reads, even though
        nothing structurally invalid ever entered memory. Placing the reviewable
        write last is what makes the error persist: a later authoritative
        correction would otherwise overwrite it and hide the mistake.
        """
        chain_id = self._next_chain_id("slot")
        slot = self.pool.slot()
        first = self.pool.slot_value("initial")
        corrected = self.pool.slot_value("corrected")
        contested = self.pool.slot_value("contested")
        return [
            self._build(
                write_class=WriteClass.routine,
                template="chain_slot_first_value",
                subject_id=slot,
                predicate="HAS_VALUE",
                object_id=first,
                confidence=self.rng.uniform(0.86, 0.96),
                source=self.rng.choice(ROUTINE_SOURCES),
                intent=WriteIntent.new_fact,
                consequence=0.35,
                reversibility=0.90,
                gold=Tier.accept,
                minimum_evidence="none: first value for the slot",
                perturbations=("source_authority",),
                expected_object_after=first,
                chain_id=chain_id,
                chain_position=0,
            ),
            self._build(
                write_class=WriteClass.correction,
                template="chain_slot_authoritative_revision",
                subject_id=slot,
                predicate="HAS_VALUE",
                object_id=corrected,
                confidence=self.rng.uniform(0.92, 0.99),
                source=self.rng.choice(AUTHORITATIVE_SOURCES),
                intent=WriteIntent.correction,
                consequence=0.45,
                reversibility=0.90,
                gold=Tier.supersede,
                minimum_evidence="authoritative revision postdating the first value",
                perturbations=("schema_cardinality", "source_authority"),
                expected_object_after=corrected,
                chain_id=chain_id,
                chain_position=1,
            ),
            self._build(
                write_class=WriteClass.conflict,
                template="chain_slot_weak_contradiction",
                subject_id=slot,
                predicate="HAS_VALUE",
                object_id=contested,
                confidence=self.rng.uniform(0.82, 0.94),
                source=self.rng.choice(WEAK_SOURCES),
                intent=WriteIntent.correction,
                consequence=0.55,
                reversibility=0.90,
                gold=Tier.review,
                minimum_evidence="corroboration from a source of at least routine authority",
                perturbations=("source_authority", "contradiction_type"),
                poisoned=self.poisoned_scenario,
                chain_id=chain_id,
                chain_position=2,
            ),
        ]

    def _chain_assignment_churn(self) -> list[CandidateWrite]:
        """An assignment chain where an unresolved alias sits mid-sequence.

        0. first assignment — accept.
        1. alias-ambiguous reassignment — review.
        2. authoritative reassignment — supersede.

        Committing position 1 does not change the final value, but it does change
        *which* assertion position 2 must retire, so the incumbent the router sees
        differs from the one the gold trajectory assumes.
        """
        chain_id = self._next_chain_id("assign")
        task = self.pool.task()
        first = self.pool.person(name=f"{self.scenario_id} Dana Okafor")
        alias = self.pool.person(name=f"{self.scenario_id} D. Okafor")
        final = self.pool.person()
        return [
            self._build(
                write_class=WriteClass.routine,
                template="chain_assign_first",
                subject_id=task,
                predicate="ASSIGNED_TO",
                object_id=first,
                confidence=self.rng.uniform(0.87, 0.96),
                source=self.rng.choice(ROUTINE_SOURCES),
                intent=WriteIntent.new_fact,
                consequence=0.40,
                reversibility=0.90,
                gold=Tier.accept,
                minimum_evidence="none: first assignment for the task",
                perturbations=("source_authority",),
                expected_object_after=first,
                chain_id=chain_id,
                chain_position=0,
            ),
            self._build(
                write_class=WriteClass.conflict,
                template="chain_assign_alias",
                subject_id=task,
                predicate="ASSIGNED_TO",
                object_id=alias,
                confidence=self.rng.uniform(0.80, 0.93),
                source=self.rng.choice(ROUTINE_SOURCES),
                intent=WriteIntent.update,
                consequence=0.60,
                reversibility=0.90,
                gold=Tier.review,
                minimum_evidence="an identity resolution linking or separating the two aliases",
                perturbations=("entity_alias_ambiguity", "schema_cardinality"),
                alias_ambiguous=True,
                chain_id=chain_id,
                chain_position=1,
            ),
            self._build(
                write_class=WriteClass.correction,
                template="chain_assign_authoritative",
                subject_id=task,
                predicate="ASSIGNED_TO",
                object_id=final,
                confidence=self.rng.uniform(0.93, 0.99),
                source="system-of-record",
                intent=WriteIntent.update,
                consequence=0.60,
                reversibility=0.90,
                gold=Tier.supersede,
                minimum_evidence="system-of-record reassignment record",
                perturbations=("schema_cardinality", "source_authority"),
                expected_object_after=final,
                chain_id=chain_id,
                chain_position=2,
            ),
        ]

    # -- discriminating pairs ----------------------------------------------
    def _discriminating_base(
        self,
        *,
        template: str,
        source: str,
        consequence: float,
        reversibility: float,
        gold: Tier,
        write_class: WriteClass,
        minimum_evidence: str,
        perturbations: Sequence[str],
    ) -> CandidateWrite:
        """One half of a discriminating pair.

        Every discriminating case is a single-valued contradiction against a
        seeded incumbent, proposed with ``new_fact`` intent so ``h(u) = 0``. Holding
        the failure pattern, the intent, and the guards fixed means the *only*
        thing that can separate the pair's routes is the scalar the pair varies.
        Whether the router actually uses that scalar is what the Table 4 ablation
        then measures.
        """
        slot = self.pool.slot()
        old = self.pool.slot_value("incumbent")
        new = self.pool.slot_value("proposed")
        incumbent = self.pool.incumbent(
            slot, "HAS_VALUE", old, confidence=0.90, source="system-of-record"
        )
        return self._build(
            write_class=write_class,
            template=template,
            subject_id=slot,
            predicate="HAS_VALUE",
            object_id=new,
            confidence=self.rng.uniform(0.84, 0.94),
            source=source,
            intent=WriteIntent.new_fact,
            consequence=consequence,
            reversibility=reversibility,
            gold=gold,
            minimum_evidence=minimum_evidence,
            perturbations=perturbations,
            incumbent_id=incumbent,
            expected_object_after=new if gold is Tier.supersede else None,
        )

    def _pair_consequence(self) -> list[CandidateWrite]:
        """A pair differing **only** in consequence ``q``.

        Both cases are single-valued contradictions from a mid-authority source
        (``tool``, ``a = 0.75``) that are cheap to undo (``v = 0.90``), with
        ``h(u) = 0``. The low-consequence case should commit autonomously with an
        audit trail; the high-consequence case should escalate. This is the paper's
        core claim that reversibility and consequence modulate the confidence
        required for autonomous action, isolated so that nothing else can explain
        a routing difference.
        """
        return [
            self._discriminating_base(
                template="discriminating_consequence_low",
                source="tool",
                consequence=0.30,
                reversibility=0.90,
                gold=Tier.supersede,
                write_class=WriteClass.correction,
                minimum_evidence="none: low-consequence, cheaply reversible, attributed",
                perturbations=("consequence", "schema_cardinality"),
            ),
            self._discriminating_base(
                template="discriminating_consequence_high",
                source="tool",
                consequence=0.90,
                reversibility=0.90,
                gold=Tier.review,
                write_class=WriteClass.conflict,
                minimum_evidence="corroboration proportionate to the consequence of the claim",
                perturbations=("consequence", "schema_cardinality"),
            ),
        ]

    def _pair_authority(self) -> list[CandidateWrite]:
        """A pair differing **only** in source authority ``a``.

        Both sit below the 0.90 authoritative floor, so ``h(u) = 0`` for both and
        the ``h`` guard cannot separate them. Only the displayed authority discount
        ``β_a`` can. If the ablation shows authority is removable without effect,
        this pair is where that failure will be visible.
        """
        return [
            self._discriminating_base(
                template="discriminating_authority_high",
                source="corroborated",
                consequence=0.35,
                reversibility=0.90,
                gold=Tier.supersede,
                write_class=WriteClass.correction,
                minimum_evidence="none: corroborated source, low consequence, reversible",
                perturbations=("source_authority", "schema_cardinality"),
            ),
            self._discriminating_base(
                template="discriminating_authority_low",
                source="untrusted",
                consequence=0.35,
                reversibility=0.90,
                gold=Tier.review,
                write_class=WriteClass.conflict,
                minimum_evidence="corroboration from a source of at least routine authority",
                perturbations=("source_authority", "schema_cardinality"),
            ),
        ]

    def _pair_reversibility(self) -> list[CandidateWrite]:
        """A pair differing **only** in reversibility ``v``.

        Identical consequence and authority; one transition is cheap to undo and
        the other is not. This isolates the reversibility discount, which is the
        mechanism the paper claims lets a moderately uncertain but safe update
        commit while an equally uncertain irreversible one escalates.
        """
        return [
            self._discriminating_base(
                template="discriminating_reversibility_high",
                source="tool",
                consequence=0.55,
                reversibility=0.95,
                gold=Tier.supersede,
                write_class=WriteClass.correction,
                minimum_evidence="none: the transition is cheap to reverse if wrong",
                perturbations=("reversibility", "schema_cardinality"),
            ),
            self._discriminating_base(
                template="discriminating_reversibility_low",
                source="tool",
                consequence=0.55,
                reversibility=0.10,
                gold=Tier.review,
                write_class=WriteClass.conflict,
                minimum_evidence="authorization proportionate to an irreversible change",
                perturbations=("reversibility", "schema_cardinality"),
            ),
        ]

    # -- routine (gold: accept) -------------------------------------------
    def routine_writes(self, count: int) -> list[CandidateWrite]:
        """Compatible updates that satisfy every relevant constraint.

        No incumbent is displaced (each uses a many-to-many predicate or a fresh
        single-valued subject), evidence is present, the source is at least routine
        authority, and the write is timestamped. The correct transition is
        ``accept``.
        """
        templates: tuple[Callable[[], CandidateWrite], ...] = (
            self._routine_membership,
            self._routine_evidence_link,
            self._routine_participation,
            self._routine_about,
            self._routine_first_status,
        )
        return [templates[i % len(templates)]() for i in range(count)]

    def _routine_membership(self) -> CandidateWrite:
        return self._build(
            write_class=WriteClass.routine,
            template="routine_membership",
            subject_id=self.pool.person(),
            predicate="MEMBER_OF",
            object_id=self.pool.organization(),
            confidence=self.rng.uniform(0.86, 0.97),
            source=self.rng.choice(ROUTINE_SOURCES),
            intent=WriteIntent.new_fact,
            consequence=0.35,
            reversibility=0.90,
            gold=Tier.accept,
            minimum_evidence="none: compatible new membership with an attributed source",
            perturbations=("source_authority",),
        )

    def _routine_evidence_link(self) -> CandidateWrite:
        return self._build(
            write_class=WriteClass.routine,
            template="routine_evidence_link",
            subject_id=self.pool.document(),
            predicate="EVIDENCE_FOR",
            object_id=self.pool.decision(),
            confidence=self.rng.uniform(0.88, 0.98),
            source="document",
            intent=WriteIntent.new_fact,
            consequence=0.30,
            reversibility=0.90,
            gold=Tier.accept,
            minimum_evidence="none: additive evidence link",
            perturbations=("evidentiary_support",),
        )

    def _routine_participation(self) -> CandidateWrite:
        return self._build(
            write_class=WriteClass.routine,
            template="routine_participation",
            subject_id=self.pool.person(),
            predicate="PARTICIPATES_IN",
            object_id=self.pool.event(),
            confidence=self.rng.uniform(0.84, 0.95),
            source=self.rng.choice(ROUTINE_SOURCES),
            intent=WriteIntent.new_fact,
            consequence=0.30,
            reversibility=0.90,
            gold=Tier.accept,
            minimum_evidence="none: compatible participation record",
            perturbations=("source_authority",),
        )

    def _routine_about(self) -> CandidateWrite:
        return self._build(
            write_class=WriteClass.routine,
            template="routine_about",
            subject_id=self.pool.document(),
            predicate="ABOUT",
            object_id=self.pool.project(),
            confidence=self.rng.uniform(0.85, 0.96),
            source="document",
            intent=WriteIntent.new_fact,
            consequence=0.20,
            reversibility=0.95,
            gold=Tier.accept,
            minimum_evidence="none: low-consequence topical link",
            perturbations=("consequence",),
        )

    def _routine_first_status(self) -> CandidateWrite:
        """A legal first status on a task with no incumbent HAS_STATUS assertion."""
        return self._build(
            write_class=WriteClass.routine,
            template="routine_first_status",
            subject_id=self.pool.task(status="todo"),
            predicate="HAS_STATUS",
            object_id="status:in_progress",
            confidence=self.rng.uniform(0.88, 0.96),
            source=self.rng.choice(ROUTINE_SOURCES),
            intent=WriteIntent.update,
            consequence=0.40,
            reversibility=0.85,
            gold=Tier.accept,
            minimum_evidence="none: first status for the task",
            perturbations=("temporal_consistency",),
            expected_object_after="status:in_progress",
            question=True,
        )

    # -- correction (gold: supersede) --------------------------------------
    def correction_writes(self, count: int) -> list[CandidateWrite]:
        """Authoritative corrections and temporal supersessions.

        Each displaces exactly one recoverable incumbent on a single-valued
        predicate, carries authority ``≥ 0.90``, is timestamped after that
        incumbent, and cites evidence — the full ``h(u)`` condition — so the
        correct transition is ``supersede``.
        """
        templates: tuple[Callable[[], CandidateWrite], ...] = (
            self._correction_slot_value,
            self._correction_assignment,
            self._correction_project_status,
            self._correction_person_status,
        )
        return [templates[i % len(templates)]() for i in range(count)]

    def _correction_slot_value(self) -> CandidateWrite:
        slot = self.pool.slot()
        old = self.pool.slot_value("stale")
        new = self.pool.slot_value("corrected")
        incumbent = self.pool.incumbent(
            slot, "HAS_VALUE", old, confidence=0.85, source="observation"
        )
        return self._build(
            write_class=WriteClass.correction,
            template="correction_slot_value",
            subject_id=slot,
            predicate="HAS_VALUE",
            object_id=new,
            confidence=self.rng.uniform(0.92, 0.99),
            source=self.rng.choice(AUTHORITATIVE_SOURCES),
            intent=WriteIntent.correction,
            consequence=0.45,
            reversibility=0.90,
            gold=Tier.supersede,
            minimum_evidence="authoritative source plus a timestamp after the incumbent",
            perturbations=("schema_cardinality", "source_authority", "temporal_consistency"),
            expected_object_after=new,
            incumbent_id=incumbent,
            question=True,
        )

    def _correction_assignment(self) -> CandidateWrite:
        task = self.pool.task()
        old = self.pool.person()
        new = self.pool.person()
        incumbent = self.pool.incumbent(
            task, "ASSIGNED_TO", old, confidence=0.86, source="tool"
        )
        return self._build(
            write_class=WriteClass.correction,
            template="correction_assignment",
            subject_id=task,
            predicate="ASSIGNED_TO",
            object_id=new,
            confidence=self.rng.uniform(0.93, 0.99),
            source="system-of-record",
            intent=WriteIntent.update,
            consequence=0.60,
            reversibility=0.90,
            gold=Tier.supersede,
            minimum_evidence="system-of-record reassignment record",
            perturbations=("schema_cardinality", "source_authority"),
            expected_object_after=new,
            incumbent_id=incumbent,
            question=True,
        )

    def _correction_project_status(self) -> CandidateWrite:
        project = self.pool.project()
        incumbent = self.pool.incumbent(
            project, "HAS_STATUS", "status:active", confidence=0.88, source="tool"
        )
        return self._build(
            write_class=WriteClass.correction,
            template="correction_project_status",
            subject_id=project,
            predicate="HAS_STATUS",
            object_id="status:completed",
            confidence=self.rng.uniform(0.92, 0.98),
            source="analyst",
            intent=WriteIntent.update,
            consequence=0.65,
            reversibility=0.90,
            gold=Tier.supersede,
            minimum_evidence="analyst-verified project closure record",
            perturbations=("consequence", "source_authority", "temporal_consistency"),
            expected_object_after="status:completed",
            incumbent_id=incumbent,
            question=True,
        )

    def _correction_person_status(self) -> CandidateWrite:
        person = self.pool.person()
        incumbent = self.pool.incumbent(
            person, "HAS_STATUS", "status:active", confidence=0.84, source="observation"
        )
        return self._build(
            write_class=WriteClass.correction,
            template="correction_person_status",
            subject_id=person,
            predicate="HAS_STATUS",
            object_id="status:inactive",
            confidence=self.rng.uniform(0.91, 0.98),
            source="verified",
            intent=WriteIntent.correction,
            consequence=0.50,
            reversibility=0.90,
            gold=Tier.supersede,
            minimum_evidence="verified departure record postdating the incumbent",
            perturbations=("schema_cardinality", "temporal_consistency"),
            expected_object_after="status:inactive",
            incumbent_id=incumbent,
            question=True,
        )

    # -- conflict (gold: review, or reject for the malformed subset) -------
    def conflict_writes(self, count: int, *, offset: int = 0) -> list[CandidateWrite]:
        """Ambiguous, consequential, weakly supported, or costly-to-undo cases.

        Two thirds are genuine ambiguity or consequence a human must adjudicate
        (``review``); one third is malformed or prohibited and must be rejected
        outright without consuming review capacity (``reject``). The two families
        advance on independent counters so every template is exercised, and
        ``offset`` continues the rotation across scenarios so a family that fires
        only once per scenario still covers all of its templates corpus-wide.
        """
        review_templates: tuple[Callable[[], CandidateWrite], ...] = (
            self._conflict_alias_ambiguity,
            self._conflict_weak_authority,
            self._conflict_terminal_status_flip,
            self._conflict_unsupported_final_decision,
            self._conflict_undated_update,
            self._conflict_irreversible_deletion,
        )
        reject_templates: tuple[Callable[[], CandidateWrite], ...] = (
            self._reject_unregistered_predicate,
            self._reject_unattributed,
            self._reject_domain_range,
        )
        out: list[CandidateWrite] = []
        review_index = offset
        reject_index = offset
        for i in range(count):
            if i % 3 == 2:
                out.append(reject_templates[reject_index % len(reject_templates)]())
                reject_index += 1
            else:
                out.append(review_templates[review_index % len(review_templates)]())
                review_index += 1
        return out

    def _conflict_alias_ambiguity(self) -> CandidateWrite:
        """The dominant OCMR false-quarantine cause: an unresolved entity alias.

        Two person records may or may not be the same individual. The write is
        otherwise well formed, so OCMR alone can only hold it indefinitely; RAHGM
        should route it to a reviewer who resolves the identity and releases it.
        """
        task = self.pool.task()
        incumbent_person = self.pool.person(name=f"{self.scenario_id} Alice Moreau")
        alias_person = self.pool.person(name=f"{self.scenario_id} A. Moreau")
        incumbent = self.pool.incumbent(
            task, "ASSIGNED_TO", incumbent_person, confidence=0.87, source="tool"
        )
        return self._build(
            write_class=WriteClass.conflict,
            template="conflict_alias_ambiguity",
            subject_id=task,
            predicate="ASSIGNED_TO",
            object_id=alias_person,
            confidence=self.rng.uniform(0.80, 0.93),
            source=self.rng.choice(ROUTINE_SOURCES),
            intent=WriteIntent.update,
            consequence=0.60,
            reversibility=0.90,
            gold=Tier.review,
            minimum_evidence="an identity resolution linking or separating the two person aliases",
            perturbations=("entity_alias_ambiguity", "schema_cardinality"),
            alias_ambiguous=True,
            incumbent_id=incumbent,
        )

    def _conflict_weak_authority(self) -> CandidateWrite:
        """A contradiction proposed by a weakly attributed source."""
        slot = self.pool.slot()
        old = self.pool.slot_value("established")
        new = self.pool.slot_value("rumoured")
        incumbent = self.pool.incumbent(
            slot, "HAS_VALUE", old, confidence=0.90, source="system-of-record"
        )
        return self._build(
            write_class=WriteClass.conflict,
            template="conflict_weak_authority",
            subject_id=slot,
            predicate="HAS_VALUE",
            object_id=new,
            confidence=self.rng.uniform(0.82, 0.94),
            source=self.rng.choice(WEAK_SOURCES),
            intent=WriteIntent.correction,
            consequence=0.55,
            reversibility=0.90,
            gold=Tier.review,
            minimum_evidence="corroboration from a source of at least routine authority",
            perturbations=("source_authority", "contradiction_type", "schema_cardinality"),
            poisoned=self.poisoned_scenario,
            incumbent_id=incumbent,
        )

    def _conflict_terminal_status_flip(self) -> CandidateWrite:
        """Reopening a completed task — high consequence, illegal transition."""
        task = self.pool.task(status="done")
        event = self.pool.event()
        # RESULTS_IN runs Event -> Task (the completion event results in the task
        # being done), which is also the direction C4 checks as an in-edge.
        self.pool.incumbent(event, "RESULTS_IN", task, confidence=0.90, source="document")
        incumbent = self.pool.incumbent(
            task, "HAS_STATUS", "status:done", confidence=0.95, source="system-of-record"
        )
        return self._build(
            write_class=WriteClass.conflict,
            template="conflict_terminal_status_flip",
            subject_id=task,
            predicate="HAS_STATUS",
            object_id="status:todo",
            confidence=self.rng.uniform(0.84, 0.95),
            source=self.rng.choice(ROUTINE_SOURCES),
            intent=WriteIntent.new_fact,
            consequence=0.85,
            reversibility=0.55,
            gold=Tier.review,
            minimum_evidence="an authoritative reopening record superseding the completion event",
            perturbations=("temporal_consistency", "consequence", "contradiction_type"),
            creates_violation=True,
            incumbent_id=incumbent,
        )

    def _conflict_unsupported_final_decision(self) -> CandidateWrite:
        """Finalizing a decision without the evidence floor — the C8 condition."""
        decision = self.pool.decision(status="draft")
        incumbent = self.pool.incumbent(
            decision, "HAS_STATUS", "status:draft", confidence=0.90, source="analyst"
        )
        return self._build(
            write_class=WriteClass.conflict,
            template="conflict_unsupported_final_decision",
            subject_id=decision,
            predicate="HAS_STATUS",
            object_id="status:final",
            confidence=self.rng.uniform(0.86, 0.96),
            source=self.rng.choice(ROUTINE_SOURCES),
            intent=WriteIntent.update,
            consequence=0.92,
            reversibility=0.30,
            gold=Tier.review,
            minimum_evidence="at least one EVIDENCE_FOR document supporting the decision",
            perturbations=("evidentiary_support", "consequence", "reversibility"),
            creates_violation=True,
            incumbent_id=incumbent,
        )

    def _conflict_undated_update(self) -> CandidateWrite:
        """An undated update to dated memory: which value is current is unresolved."""
        slot = self.pool.slot()
        old = self.pool.slot_value("dated")
        new = self.pool.slot_value("undated")
        incumbent = self.pool.incumbent(
            slot, "HAS_VALUE", old, confidence=0.86, source="document"
        )
        return self._build(
            write_class=WriteClass.conflict,
            template="conflict_undated_update",
            subject_id=slot,
            predicate="HAS_VALUE",
            object_id=new,
            confidence=self.rng.uniform(0.83, 0.94),
            source=self.rng.choice(AUTHORITATIVE_SOURCES),
            intent=WriteIntent.update,
            consequence=0.50,
            reversibility=0.90,
            gold=Tier.review,
            minimum_evidence="a timestamp ordering the proposal against the incumbent",
            perturbations=("temporal_consistency", "schema_cardinality"),
            undated=True,
            incumbent_id=incumbent,
        )

    def _conflict_irreversible_deletion(self) -> CandidateWrite:
        """A deletion-intent write: uncertain and irreversible, so it escalates."""
        person = self.pool.person()
        project = self.pool.project()
        incumbent = self.pool.incumbent(
            person, "OWNS", project, confidence=0.91, source="system-of-record"
        )
        return self._build(
            write_class=WriteClass.conflict,
            template="conflict_irreversible_deletion",
            subject_id=person,
            predicate="OWNS",
            object_id=project,
            confidence=self.rng.uniform(0.80, 0.92),
            source=self.rng.choice(ROUTINE_SOURCES),
            intent=WriteIntent.deletion,
            consequence=0.80,
            reversibility=0.15,
            gold=Tier.review,
            minimum_evidence="an authoritative retraction of the ownership assertion",
            perturbations=("reversibility", "consequence"),
            incumbent_id=incumbent,
        )

    def _reject_unregistered_predicate(self) -> CandidateWrite:
        """W5: the predicate is not in the relation registry."""
        return self._build(
            write_class=WriteClass.conflict,
            template="reject_unregistered_predicate",
            subject_id=self.pool.person(),
            predicate="EMPLOYS_SECRETLY",
            object_id=self.pool.person(),
            confidence=self.rng.uniform(0.70, 0.95),
            source=self.rng.choice(ROUTINE_SOURCES),
            intent=WriteIntent.new_fact,
            consequence=0.50,
            reversibility=0.90,
            gold=Tier.reject,
            minimum_evidence="none: the predicate is not part of the ontology",
            perturbations=("schema_cardinality",),
        )

    def _reject_unattributed(self) -> CandidateWrite:
        """An unattributed write: ``g(u) = 1`` by the blank-source rule."""
        return self._build(
            write_class=WriteClass.conflict,
            template="reject_unattributed",
            subject_id=self.pool.person(),
            predicate="MEMBER_OF",
            object_id=self.pool.organization(),
            confidence=self.rng.uniform(0.75, 0.95),
            source=None,
            intent=WriteIntent.new_fact,
            consequence=0.45,
            reversibility=0.90,
            gold=Tier.reject,
            minimum_evidence="none: an unattributed write cannot be adjudicated",
            perturbations=("source_authority", "evidentiary_support"),
        )

    def _reject_domain_range(self) -> CandidateWrite:
        """C9: the subject/object types violate the relation signature."""
        return self._build(
            write_class=WriteClass.conflict,
            template="reject_domain_range",
            subject_id=self.pool.document(),
            predicate="ASSIGNED_TO",
            object_id=self.pool.event(),
            confidence=self.rng.uniform(0.70, 0.93),
            source=self.rng.choice(ROUTINE_SOURCES),
            intent=WriteIntent.new_fact,
            consequence=0.50,
            reversibility=0.90,
            gold=Tier.reject,
            minimum_evidence="none: the assertion is ontology-illegal",
            perturbations=("schema_cardinality",),
        )

    # -- assembly ----------------------------------------------------------
    def _build(
        self,
        *,
        write_class: WriteClass,
        template: str,
        subject_id: str,
        predicate: str,
        object_id: str,
        confidence: float,
        source: str | None,
        intent: WriteIntent,
        consequence: float,
        reversibility: float,
        gold: Tier,
        minimum_evidence: str,
        perturbations: Sequence[str],
        alias_ambiguous: bool = False,
        poisoned: bool = False,
        undated: bool = False,
        creates_violation: bool = False,
        expected_object_after: str | None = None,
        incumbent_id: str | None = None,
        question: bool = False,
        chain_id: str | None = None,
        chain_position: int | None = None,
    ) -> CandidateWrite:
        """Instantiate one write and derive its ground truth."""
        index = self._counter
        self._counter += 1
        source_ref = "" if source is None else f"{source}:{self.scenario_id}:w{index:02d}"
        authority = AUTHORITY_BY_SCHEME.get(source or "", 0.0)
        if poisoned:
            authority = min(authority, 0.05)

        # A placeholder timestamp; ``_finalize`` reassigns it to the write's slot.
        stamp = None if undated else self.pool.base + timedelta(days=1, hours=index + 1)

        # Consequential: a high-consequence assertion, or one that is costly or
        # impossible to undo (Req 9.6).
        consequential = consequence >= 0.60 or reversibility <= 0.30

        axes = set(perturbations)
        if alias_ambiguous:
            axes.add("entity_alias_ambiguity")
        if poisoned:
            axes.add("evidentiary_support")
        if undated:
            axes.add("temporal_consistency")
        if consequence >= 0.70:
            axes.add("consequence")
        if reversibility <= 0.50:
            axes.add("reversibility")
        if authority >= 0.90 or authority <= 0.35:
            axes.add("source_authority")
        if predicate in SINGLE_VALUED_PREDICATES and (
            incumbent_id or (chain_position is not None and chain_position > 0)
        ):
            axes.add("schema_cardinality")

        # ``question`` marks a write whose gold transition changes a single-valued
        # current value; :func:`build_questions` reads the flag back off the
        # ``expected_object_after`` field, so nothing further is recorded here.
        _ = question

        return CandidateWrite(
            write_id=f"{self.scenario_id}-w{index:02d}",
            scenario_id=self.scenario_id,
            index=index,
            write_class=write_class,
            template=template,
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            confidence=round(confidence, 4),
            source_ref=source_ref,
            write_intent=intent,
            valid_from=stamp,
            valid_to=None,
            consequence=consequence,
            reversibility=reversibility,
            authority=authority,
            alias_ambiguous=alias_ambiguous,
            poisoned_evidence=poisoned,
            gold_transition=gold,
            consequential=consequential,
            minimum_evidence=minimum_evidence,
            perturbations=tuple(sorted(axes)),
            creates_violation=creates_violation,
            expected_object_after=expected_object_after,
            incumbent_id=incumbent_id,
            chain_id=chain_id,
            chain_position=chain_position,
        )


# --------------------------------------------------------------------------- #
# Downstream questions (§4.5)
# --------------------------------------------------------------------------- #
#: Question phrasing by predicate.
_QUESTION_LABEL = {
    "HAS_VALUE": "What is the current value of",
    "ASSIGNED_TO": "Who is currently assigned to",
    "HAS_STATUS": "What is the current status of",
}


def build_questions(
    writes: Sequence[CandidateWrite], pool: EntityPool
) -> tuple[ScenarioQuestion, ...]:
    """Build downstream questions answered from the final memory state (§4.5).

    A question is created for every write whose gold transition *changes* the
    current value of a single-valued target. The gold answer is that new value; the
    stale answer is the incumbent it replaced. A system that holds a valid
    correction forever returns the stale value — exactly what the
    stale-propagation metric measures — while a system that wrongly commits an
    escalation-worthy write returns something that is neither.
    """
    incumbent_by_target: dict[tuple[str, str], str] = {
        (a.subject_id, a.predicate): a.object_id for a in pool.incumbents
    }

    # A contended target is written several times, so only the *last* write whose
    # gold transition changes the current value defines the question. The value it
    # displaces — the previous gold value, or the seeded incumbent — is the stale
    # answer a system returns when it fails to release that final correction.
    final: dict[tuple[str, str], CandidateWrite] = {}
    previous: dict[tuple[str, str], str | None] = {}
    for write in writes:
        if write.expected_object_after is None:
            continue
        if write.gold_transition not in (Tier.accept, Tier.supersede):
            continue
        key = (write.subject_id, write.predicate)
        prior = final.get(key)
        previous[key] = (
            prior.expected_object_after if prior else incumbent_by_target.get(key)
        )
        final[key] = write

    out: list[ScenarioQuestion] = []
    for (subject_id, predicate), write in final.items():
        stale = previous.get((subject_id, predicate))
        out.append(
            ScenarioQuestion(
                query=f"{_QUESTION_LABEL.get(predicate, 'What is')} {subject_id}?",
                subject_id=subject_id,
                predicate=predicate,
                gold_object_id=write.expected_object_after,
                stale_object_id=stale if stale != write.expected_object_after else None,
                requires_evidence=predicate == "HAS_STATUS",
            )
        )
    return tuple(out)


def generate_corpus(
    seed: int = DEFAULT_SEED,
    *,
    n_scenarios: int = N_SCENARIOS,
    writes_per_scenario: int = WRITES_PER_SCENARIO,
) -> RahgmCorpus:
    """Generate the RAHGM evaluation corpus (convenience wrapper)."""
    return CorpusGenerator(seed=seed).generate(
        n_scenarios=n_scenarios, writes_per_scenario=writes_per_scenario
    )
