"""RAHGM evaluation corpus: dimensions, balance, partitions, and ground truth.

Covers Req 9.x. The corpus is the reference standard for every routing metric, so
these tests assert the §3.2 construction exactly rather than approximately.
"""

from __future__ import annotations

import pytest

from ocm.evaluation.rahgm.corpus import (
    DEFAULT_SEED,
    N_SCENARIOS,
    PARTITION_SIZES,
    PERTURBATION_AXES,
    POISONED_SCENARIO_FRACTION,
    SINGLE_VALUED_PREDICATES,
    TOTAL_WRITES,
    WRITES_PER_SCENARIO,
    Partition,
    WriteClass,
    generate_corpus,
)
from ocm.governance.policy import Tier


@pytest.fixture(scope="module")
def corpus():
    """The full paper-scale corpus, generated once."""
    return generate_corpus()


# --------------------------------------------------------------------------- #
# Dimensions and balance (Req 9.1, 9.2)
# --------------------------------------------------------------------------- #
def test_corpus_has_1500_writes_in_50_scenarios_of_30(corpus):
    """1,500 candidate writes, 50 scenarios, 30 writes each (Req 9.1)."""
    assert len(corpus) == N_SCENARIOS == 50
    assert len(corpus.writes) == TOTAL_WRITES == 1500
    assert all(len(s.writes) == WRITES_PER_SCENARIO == 30 for s in corpus)


def test_write_classes_are_balanced_500_each(corpus):
    """500 routine, 500 correction, 500 conflict (Req 9.2)."""
    counts = corpus.class_counts()
    assert counts == {
        WriteClass.routine.value: 500,
        WriteClass.correction.value: 500,
        WriteClass.conflict.value: 500,
    }


def test_each_scenario_is_class_balanced(corpus):
    """Balance holds within every scenario, not just in aggregate."""
    for scenario in corpus:
        counts: dict[str, int] = {}
        for write in scenario.writes:
            counts[write.write_class.value] = counts.get(write.write_class.value, 0) + 1
        assert set(counts.values()) == {WRITES_PER_SCENARIO // 3}


def test_gold_transitions_cover_all_four_tiers(corpus):
    """Routine accepts, corrections supersede, conflicts review or reject."""
    counts = corpus.gold_counts()
    assert counts[Tier.accept.value] == 500
    assert counts[Tier.supersede.value] == 500
    assert counts[Tier.review.value] + counts[Tier.reject.value] == 500
    assert counts[Tier.review.value] > 0 and counts[Tier.reject.value] > 0


def test_gold_transition_follows_the_write_class(corpus):
    """Class and gold transition agree by construction (Req 9.7)."""
    for write in corpus.writes:
        if write.write_class is WriteClass.routine:
            assert write.gold_transition is Tier.accept
        elif write.write_class is WriteClass.correction:
            assert write.gold_transition is Tier.supersede
        else:
            assert write.gold_transition in (Tier.review, Tier.reject)


# --------------------------------------------------------------------------- #
# Variation axes (Req 9.3)
# --------------------------------------------------------------------------- #
def test_every_variation_axis_is_exercised(corpus):
    """All eight §3.2 axes appear in the corpus (Req 9.3)."""
    coverage = corpus.perturbation_coverage()
    assert set(coverage) == set(PERTURBATION_AXES)
    for axis, count in coverage.items():
        assert count > 0, f"axis {axis} is never exercised"


def test_every_template_family_is_generated(corpus):
    """No template is unreachable — each contributes cases (Req 9.3)."""
    counts = corpus.template_counts()
    assert len(counts) >= 15
    for template, count in counts.items():
        assert count > 0, f"template {template} never fires"


# --------------------------------------------------------------------------- #
# Poisoned evidence (Req 9.4)
# --------------------------------------------------------------------------- #
def test_twenty_percent_of_scenarios_are_poisoned(corpus):
    """Poisoned or unsupported evidence appears in 20% of scenarios (Req 9.4)."""
    poisoned = [s for s in corpus if s.poisoned]
    assert len(poisoned) == int(round(N_SCENARIOS * POISONED_SCENARIO_FRACTION)) == 10


def test_poisoned_scenarios_contain_poisoned_writes(corpus):
    """A scenario marked poisoned actually carries poisoned-evidence writes."""
    for scenario in corpus:
        if scenario.poisoned:
            assert any(w.poisoned_evidence for w in scenario.writes)
        else:
            assert not any(w.poisoned_evidence for w in scenario.writes)


def test_poisoned_writes_have_minimal_authority(corpus):
    """Poisoned evidence cannot confer authority (Req 1.9)."""
    for write in corpus.writes:
        if write.poisoned_evidence:
            assert write.authority <= 0.05


# --------------------------------------------------------------------------- #
# Partitions (Req 9.5)
# --------------------------------------------------------------------------- #
def test_partition_sizes_match_the_paper(corpus):
    """Training 25, development 10, canary 5, test 10 (Req 9.5)."""
    for name, expected in PARTITION_SIZES.items():
        assert len(corpus.partition(name)) == expected


def test_partitions_are_disjoint_and_exhaustive(corpus):
    """Every scenario is in exactly one partition."""
    seen: set[str] = set()
    for partition in Partition:
        ids = {s.scenario_id for s in corpus.partition(partition)}
        assert not (ids & seen)
        seen |= ids
    assert len(seen) == len(corpus)


def test_no_entity_id_is_shared_across_partitions(corpus):
    """Namespacing prevents any fact or alias crossing a partition (Req 9.5).

    Shared vocabulary nodes (``status:*``) are excluded: they are a fixed
    enumeration, not scenario content.
    """
    by_partition: dict[str, set[str]] = {}
    for partition in Partition:
        ids: set[str] = set()
        for scenario in corpus.partition(partition):
            ids |= {
                e.entity_id
                for e in scenario.entities
                if not e.entity_id.startswith("status:")
            }
        by_partition[partition.value] = ids

    names = list(by_partition)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = by_partition[left] & by_partition[right]
            assert not overlap, f"{left} and {right} share {sorted(overlap)[:3]}"


def test_no_write_id_is_duplicated(corpus):
    """Write ids are globally unique."""
    ids = [w.write_id for w in corpus.writes]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- #
# Ground truth (Req 9.6, 9.7)
# --------------------------------------------------------------------------- #
def test_every_write_carries_complete_ground_truth(corpus):
    """Correct transition, consequentiality, and least evidence (Req 9.6)."""
    for write in corpus.writes:
        assert isinstance(write.gold_transition, Tier)
        assert isinstance(write.consequential, bool)
        assert write.minimum_evidence and write.minimum_evidence.strip()


def test_consequentiality_follows_the_stated_rule(corpus):
    """``consequential = q ≥ 0.60 or v ≤ 0.30`` (Req 9.6)."""
    for write in corpus.writes:
        expected = write.consequence >= 0.60 or write.reversibility <= 0.30
        assert write.consequential is expected


def test_rubric_values_are_in_the_unit_interval(corpus):
    """``q``, ``v``, ``a`` are all valid probabilities."""
    for write in corpus.writes:
        assert 0.0 <= write.consequence <= 1.0
        assert 0.0 <= write.reversibility <= 1.0
        assert 0.0 <= write.authority <= 1.0
        assert 0.0 <= write.confidence <= 1.0


def test_independent_corrections_displace_a_seeded_incumbent(corpus):
    """An independent correction's gold ``supersede`` needs a seeded incumbent.

    Chain corrections are excluded: their incumbent is created by an earlier write
    in the same chain rather than seeded up front.
    """
    for scenario in corpus:
        incumbents = {
            (a.subject_id, a.predicate): a.assertion_id for a in scenario.incumbents
        }
        for write in scenario.writes:
            if write.write_class is not WriteClass.correction or write.contended:
                continue
            key = (write.subject_id, write.predicate)
            assert key in incumbents, f"{write.write_id} has nothing to supersede"
            assert write.predicate in SINGLE_VALUED_PREDICATES
            assert incumbents[key] == write.incumbent_id


def test_authoritative_correction_templates_clear_the_authority_floor(corpus):
    """The ``correction_*`` and chain-authoritative families satisfy ``h(u)`` (§3.3).

    The discriminating pairs deliberately sit *below* the floor so they can vary a
    single scalar while holding ``h(u) = 0``, so they are excluded here.
    """
    for write in corpus.writes:
        if write.template.startswith("correction_") or "authoritative" in write.template:
            assert write.authority >= 0.90, write.template


def test_discriminating_pairs_hold_the_authority_guard_off(corpus):
    """Every discriminating case has ``h(u) = 0`` so one scalar is decisive."""
    for write in corpus.writes:
        if write.template.startswith("discriminating_"):
            assert write.authority < 0.90
            assert write.write_intent.value == "new_fact"


def test_routine_writes_displace_no_seeded_incumbent(corpus):
    """A routine accept must not contend with prior durable memory."""
    for scenario in corpus:
        incumbents = {(a.subject_id, a.predicate) for a in scenario.incumbents}
        for write in scenario.writes:
            if write.write_class is WriteClass.routine:
                assert (write.subject_id, write.predicate) not in incumbents


def test_only_declared_chains_contend_for_a_target(corpus):
    """Target reuse happens only inside a declared contention chain (Req 9.7).

    Independent writes keep order-independent gold labels. Chain writes are
    order-*dependent* by design, and their shared target is what creates the
    cascade §3.1 requires — so reuse must be confined to them and declared.
    """
    for scenario in corpus:
        seen: dict[tuple[str, str], str | None] = {}
        for write in scenario.writes:
            if write.predicate not in SINGLE_VALUED_PREDICATES:
                continue
            key = (write.subject_id, write.predicate)
            if key in seen:
                assert write.chain_id is not None, (
                    f"{write.write_id} reuses {key} without declaring a chain"
                )
                assert write.chain_id == seen[key], (
                    f"{write.write_id} reuses {key} from a different chain"
                )
            seen[key] = write.chain_id


def test_writes_are_temporally_ordered(corpus):
    """Timestamps are nondecreasing in write order, except the undated case."""
    for scenario in corpus:
        stamps = [w.valid_from for w in scenario.writes if w.valid_from is not None]
        assert stamps == sorted(stamps)


def test_dated_writes_postdate_their_incumbent(corpus):
    """A correction is the newer fact, which ``h(u)`` requires (Req 4.4)."""
    for scenario in corpus:
        incumbents = {a.assertion_id: a for a in scenario.incumbents}
        for write in scenario.writes:
            if write.valid_from is None or write.incumbent_id is None:
                continue
            incumbent = incumbents.get(write.incumbent_id)
            if incumbent is None or incumbent.valid_from is None:
                continue
            assert write.valid_from > incumbent.valid_from


# --------------------------------------------------------------------------- #
# Contention chains (§3.1 order sensitivity)
# --------------------------------------------------------------------------- #
def test_corpus_contains_contention_chains(corpus):
    """Some writes must share a target, or replay order cannot matter (§3.1)."""
    counts = corpus.chain_counts()
    assert counts["n_chains"] > 0
    assert counts["n_contended_writes"] > 0
    assert 0.0 < counts["contended_fraction"] < 1.0


def test_every_chain_type_appears(corpus):
    """All three cascade mechanisms are exercised across the corpus."""
    templates = set(corpus.template_counts())
    for kind in ("chain_status_", "chain_slot_", "chain_assign_"):
        assert any(t.startswith(kind) for t in templates), kind


def test_chain_writes_share_one_target_in_order(corpus):
    """A chain is a contiguous-in-time run over a single target."""
    chains: dict[str, list] = {}
    for scenario in corpus:
        for write in scenario.writes:
            if write.chain_id:
                chains.setdefault(write.chain_id, []).append(write)

    assert chains
    for chain_id, writes in chains.items():
        writes.sort(key=lambda w: w.index)
        targets = {(w.subject_id, w.predicate) for w in writes}
        assert len(targets) == 1, f"{chain_id} spans several targets"
        # Declared positions must agree with the realized temporal order.
        assert [w.chain_position for w in writes] == sorted(
            w.chain_position for w in writes
        )
        stamps = [w.valid_from for w in writes if w.valid_from]
        assert stamps == sorted(stamps)


def test_each_chain_mixes_all_three_write_classes(corpus):
    """Chains consume a 1:1:1 class budget, preserving the §3.2 balance."""
    chains: dict[str, list] = {}
    for write in corpus.writes:
        if write.chain_id:
            chains.setdefault(write.chain_id, []).append(write)
    for chain_id, writes in chains.items():
        classes = {w.write_class for w in writes}
        assert classes == {
            WriteClass.routine,
            WriteClass.correction,
            WriteClass.conflict,
        }, chain_id


def _chains(corpus) -> dict[str, list]:
    """Group writes by chain, in chain-position order."""
    out: dict[str, list] = {}
    for write in corpus.writes:
        if write.chain_id:
            out.setdefault(write.chain_id, []).append(write)
    for writes in out.values():
        writes.sort(key=lambda w: w.chain_position or 0)
    return out


def test_every_chain_contains_a_reviewable_write(corpus):
    """A chain without a reviewable write cannot cascade at all."""
    chains = _chains(corpus)
    assert chains
    for chain_id, writes in chains.items():
        assert any(
            w.gold_transition is Tier.review for w in writes
        ), f"{chain_id} has no reviewable write"


def test_corpus_contains_both_routing_and_answer_cascades(corpus):
    """Chains come in two flavours and both must be present.

    A **routing cascade** places the reviewable write before a later write, so a
    wrong commitment changes the decision the system faces next. An **answer
    cascade** places it last, so a wrong commitment survives into the durable value
    the downstream question reads. They stress different failure paths, and a
    corpus with only one kind would leave the other untested.
    """
    routing = 0
    answer = 0
    for writes in _chains(corpus).values():
        last = max(w.chain_position or 0 for w in writes)
        review_positions = [
            w.chain_position or 0 for w in writes if w.gold_transition is Tier.review
        ]
        if min(review_positions) < last:
            routing += 1
        if last in review_positions:
            answer += 1
    assert routing > 0, "no chain places a reviewable write before a dependent one"
    assert answer > 0, "no chain places a reviewable write last"


def test_routing_cascade_chains_have_a_dependent_write(corpus):
    """In a routing cascade, some write follows the reviewable one."""
    found = False
    for writes in _chains(corpus).values():
        last = max(w.chain_position or 0 for w in writes)
        review_positions = [
            w.chain_position or 0 for w in writes if w.gold_transition is Tier.review
        ]
        if min(review_positions) < last:
            found = True
            dependent = [w for w in writes if (w.chain_position or 0) > min(review_positions)]
            assert dependent
            # The dependent write must itself change durable state, or the earlier
            # error has nothing to propagate into.
            assert any(w.gold_transition is Tier.supersede for w in dependent)
    assert found


def test_answer_cascade_chains_end_on_a_reviewable_write(corpus):
    """In an answer cascade, the reviewable write is last and would corrupt the answer."""
    found = False
    for scenario in corpus:
        for writes in _chains_of(scenario).values():
            last = max(w.chain_position or 0 for w in writes)
            final_write = next(w for w in writes if (w.chain_position or 0) == last)
            if final_write.gold_transition is not Tier.review:
                continue
            found = True
            # The value the question expects comes from an earlier write in the
            # chain, so committing the final write would replace it.
            target = (final_write.subject_id, final_write.predicate)
            questions = {
                (q.subject_id, q.predicate): q for q in scenario.questions
            }
            assert target in questions
            assert questions[target].gold_object_id != final_write.object_id
    assert found


def _chains_of(scenario) -> dict[str, list]:
    """Group one scenario's writes by chain, in chain-position order."""
    out: dict[str, list] = {}
    for write in scenario.writes:
        if write.chain_id:
            out.setdefault(write.chain_id, []).append(write)
    for writes in out.values():
        writes.sort(key=lambda w: w.chain_position or 0)
    return out


def test_gold_trajectory_is_internally_admissible(corpus):
    """Applying gold transitions in order never produces an inadmissible step.

    This validates the forward-simulation property that makes chain gold labels
    objective: at each write, the declared gold transition is consistent with the
    state the *earlier* gold transitions produced. A gold ``supersede`` must have
    something to displace, and a gold ``accept`` must not silently contend with a
    current value.
    """
    for scenario in corpus:
        current: dict[tuple[str, str], str] = {
            (a.subject_id, a.predicate): a.object_id for a in scenario.incumbents
        }
        for write in scenario.writes:
            if write.predicate not in SINGLE_VALUED_PREDICATES:
                continue
            key = (write.subject_id, write.predicate)
            held = current.get(key)

            if write.gold_transition is Tier.supersede:
                assert held is not None, (
                    f"{write.write_id} golds supersede with no current value"
                )
                assert held != write.expected_object_after
            elif write.gold_transition is Tier.accept:
                assert held is None, (
                    f"{write.write_id} golds accept while {key} already holds {held}"
                )

            if (
                write.expected_object_after is not None
                and write.gold_transition in (Tier.accept, Tier.supersede)
            ):
                current[key] = write.expected_object_after


def test_indices_are_contiguous(corpus):
    """A scenario's write indices are ``0..29`` in temporal order."""
    for scenario in corpus:
        assert [w.index for w in scenario.writes] == list(range(len(scenario.writes)))


# --------------------------------------------------------------------------- #
# Determinism (Req 9.7)
# --------------------------------------------------------------------------- #
def test_generation_is_deterministic_for_a_seed():
    """The same seed yields an identical corpus (Req 9.7)."""
    a = generate_corpus(DEFAULT_SEED, n_scenarios=6)
    b = generate_corpus(DEFAULT_SEED, n_scenarios=6)
    assert [w.as_dict() for w in a.writes] == [w.as_dict() for w in b.writes]


def test_different_seeds_yield_different_corpora():
    """The seed actually varies the content."""
    a = generate_corpus(1337, n_scenarios=6)
    b = generate_corpus(4242, n_scenarios=6)
    assert [w.as_dict() for w in a.writes] != [w.as_dict() for w in b.writes]


def test_reduced_corpus_keeps_class_balance():
    """A smaller corpus stays balanced, so smoke runs remain interpretable."""
    small = generate_corpus(n_scenarios=9)
    counts = small.class_counts()
    assert len(set(counts.values())) == 1


def test_writes_per_scenario_must_be_divisible_by_three():
    """An unbalanced request is rejected rather than silently skewed."""
    with pytest.raises(ValueError):
        generate_corpus(n_scenarios=3, writes_per_scenario=31)


# --------------------------------------------------------------------------- #
# Downstream questions (§4.5)
# --------------------------------------------------------------------------- #
def test_questions_reference_writes_that_change_state(corpus):
    """Every question's gold answer is produced by some write's gold transition."""
    for scenario in corpus:
        expected = {
            (w.subject_id, w.predicate): w.expected_object_after
            for w in scenario.writes
            if w.expected_object_after is not None
            and w.gold_transition in (Tier.accept, Tier.supersede)
        }
        for question in scenario.questions:
            key = (question.subject_id, question.predicate)
            assert key in expected
            assert expected[key] == question.gold_object_id


def test_stale_answer_differs_from_the_gold_answer(corpus):
    """The stale value is a genuine alternative, not a restatement of the answer."""
    for scenario in corpus:
        for question in scenario.questions:
            if question.stale_object_id is not None:
                assert question.stale_object_id != question.gold_object_id


def test_every_scenario_has_questions(corpus):
    """Experiment 4 needs downstream questions for every scenario."""
    for scenario in corpus:
        assert scenario.questions
