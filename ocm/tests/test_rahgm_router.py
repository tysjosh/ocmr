"""RAHGM router, review queue, explanation depths, and OCMR non-regression.

Covers Req 4.6 (the commit seam), 5.x (review and release), 6.x (explanation
depth), and 15.2 (byte-identical OCMR behavior when RAHGM is disabled).
"""

from __future__ import annotations

import pytest

from ocm.core.config import Settings
from ocm.core.container import CoreContainer
from ocm.governance.conditions import Condition, build_governance
from ocm.governance.features import FeatureExtractor, RiskFeatures, WriteContext
from ocm.governance.policy import (
    EscalationPolicy,
    PolicyParameters,
    RouteGuards,
    Tier,
)
from ocm.governance.review_queue import (
    DEPTH_ORDER,
    ExplanationDepth,
    ReviewAction,
    ReviewQueue,
    depth_schedule,
    latin_square,
    render_explanation,
)
from ocm.governance.router import (
    GovernedCommitManager,
    RiskAdaptiveRouter,
    RoutingDecision,
)
from ocm.memory.contracts import CandidateAssertion, ValidationResult
from ocm.ontology.enums import AssertionStatus, QuarantineStatus, WriteIntent
from ocm.ontology.models import Assertion


@pytest.fixture()
def container() -> CoreContainer:
    """A hermetic, deterministic container."""
    return CoreContainer(
        Settings(deterministic_test_mode=True, chroma_mode="memory", extractor="mock")
    )


def _seed_slot(container: CoreContainer) -> tuple[str, str, str, str]:
    """Install ``Slot -[HAS_VALUE]-> SlotValue`` and return the ids."""
    slot, old, new = "slt-1", "val-old", "val-new"
    container.graph.add_entity("Slot", {"id": slot, "name": "location"})
    container.graph.add_entity("SlotValue", {"id": old, "value": "old", "name": "old"})
    container.graph.add_entity("SlotValue", {"id": new, "value": "new", "name": "new"})
    incumbent = Assertion(
        id="ast-1",
        subject_id=slot,
        predicate="HAS_VALUE",
        object_id=old,
        confidence=0.85,
        status=AssertionStatus.accepted,
        source_ref="observation:test:0",
        created_at=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc),
    )
    container.repo.upsert_assertion(incumbent)
    container.graph.add_assertion(incumbent)
    return slot, old, new, incumbent.id


def _candidate(slot: str, new: str, **overrides) -> CandidateAssertion:
    payload = {
        "subject_id": slot,
        "predicate": "HAS_VALUE",
        "object_id": new,
        "confidence": 0.95,
        "source_ref": "analyst:test:1",
        "write_intent": WriteIntent.correction,
    }
    payload.update(overrides)
    return CandidateAssertion(**payload)


# --------------------------------------------------------------------------- #
# Non-regression (Req 15.2)
# --------------------------------------------------------------------------- #
def test_disabled_router_is_a_transparent_passthrough(container: CoreContainer):
    """With ``router=None`` the governed manager delegates verbatim (Req 15.2)."""
    slot, _old, new, _incumbent = _seed_slot(container)
    inner = container.commit_manager
    governed = GovernedCommitManager(inner=inner, router=None, graph=container.graph)

    candidate = _candidate(slot, new)
    verdict = ValidationResult(valid=True, recommended_action="accept")
    outcome = governed.commit(candidate, verdict)

    assert outcome.decision == "accepted"
    assert governed.decisions == []


def test_ocmr_write_pipeline_is_untouched_by_import():
    """Importing RAHGM does not alter the OCMR write path (Req 15.1, 15.2)."""
    import ocm.governance  # noqa: F401
    from ocm.memory.commit_manager import CommitManager
    from ocm.memory.contracts import WriteOutcome

    # The decision vocabulary is unchanged: RAHGM adds no new outcome value.
    assert set(WriteOutcome.model_fields["decision"].annotation.__args__) == {
        "accepted",
        "superseded",
        "quarantined",
        "rejected",
    }
    assert hasattr(CommitManager, "commit")


# --------------------------------------------------------------------------- #
# Tier translation (Req 4.6)
# --------------------------------------------------------------------------- #
def _decision(tier: Tier, **feature_overrides) -> RoutingDecision:
    features = RiskFeatures(**feature_overrides)
    return RoutingDecision(
        tier=tier,
        risk=0.5,
        score=0.0,
        features=features,
        guards=RouteGuards(),
        rule="test",
        ocmr_action=None,
        ocmr_failed_check=None,
        ocmr_reason=None,
    )


def test_accept_tier_commits_as_accepted(container: CoreContainer):
    """The accept tier produces a durable accepted assertion."""
    slot, _old, new, _aid = _seed_slot(container)
    governed = GovernedCommitManager(inner=container.commit_manager, graph=container.graph)
    verdict = GovernedCommitManager.translate(
        ValidationResult(valid=True), _decision(Tier.accept)
    )
    assert verdict.recommended_action == "accept"
    outcome = container.commit_manager.commit(_candidate(slot, new), verdict)
    assert outcome.decision == "accepted"


def test_supersede_tier_retains_the_incumbent(container: CoreContainer):
    """Supersession retires the prior assertion but keeps the row (Req 5.4)."""
    slot, old, new, incumbent = _seed_slot(container)
    verdict = GovernedCommitManager.translate(
        ValidationResult(valid=True, conflicting_ids=[incumbent]),
        _decision(Tier.supersede, incumbent_ids=(incumbent,), incumbent_recoverable=True),
    )
    outcome = container.commit_manager.commit(_candidate(slot, new), verdict)

    assert outcome.decision == "superseded"
    retired = container.repo.get_assertion(incumbent)
    assert retired is not None
    assert retired.status == AssertionStatus.superseded
    assert container.provenance_tracker.for_subject(incumbent)


def test_supersede_without_an_incumbent_falls_back_to_review(container: CoreContainer):
    """A supersede route with no target is escalated, never silently accepted."""
    verdict = GovernedCommitManager.translate(
        ValidationResult(valid=True), _decision(Tier.supersede)
    )
    assert verdict.recommended_action == "quarantine"


def test_review_tier_quarantines_and_leaves_memory_intact(container: CoreContainer):
    """A held write never overwrites accepted memory (Req 5.1)."""
    slot, old, new, _incumbent = _seed_slot(container)
    verdict = GovernedCommitManager.translate(
        ValidationResult(valid=True), _decision(Tier.review)
    )
    outcome = container.commit_manager.commit(_candidate(slot, new), verdict)

    assert outcome.decision == "quarantined"
    assert outcome.quarantine_id
    # The incumbent value is still the current one.
    edges = container.graph.out_edges(slot, "HAS_VALUE")
    assert [obj for _s, obj, _k, _d in edges] == [old]


def test_review_verdict_reason_names_the_failed_checks():
    """The escalation reason explains itself rather than showing a bare score."""
    decision = _decision(
        Tier.review, f_c=1.0, consequence=0.8, failed_checks=("C7",)
    )
    verdict = GovernedCommitManager.translate(ValidationResult(valid=True), decision)
    assert "escalated for human review" in (verdict.reason or "")
    assert "C7" in (verdict.reason or "")


def test_routing_decision_rationale_lists_inputs_not_just_a_number():
    """``rationale()`` surfaces the features and the rule (Req 4.5)."""
    features = RiskFeatures(
        f_c=1.0,
        failed_checks=("C7",),
        unresolved_checks=("cardinality",),
        consequence=0.8,
        reversibility=0.3,
        authority=0.4,
    )
    decision = RoutingDecision(
        tier=Tier.review,
        risk=0.9,
        score=2.2,
        features=features,
        guards=RouteGuards(),
        rule="otherwise -> review",
        ocmr_action="quarantine",
        ocmr_failed_check="C7",
        ocmr_reason="conflict",
    )
    text = decision.rationale()
    assert "C7" in text and "cardinality" in text
    assert "consequence" in text and "reversibility" in text and "authority" in text
    assert "otherwise -> review" in text


# --------------------------------------------------------------------------- #
# Review and release (Req 5.3, 5.4)
# --------------------------------------------------------------------------- #
def _escalate(container: CoreContainer) -> tuple[ReviewQueue, str, str, str]:
    """Escalate one write and return ``(queue, item_id, slot, incumbent)``."""
    slot, _old, new, incumbent = _seed_slot(container)
    queue = ReviewQueue(
        commit_manager=container.commit_manager,
        quarantine_store=container.quarantine_store,
        graph=container.graph,
        repo=container.repo,
    )
    candidate = _candidate(slot, new)
    decision = _decision(
        Tier.review, incumbent_ids=(incumbent,), incumbent_recoverable=True
    )
    verdict = GovernedCommitManager.translate(ValidationResult(valid=True), decision)
    outcome = container.commit_manager.commit(candidate, verdict)
    item = queue.enqueue(candidate, decision, outcome, minimum_evidence="an authority check")
    return queue, item.item_id, slot, incumbent


def test_release_on_accept_retires_a_single_valued_incumbent(container: CoreContainer):
    """Releasing on ``accept`` must not leave two active values (Req 5.3, 5.4).

    ``HAS_VALUE`` is single-valued, so releasing the held write is a supersession
    whatever the analyst called the action. Committing it as a bare accept would
    leave a durable cardinality violation — the exact failure OCMR's gate exists
    to prevent.
    """
    queue, item_id, slot, incumbent = _escalate(container)
    record = queue.adjudicate(item_id, ReviewAction.accept, confidence=0.9)

    assert record.released
    assert record.outcome is not None
    assert record.outcome.decision == "superseded"
    # Exactly one active value remains.
    assert len(container.graph.out_edges(slot, "HAS_VALUE")) == 1
    assert container.repo.get_assertion(incumbent).status == AssertionStatus.superseded
    quarantine = queue.items[item_id].quarantine_id
    stored = [r for r in container.quarantine_store.list() if r.id == quarantine]
    assert stored and stored[0].status == QuarantineStatus.resolved


def test_release_on_accept_of_a_many_valued_predicate_accepts(container: CoreContainer):
    """With no single-valued incumbent to retire, a release is a plain accept."""
    container.graph.add_entity("Person", {"id": "per-1", "name": "A", "status": "active"})
    container.graph.add_entity("Organization", {"id": "org-1", "name": "O", "status": "active"})
    queue = ReviewQueue(
        commit_manager=container.commit_manager,
        quarantine_store=container.quarantine_store,
        graph=container.graph,
    )
    candidate = CandidateAssertion(
        subject_id="per-1",
        predicate="MEMBER_OF",
        object_id="org-1",
        confidence=0.9,
        source_ref="analyst:test:1",
        write_intent=WriteIntent.new_fact,
    )
    decision = _decision(Tier.review)
    verdict = GovernedCommitManager.translate(ValidationResult(valid=True), decision)
    outcome = container.commit_manager.commit(candidate, verdict)
    item = queue.enqueue(candidate, decision, outcome)

    record = queue.adjudicate(item.item_id, ReviewAction.accept)
    assert record.released
    assert record.outcome.decision == "accepted"


def test_upheld_hold_leaves_the_queue(container: CoreContainer):
    """A ``quarantine`` adjudication is a completed decision, not a deferral.

    Leaving it pending would present the same item on every later pass, which
    inflates any review-cost measurement.
    """
    queue, item_id, _slot, _incumbent = _escalate(container)
    queue.adjudicate(item_id, ReviewAction.quarantine)

    assert queue.items[item_id].resolved
    assert queue.pending() == []


def test_release_on_supersede_retires_the_incumbent(container: CoreContainer):
    """A released supersession retires the prior value and keeps it (Req 5.4)."""
    queue, item_id, slot, incumbent = _escalate(container)
    record = queue.adjudicate(item_id, ReviewAction.supersede, confidence=0.95)

    assert record.released
    assert record.outcome.decision == "superseded"
    retired = container.repo.get_assertion(incumbent)
    assert retired.status == AssertionStatus.superseded


def test_reject_dismisses_and_does_not_release(container: CoreContainer):
    """Rejecting dismisses the record and commits nothing."""
    queue, item_id, slot, _incumbent = _escalate(container)
    before = len(container.repo.list_assertions("accepted"))
    record = queue.adjudicate(item_id, ReviewAction.reject)

    assert not record.released
    assert len(container.repo.list_assertions("accepted")) == before
    quarantine = queue.items[item_id].quarantine_id
    stored = [r for r in container.quarantine_store.list() if r.id == quarantine]
    assert stored[0].status == QuarantineStatus.dismissed


def test_request_evidence_returns_the_item_to_the_queue(container: CoreContainer):
    """Asking for evidence keeps the item pending and counts the request."""
    queue, item_id, _slot, _incumbent = _escalate(container)
    queue.adjudicate(item_id, ReviewAction.request_evidence)

    item = queue.items[item_id]
    assert item.evidence_requests == 1
    assert not item.resolved
    assert item in queue.pending()


def test_queue_without_a_commit_manager_cannot_release(container: CoreContainer):
    """A queue with no commit manager records the decision but releases nothing."""
    slot, _old, new, incumbent = _seed_slot(container)
    queue = ReviewQueue(graph=container.graph)
    decision = _decision(Tier.review, incumbent_ids=(incumbent,))
    verdict = GovernedCommitManager.translate(ValidationResult(valid=True), decision)
    outcome = container.commit_manager.commit(_candidate(slot, new), verdict)
    item = queue.enqueue(_candidate(slot, new), decision, outcome)

    record = queue.adjudicate(item.item_id, ReviewAction.accept)
    assert not record.released


# --------------------------------------------------------------------------- #
# Explanation depth (Req 6.1, 6.2, 6.3)
# --------------------------------------------------------------------------- #
def test_explanation_depths_are_strictly_nested(container: CoreContainer):
    """``minimal ⊂ evidence ⊂ full`` (Req 6.1)."""
    queue, item_id, _slot, _incumbent = _escalate(container)
    item = queue.items[item_id]

    minimal = render_explanation(item, ExplanationDepth.minimal)
    evidence = render_explanation(item, ExplanationDepth.evidence)
    full = render_explanation(item, ExplanationDepth.full)

    assert set(minimal) < set(evidence) < set(full)
    for key, value in minimal.items():
        if key == "depth":
            continue
        assert evidence[key] == value
        assert full[key] == value


def test_minimal_shows_recommendation_and_failed_checks(container: CoreContainer):
    """The minimal depth carries exactly what §3.4 specifies."""
    queue, item_id, _slot, _incumbent = _escalate(container)
    payload = render_explanation(queue.items[item_id], ExplanationDepth.minimal)
    for key in (
        "recommended_action",
        "failed_checks",
        "unresolved_checks",
        "incumbent_value",
        "proposed_value",
        "source",
        "available_actions",
    ):
        assert key in payload
    assert "supporting_evidence" not in payload


def test_evidence_depth_adds_provenance(container: CoreContainer):
    """The evidence depth adds snippets with provenance (Req 6.1)."""
    queue, item_id, _slot, _incumbent = _escalate(container)
    payload = render_explanation(queue.items[item_id], ExplanationDepth.evidence)
    assert "supporting_evidence" in payload
    assert "conflicting_evidence" in payload
    assert payload["provenance"]["source_ref"]
    assert "memory_timeline" not in payload


def test_full_depth_adds_timeline_alternatives_and_consequence(container: CoreContainer):
    """The full depth adds the timeline, alternatives, and reversibility (Req 6.1)."""
    queue, item_id, _slot, _incumbent = _escalate(container)
    payload = render_explanation(queue.items[item_id], ExplanationDepth.full)
    for key in (
        "memory_timeline",
        "alternative_actions",
        "reversibility",
        "predicted_consequence",
        "risk",
    ):
        assert key in payload
    assert payload["alternative_actions"]


def test_depth_does_not_change_the_route(container: CoreContainer):
    """Rendering at any depth leaves the routed tier untouched (Req 6.2)."""
    queue, item_id, _slot, _incumbent = _escalate(container)
    item = queue.items[item_id]
    before = item.decision.tier
    for depth in DEPTH_ORDER:
        render_explanation(item, depth)
    assert item.decision.tier is before


def test_latin_square_is_balanced():
    """Each level appears equally often across offsets (Req 6.3)."""
    n_levels, n_blocks = 3, 9
    counts = [0] * n_levels
    for offset in range(n_levels):
        for level in latin_square(n_levels, n_blocks, offset):
            counts[level] += 1
    assert len(set(counts)) == 1


def test_depth_schedule_covers_every_level():
    """A participant sees every depth (Req 6.3)."""
    schedule = depth_schedule(len(DEPTH_ORDER), offset=1)
    assert set(schedule) == set(DEPTH_ORDER)


# --------------------------------------------------------------------------- #
# Condition wiring (Req 10.1, 10.2)
# --------------------------------------------------------------------------- #
def test_build_governance_installs_the_governed_manager(container: CoreContainer):
    """Installing swaps the commit manager on the container and pipeline (Req 4.6)."""
    harness = build_governance(Condition.frozen_rahgm, container)
    assert container.commit_manager is harness.governed
    assert container.write_pipeline.commit_manager is harness.governed


def test_universal_review_escalates_everything_admissible(container: CoreContainer):
    """C1 queues every admissible write."""
    slot, _old, new, _incumbent = _seed_slot(container)
    harness = build_governance(Condition.universal_review, container)
    decision = harness.router.decide(
        _candidate(slot, new), ValidationResult(valid=True), container.graph, WriteContext()
    )
    assert decision.tier is Tier.review


def test_universal_review_still_rejects_malformed_writes(container: CoreContainer):
    """Queueing a structurally invalid write would waste review capacity."""
    slot, _old, new, _incumbent = _seed_slot(container)
    harness = build_governance(Condition.universal_review, container)
    decision = harness.router.decide(
        _candidate(slot, new, source_ref=""),
        ValidationResult(valid=False, failed_check="schema.required_fields"),
        container.graph,
        WriteContext(),
    )
    assert decision.tier is Tier.reject


def test_autonomous_ocmr_mirrors_the_native_verdict(container: CoreContainer):
    """C2 executes OCMR's own decision, with no review tier staffed."""
    slot, _old, new, _incumbent = _seed_slot(container)
    harness = build_governance(Condition.autonomous_ocmr, container)
    assert not harness.staffed
    for action, expected in (
        ("accept", Tier.accept),
        ("supersede", Tier.supersede),
        ("reject", Tier.reject),
        ("quarantine", Tier.review),
    ):
        decision = harness.router.decide(
            _candidate(slot, new),
            ValidationResult(valid=True, recommended_action=action),
            container.graph,
            WriteContext(),
        )
        assert decision.tier is expected


def test_fixed_threshold_escalates_on_low_confidence(container: CoreContainer):
    """C3 escalates below 0.80 confidence."""
    slot, _old, new, _incumbent = _seed_slot(container)
    harness = build_governance(Condition.fixed_threshold, container)
    decision = harness.router.decide(
        _candidate(slot, new, confidence=0.5),
        ValidationResult(valid=True, recommended_action="accept"),
        container.graph,
        WriteContext(consequence=0.1),
    )
    assert decision.tier is Tier.review
    assert "confidence" in decision.rule


def test_fixed_threshold_escalates_on_high_consequence(container: CoreContainer):
    """C3 also escalates high-consequence writes."""
    slot, _old, new, _incumbent = _seed_slot(container)
    harness = build_governance(Condition.fixed_threshold, container)
    decision = harness.router.decide(
        _candidate(slot, new, confidence=0.99),
        ValidationResult(valid=True, recommended_action="accept"),
        container.graph,
        WriteContext(consequence=0.95),
    )
    assert decision.tier is Tier.review
    assert "consequence" in decision.rule


def test_all_conditions_share_one_feature_extractor(container: CoreContainer):
    """Conditions are compared on identical features (Req 10.2)."""
    extractor = FeatureExtractor(settings=container.settings)
    harnesses = [
        build_governance(condition, container, feature_extractor=extractor, install=False)
        for condition in Condition
    ]
    assert all(h.router.features is extractor for h in harnesses)
