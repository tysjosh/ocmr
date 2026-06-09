"""W3 — Entity Resolver.

Conservative entity resolution over the accepted-memory :class:`GraphStore`.
Given a normalized entity mention (``entity_ref``: at least ``type`` and
``name``, optionally ``id``, ``aliases``, and a ``context`` block), the resolver
decides whether the mention refers to an entity that already exists or whether a
new entity should be minted — and, crucially, it **never silently merges two
distinct entities**. When the evidence is suggestive but not conclusive it
creates a new entity and flags a ``POSSIBLY_SAME_AS`` link to the candidate(s)
for downstream/human review (Req 5.6).

The matching priority order is applied **exactly** (Req 5.8):

1. **Exact ID match** → ``resolved_existing`` (Req 5.1).
2. **Exact normalized name + type match** → ``resolved_existing`` (Req 5.2).
3. **Alias + type match** → ``resolved_existing`` (Req 5.3).
4. **Contextual match** (co-occurring relation / source_ref evidence) →
   ``resolved_existing`` (Req 5.4).
5. **No match** → create a new entity, ``created_new`` (Req 5.5).
6. **Uncertain match** → create a ``POSSIBLY_SAME_AS`` relation, ``possible_match``
   (Req 5.6).

The result is a :class:`ResolutionOutcome` carrying ``resolution_status``,
``entity_id``, and ``candidate_matches`` (Req 5.7). When the resolver cannot
mint an id (no :class:`IdGenerator` supplied) and cannot confidently resolve, it
returns ``unresolved`` so the dependent candidate assertion is quarantined
rather than committed against a guessed identity.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8.
"""

from __future__ import annotations

import re
from typing import Any

from ocm.core.ids import IdGenerator
from ocm.memory.contracts import ResolutionOutcome
from ocm.memory.graph_store import GraphStore
from ocm.ontology.enums import ResolutionStatus

POSSIBLY_SAME_AS = "POSSIBLY_SAME_AS"

# Confidence attached to a generated POSSIBLY_SAME_AS link. It is intentionally
# below the contradiction-gate / high-confidence threshold (0.8) so an uncertain
# match is never treated as a confident fact downstream.
POSSIBLY_SAME_AS_CONFIDENCE = 0.5

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize_name(name: str) -> str:
    """Canonical, case-insensitive form used purely for *matching*.

    Lowercases, strips punctuation, and collapses internal whitespace. This is a
    matching key only — it never mutates or merges the stored entities (the
    Normalizer (W2) owns canonical storage; merging is the Resolver's job and is
    done conservatively).
    """
    if not name:
        return ""
    lowered = _PUNCT.sub(" ", name.lower())
    return _WS.sub(" ", lowered).strip()


def _tokens(name: str) -> set[str]:
    norm = normalize_name(name)
    return set(norm.split(" ")) if norm else set()


class EntityResolver:
    """Conservative resolver applying the Req 5.8 priority order exactly."""

    def resolve(
        self,
        entity_ref: dict[str, Any],
        graph: GraphStore,
        ids: IdGenerator | None = None,
        source_ref: str = "",
    ) -> ResolutionOutcome:
        """Resolve a single normalized entity mention to an id.

        :param entity_ref: normalized mention with at least ``type`` and
            ``name``; may also carry ``id``, ``aliases``, and a ``context`` dict
            (``{"related_ids": [...], "source_ref": "..."}``) used for
            contextual matching (Req 5.4).
        :param graph: the accepted-memory graph used to look up existing
            entities.
        :param ids: id generator used to mint a new id for create-new /
            possible-match outcomes. When ``None`` and no confident match
            exists, the outcome is ``unresolved``.
        :param source_ref: source reference threaded into id minting and the
            generated POSSIBLY_SAME_AS link.
        """
        entity_type = entity_ref.get("type")
        name = entity_ref.get("name", "") or ""

        # A reference without a type cannot be matched against typed nodes nor
        # minted meaningfully: surface it as unresolved (Req 5.7).
        if not entity_type:
            return ResolutionOutcome(
                resolution_status=ResolutionStatus.unresolved,
                entity_id=None,
                candidate_matches=[],
            )

        # --- 1. Exact ID match (Req 5.1) ----------------------------------
        ref_id = entity_ref.get("id")
        if ref_id and graph.has_entity(ref_id):
            return ResolutionOutcome(
                resolution_status=ResolutionStatus.resolved_existing,
                entity_id=ref_id,
                candidate_matches=[],
            )

        # --- 2. Exact normalized name + type match (Req 5.2) --------------
        name_matches = self._match_by_name(graph, entity_type, name)
        if len(name_matches) == 1:
            return ResolutionOutcome(
                resolution_status=ResolutionStatus.resolved_existing,
                entity_id=name_matches[0],
                candidate_matches=[],
            )
        if len(name_matches) > 1:
            # Multiple entities of the same type share the exact normalized
            # name: genuinely ambiguous, so do not pick one — flag for review.
            return self._uncertain(entity_ref, name_matches, ids, source_ref)

        # --- 3. Alias + type match (Req 5.3) ------------------------------
        alias_matches = self._match_by_alias(graph, entity_type, entity_ref, name)
        if len(alias_matches) == 1:
            return ResolutionOutcome(
                resolution_status=ResolutionStatus.resolved_existing,
                entity_id=alias_matches[0],
                candidate_matches=[],
            )
        if len(alias_matches) > 1:
            return self._uncertain(entity_ref, alias_matches, ids, source_ref)

        # --- 4. Contextual match (Req 5.4) --------------------------------
        # Near matches: same type, token overlap with the mention, but neither
        # an exact-name nor an alias hit. These are the *uncertain* candidates.
        near_matches = self._near_matches(graph, entity_type, name)
        contextual = self._contextual_match(graph, entity_ref, near_matches)
        if contextual is not None:
            return ResolutionOutcome(
                resolution_status=ResolutionStatus.resolved_existing,
                entity_id=contextual,
                candidate_matches=[],
            )

        # --- 6. Uncertain match (Req 5.6) ---------------------------------
        # Near matches exist but context did not confidently disambiguate.
        if near_matches:
            return self._uncertain(entity_ref, near_matches, ids, source_ref)

        # --- 5. No match → create new (Req 5.5) ---------------------------
        return self._create_new(entity_type, name, ids, source_ref)

    # ------------------------------------------------------------------
    # matching helpers
    # ------------------------------------------------------------------
    def _match_by_name(
        self, graph: GraphStore, entity_type: str, name: str
    ) -> list[str]:
        """Existing entity ids of ``entity_type`` whose name matches exactly."""
        target = normalize_name(name)
        if not target:
            return []
        matches: list[str] = []
        for node_id in sorted(graph.node_ids()):
            if graph.get_entity_type(node_id) != entity_type:
                continue
            payload = graph.get_entity_payload(node_id) or {}
            if normalize_name(payload.get("name", "")) == target:
                matches.append(node_id)
        return matches

    def _match_by_alias(
        self,
        graph: GraphStore,
        entity_type: str,
        entity_ref: dict[str, Any],
        name: str,
    ) -> list[str]:
        """Existing ids of ``entity_type`` matched via alias on either side.

        A match occurs when the mention's name/aliases intersect an existing
        entity's name/aliases (all normalized). Entities already covered by an
        exact-name hit are excluded so the alias step is strictly additive.
        """
        # Build the mention's normalized name+alias set.
        ref_keys = {normalize_name(name)} if name else set()
        for alias in entity_ref.get("aliases", []) or []:
            na = normalize_name(alias)
            if na:
                ref_keys.add(na)
        if not ref_keys:
            return []

        exact = set(self._match_by_name(graph, entity_type, name))
        matches: list[str] = []
        for node_id in sorted(graph.node_ids()):
            if node_id in exact:
                continue
            if graph.get_entity_type(node_id) != entity_type:
                continue
            payload = graph.get_entity_payload(node_id) or {}
            existing_keys = set()
            existing_name = normalize_name(payload.get("name", ""))
            if existing_name:
                existing_keys.add(existing_name)
            for alias in payload.get("aliases", []) or []:
                na = normalize_name(alias)
                if na:
                    existing_keys.add(na)
            if ref_keys & existing_keys:
                matches.append(node_id)
        return matches

    def _near_matches(
        self, graph: GraphStore, entity_type: str, name: str
    ) -> list[str]:
        """Same-type entities with partial token overlap (uncertain matches).

        Captures cases like ``"Bob"`` vs ``"Bob Smith"``: one mention's tokens
        are a subset of the other's, but it is not an exact or alias hit. These
        are treated as *uncertain* — never auto-merged.
        """
        ref_tokens = _tokens(name)
        if not ref_tokens:
            return []
        exact = set(self._match_by_name(graph, entity_type, name))
        matches: list[str] = []
        for node_id in sorted(graph.node_ids()):
            if node_id in exact:
                continue
            if graph.get_entity_type(node_id) != entity_type:
                continue
            payload = graph.get_entity_payload(node_id) or {}
            existing_tokens = _tokens(payload.get("name", ""))
            if not existing_tokens:
                continue
            # Subset overlap in either direction signals a possible same-entity.
            if ref_tokens <= existing_tokens or existing_tokens <= ref_tokens:
                matches.append(node_id)
        return matches

    def _contextual_match(
        self,
        graph: GraphStore,
        entity_ref: dict[str, Any],
        near_matches: list[str],
    ) -> str | None:
        """Resolve via contextual evidence, conservatively (Req 5.4).

        Returns a single id only when exactly one near-match candidate is
        positively supported by context (it is graph-connected to one of the
        mention's ``context.related_ids``). Any ambiguity yields ``None`` so the
        decision falls through to the uncertain (POSSIBLY_SAME_AS) path.
        """
        context = entity_ref.get("context") or {}
        related_ids = set(context.get("related_ids", []) or [])
        if not related_ids or not near_matches:
            return None

        supported: list[str] = []
        for cand in near_matches:
            neighbors = set(graph.neighbors_out(cand)) | set(graph.neighbors_in(cand))
            if neighbors & related_ids:
                supported.append(cand)
        # Only resolve when the context points to exactly one candidate.
        if len(supported) == 1:
            return supported[0]
        return None

    # ------------------------------------------------------------------
    # outcome builders
    # ------------------------------------------------------------------
    def _create_new(
        self,
        entity_type: str,
        name: str,
        ids: IdGenerator | None,
        source_ref: str,
    ) -> ResolutionOutcome:
        """Mint a brand-new entity id, or ``unresolved`` if none can be minted."""
        if ids is None:
            return ResolutionOutcome(
                resolution_status=ResolutionStatus.unresolved,
                entity_id=None,
                candidate_matches=[],
            )
        new_id = ids.entity_id(entity_type, normalize_name(name), source_ref)
        return ResolutionOutcome(
            resolution_status=ResolutionStatus.created_new,
            entity_id=new_id,
            candidate_matches=[],
        )

    def _uncertain(
        self,
        entity_ref: dict[str, Any],
        candidates: list[str],
        ids: IdGenerator | None,
        source_ref: str,
    ) -> ResolutionOutcome:
        """Produce a ``possible_match`` outcome (Req 5.6).

        Conservatively creates a *new* entity for the mention (no silent merge)
        and records the candidate existing ids in ``candidate_matches`` so a
        ``POSSIBLY_SAME_AS`` link can be emitted for review
        (:meth:`build_possibly_same_as`). If no id can be minted the outcome is
        ``unresolved`` but the candidates are still surfaced.
        """
        entity_type = entity_ref.get("type")
        name = entity_ref.get("name", "") or ""
        unique_candidates = sorted(set(candidates))
        if ids is None:
            return ResolutionOutcome(
                resolution_status=ResolutionStatus.unresolved,
                entity_id=None,
                candidate_matches=unique_candidates,
            )
        new_id = ids.entity_id(entity_type, normalize_name(name), source_ref)
        return ResolutionOutcome(
            resolution_status=ResolutionStatus.possible_match,
            entity_id=new_id,
            candidate_matches=unique_candidates,
        )

    # ------------------------------------------------------------------
    # POSSIBLY_SAME_AS relation helper
    # ------------------------------------------------------------------
    @staticmethod
    def build_possibly_same_as(
        outcome: ResolutionOutcome,
        source_ref: str = "",
        confidence: float = POSSIBLY_SAME_AS_CONFIDENCE,
    ) -> list[dict[str, Any]]:
        """Build POSSIBLY_SAME_AS relation dicts for a ``possible_match`` outcome.

        Returns one relation per candidate (subject = the newly created entity,
        object = a candidate existing entity). The downstream Assertion Builder
        (W4) / Commit Manager (W8) turn these into low-confidence links so the
        uncertainty is captured in memory for later resolution. For any
        non-``possible_match`` outcome (or when there is no minted id) this
        returns an empty list.
        """
        if (
            outcome.resolution_status != ResolutionStatus.possible_match
            or not outcome.entity_id
        ):
            return []
        return [
            {
                "subject": outcome.entity_id,
                "predicate": POSSIBLY_SAME_AS,
                "object": candidate,
                "confidence": confidence,
                "source_ref": source_ref,
            }
            for candidate in outcome.candidate_matches
        ]
