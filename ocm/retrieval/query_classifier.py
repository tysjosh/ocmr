"""Query Classifier (R0) for the retrieval pipeline (Req 14.1, 14.2).

The :class:`QueryClassifier` is the first stage (R0) of the read pipeline. It
inspects an incoming natural-language query and classifies it into exactly one
of six :data:`QueryType` values, extracts candidate entity mentions and relation
predicates, and decides whether semantic fallback is needed (Req 14.1, 14.2).

Classification is deterministic and dependency-free: it relies on keyword and
light regex heuristics rather than an LLM, so it is fast, reproducible, and
testable. The six types and their trigger heuristics follow the design's R0
table:

==================== ===================================================
query_type           Trigger heuristics
==================== ===================================================
direct_fact          "who owns", "who is assigned", "what is the status of"
temporal             "before", "after", "precedes", "order of", date phrases
planning             "next steps", "plan", "what should", "tasks for", "todo"
contradiction_check  "conflict", "contradiction", "disagree", "is it true that"
provenance_request   "source", "where did", "evidence for", "how do we know"
open_ended           fallback when no structural cue matches
==================== ===================================================

``needs_semantic_fallback`` is ``True`` unless the query is a pure structural
lookup with a confidently extracted entity *and* predicate (then the symbolic
retriever's results may suffice). ``contradiction_check`` is the downstream
"conflict query" signal used to include quarantined items (Req 16.3).

Requirements: 14.1, 14.2.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

# Predicates are drawn from the relation registry so extracted hints always map
# to real, registered relations (single source of truth: relations.py).
from ocm.ontology.relations import RELATION_SIGNATURES

QueryType = Literal[
    "direct_fact",
    "temporal",
    "planning",
    "contradiction_check",
    "open_ended",
    "provenance_request",
]


class QueryClassification(BaseModel):
    """Structured result of classifying a query (Req 14.2).

    Carries the chosen ``query_type``, candidate entity mentions, candidate
    relation predicates (registry keys such as ``OWNS``/``ASSIGNED_TO``), and
    ``needs_semantic_fallback`` indicating whether semantic retrieval should run
    in addition to (or instead of) symbolic retrieval.
    """

    query_type: QueryType
    entities: list[str] = Field(default_factory=list)
    predicates: list[str] = Field(default_factory=list)
    needs_semantic_fallback: bool = True


# Verb / phrase -> registry predicate hints. Each entry is a compiled pattern so
# matching is case-insensitive and respects word boundaries.
_PREDICATE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bown(?:s|ed|er|ership|ing)?\b", re.IGNORECASE), "OWNS"),
    (re.compile(r"\bassign(?:ed|s|ee|ment)?\b", re.IGNORECASE), "ASSIGNED_TO"),
    (re.compile(r"\bresponsible for\b", re.IGNORECASE), "ASSIGNED_TO"),
    (re.compile(r"\bbefore\b", re.IGNORECASE), "PRECEDES"),
    (re.compile(r"\bafter\b", re.IGNORECASE), "PRECEDES"),
    (re.compile(r"\bprecede(?:s|d)?\b", re.IGNORECASE), "PRECEDES"),
    (re.compile(r"\border of\b", re.IGNORECASE), "PRECEDES"),
    (re.compile(r"\bcontain(?:s|ed)?\b", re.IGNORECASE), "CONTAINS"),
    (re.compile(r"\btasks? (?:for|in|of|under)\b", re.IGNORECASE), "CONTAINS"),
    (re.compile(r"\bcontradict(?:s|ion|ory|ed)?\b", re.IGNORECASE), "CONTRADICTS"),
    (re.compile(r"\bconflict(?:s|ing)?\b", re.IGNORECASE), "CONTRADICTS"),
    (re.compile(r"\bdisagree(?:s|ment)?\b", re.IGNORECASE), "CONTRADICTS"),
    (re.compile(r"\bevidence (?:for|of)\b", re.IGNORECASE), "EVIDENCE_FOR"),
    (re.compile(r"\bsupport(?:s|ed|ing)?\b", re.IGNORECASE), "SUPPORTS"),
]

# query_type trigger heuristics, evaluated in priority order. The first family
# whose pattern matches wins, so more specific intents (provenance, conflict)
# are checked before the generic structural-fact intent.
_PROVENANCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bsource(?:s|d)?\b", re.IGNORECASE),
    re.compile(r"\bwhere did .* come from\b", re.IGNORECASE),
    re.compile(r"\bwhere (?:did|does|do) (?:this|it|that|these|those)\b", re.IGNORECASE),
    re.compile(r"\bevidence (?:for|of)?\b", re.IGNORECASE),
    re.compile(r"\bhow do we know\b", re.IGNORECASE),
    re.compile(r"\bhow do you know\b", re.IGNORECASE),
    re.compile(r"\bprovenance\b", re.IGNORECASE),
    re.compile(r"\bciteation?\b", re.IGNORECASE),
    re.compile(r"\bcite\b", re.IGNORECASE),
]

_CONTRADICTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bconflict(?:s|ing)?\b", re.IGNORECASE),
    re.compile(r"\bcontradict(?:s|ion|ory|ed)?\b", re.IGNORECASE),
    re.compile(r"\bdiscrepanc(?:y|ies)\b", re.IGNORECASE),
    re.compile(r"\bdisagree(?:s|ment)?\b", re.IGNORECASE),
    re.compile(r"\binconsisten(?:t|cy|cies)\b", re.IGNORECASE),
    re.compile(r"\bis it true (?:that)?\b", re.IGNORECASE),
]

_TEMPORAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bbefore\b", re.IGNORECASE),
    re.compile(r"\bafter\b", re.IGNORECASE),
    re.compile(r"\bprecede(?:s|d)?\b", re.IGNORECASE),
    re.compile(r"\border of\b", re.IGNORECASE),
    re.compile(r"\bwhen (?:did|does|do|was|were|is)\b", re.IGNORECASE),
    re.compile(r"\btimeline\b", re.IGNORECASE),
    re.compile(r"\bwhat happened\b", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),  # ISO date phrase
]

_PLANNING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bnext steps?\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s| is) next\b", re.IGNORECASE),
    re.compile(r"\bplan(?:s|ning|ned)?\b", re.IGNORECASE),
    re.compile(r"\bwhat should\b", re.IGNORECASE),
    re.compile(r"\btasks? (?:for|in|of|under|remaining)\b", re.IGNORECASE),
    re.compile(r"\bto-?do(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bremaining work\b", re.IGNORECASE),
    re.compile(r"\bbacklog\b", re.IGNORECASE),
]

_DIRECT_FACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bwho owns\b", re.IGNORECASE),
    re.compile(r"\bwho is (?:the )?owner\b", re.IGNORECASE),
    re.compile(r"\bwho is assigned\b", re.IGNORECASE),
    re.compile(r"\bwho(?:'s| is) responsible\b", re.IGNORECASE),
    re.compile(r"\bwhat is the (?:current )?status of\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s| is) the status of\b", re.IGNORECASE),
    re.compile(r"\bwho is\b", re.IGNORECASE),
]

# Common interrogative / stop words that, although capitalized at sentence
# start, are not entity mentions.
_STOPWORD_TOKENS: frozenset[str] = frozenset(
    {
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "which",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "to",
        "for",
        "do",
        "does",
        "did",
        "this",
        "that",
        "it",
        "tell",
        "me",
        "about",
        "there",
    }
)

# Capitalized multi-word mentions, e.g. "Project Orion", "Task T1".
_CAPITALIZED_MENTION = re.compile(r"\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*)*\b")
# Identifier-like tokens, e.g. "T1", "P-12", "PROJ-7".
_IDENTIFIER_MENTION = re.compile(r"\b[A-Z]{1,}-?\d+\b")
# Quoted spans, e.g. 'about "Project Orion"'.
_QUOTED_MENTION = re.compile(r"[\"']([^\"']+)[\"']")

# Leading words that introduce an entity but are not part of its proper name on
# their own (kept when followed by a specific name, stripped when standalone).
_GENERIC_TYPE_PREFIXES = frozenset({"Project", "Task", "Event", "Document"})


class QueryClassifier:
    """Deterministic, heuristic query classifier (R0) (Req 14.1, 14.2)."""

    def classify(self, query: str) -> QueryClassification:
        """Classify ``query`` and extract entities/predicates (Req 14.1, 14.2).

        Returns a :class:`QueryClassification`. The query is assigned exactly
        one ``query_type``; ``needs_semantic_fallback`` is ``False`` only for a
        pure structural lookup (a ``direct_fact`` or ``temporal`` query with at
        least one entity and one predicate confidently extracted).
        """
        text = (query or "").strip()

        entities = self._extract_entities(text)
        predicates = self._extract_predicates(text)
        query_type = self._classify_type(text)

        # A pure structural lookup can be answered symbolically: only then can
        # we skip the semantic fallback. Open-ended (and everything lacking a
        # confident symbolic anchor) always needs the semantic retriever.
        is_structural = query_type in ("direct_fact", "temporal")
        has_symbolic_anchor = bool(entities) and bool(predicates)
        needs_semantic_fallback = not (is_structural and has_symbolic_anchor)

        return QueryClassification(
            query_type=query_type,
            entities=entities,
            predicates=predicates,
            needs_semantic_fallback=needs_semantic_fallback,
        )

    # -- type classification ------------------------------------------------

    @staticmethod
    def _matches_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
        return any(p.search(text) for p in patterns)

    def _classify_type(self, text: str) -> QueryType:
        if not text:
            return "open_ended"
        # Order matters: most specific intents first.
        if self._matches_any(text, _PROVENANCE_PATTERNS):
            return "provenance_request"
        if self._matches_any(text, _CONTRADICTION_PATTERNS):
            return "contradiction_check"
        if self._matches_any(text, _PLANNING_PATTERNS):
            return "planning"
        if self._matches_any(text, _TEMPORAL_PATTERNS):
            return "temporal"
        if self._matches_any(text, _DIRECT_FACT_PATTERNS):
            return "direct_fact"
        return "open_ended"

    # -- entity extraction --------------------------------------------------

    def _extract_entities(self, text: str) -> list[str]:
        if not text:
            return []

        mentions: list[str] = []

        # 1) Quoted spans are taken verbatim — explicit user intent.
        for match in _QUOTED_MENTION.finditer(text):
            mentions.append(match.group(1).strip())

        # Work on a copy with quoted spans removed so we don't double-count.
        residual = _QUOTED_MENTION.sub(" ", text)

        # 2) Identifier-like tokens (T1, P-12).
        for match in _IDENTIFIER_MENTION.finditer(residual):
            mentions.append(match.group(0))

        # 3) Capitalized mentions, ignoring a leading sentence-start word and
        #    pure stop words.
        for match in _CAPITALIZED_MENTION.finditer(residual):
            span = match.group(0).strip()
            cleaned = self._clean_capitalized_mention(span, match.start())
            if cleaned:
                mentions.append(cleaned)

        # Dedupe while preserving order.
        seen: set[str] = set()
        ordered: list[str] = []
        for mention in mentions:
            key = mention.lower()
            if mention and key not in seen:
                seen.add(key)
                ordered.append(mention)
        return ordered

    def _clean_capitalized_mention(self, span: str, start: int) -> str:
        tokens = span.split()

        # Drop a leading capitalized stop word (sentence-initial interrogative
        # like "Who"/"What"/"When").
        if start == 0 and tokens and tokens[0].lower() in _STOPWORD_TOKENS:
            tokens = tokens[1:]

        # Strip any remaining leading/trailing stop words.
        while tokens and tokens[0].lower() in _STOPWORD_TOKENS:
            tokens = tokens[1:]
        while tokens and tokens[-1].lower() in _STOPWORD_TOKENS:
            tokens = tokens[:-1]

        if not tokens:
            return ""

        # A standalone generic type word ("Project", "Task") with no specific
        # name is not a useful entity mention.
        if len(tokens) == 1 and tokens[0] in _GENERIC_TYPE_PREFIXES:
            return ""

        return " ".join(tokens)

    # -- predicate extraction -----------------------------------------------

    def _extract_predicates(self, text: str) -> list[str]:
        if not text:
            return []
        found: list[str] = []
        seen: set[str] = set()
        for pattern, predicate in _PREDICATE_PATTERNS:
            if predicate in seen:
                continue
            if pattern.search(text):
                # Defensive: only emit predicates that exist in the registry.
                if predicate in RELATION_SIGNATURES:
                    found.append(predicate)
                    seen.add(predicate)
        return found
