"""Agent_Loop — a lightweight, LangGraph-style memory loop (Req 20.1, 20.4).

The :class:`AgentLoop` exercises the OCM memory layer end to end **without
requiring LangGraph** as a dependency (Req 20.4). It is a plain Python state
machine whose node/edge structure mirrors LangGraph so a LangGraph port is
mechanical, but the default implementation imports nothing beyond OCM itself.

Node graph (Req 20.1)
---------------------
On each turn the loop walks six nodes::

    receive → retrieve → answer → extract → validate → commit
       ▲                                                   │
       └──────────────── next turn ◀───────────────────────┘

* **receive** — take the user message and assign a per-turn ``source_ref``.
* **retrieve** — call ``memory.query`` with the user input (Req 20.2).
* **answer** — shape a response from the :class:`EvidencePackage`, preferring
  the P1–P5 :class:`~ocm.agent.answer_policy.AnswerPolicy` (task 16.2) when it
  is available and falling back to a simple, deterministic renderer otherwise.
* **extract** — treat the turn content as candidate new memory.
* **validate** — decide whether the turn yielded memory worth persisting.
* **commit** — when it did, call ``memory.write`` with the turn content and the
  ``source_ref`` (Req 20.3); governance (validate/quarantine) runs inside the
  write pipeline.

The loop talks to memory **only** through the :class:`~ocm.agent.memory_tool.MemoryTool`
seam, so OCM stays pluggable (Req 20.1). Everything is deterministic given a
deterministic container, which makes the loop friendly to reproducible research
runs and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ocm.agent.memory_tool import MemoryTool
from ocm.retrieval.evidence_packager import EvidencePackage

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ocm.memory.write_pipeline import WriteResult


# --------------------------------------------------------------------------- #
# Answer rendering — prefer the P1–P5 AnswerPolicy (task 16.2) when present.
# --------------------------------------------------------------------------- #
def _load_answer_policy() -> Optional[Any]:
    """Return an :class:`AnswerPolicy` instance if task 16.2 has landed, else ``None``.

    The import is defensive so the loop works before (and independently of) the
    Answer Policy module; when the module is present the loop prefers it.
    """
    try:  # pragma: no cover - exercised once 16.2 lands
        from ocm.agent.answer_policy import AnswerPolicy  # type: ignore
    except Exception:
        return None
    try:  # pragma: no cover - defensive construction
        return AnswerPolicy()
    except Exception:
        return None


class _SimpleAnswerRenderer:
    """A minimal, deterministic fallback renderer for an :class:`EvidencePackage`.

    Used only when the P1–P5 :class:`AnswerPolicy` (task 16.2) is unavailable.
    It still honors the spirit of the policy at a basic level: lead with a
    derived answer or supporting assertions, surface conflicts separately, and
    state missing evidence.
    """

    def render(self, pkg: EvidencePackage, high_stakes: bool = False) -> str:
        """Render a plain-language answer from the evidence package."""
        lines: List[str] = []

        if pkg.answer:
            lines.append(str(pkg.answer))
        elif pkg.supporting_assertions:
            ids = ", ".join(sa.id for sa in pkg.supporting_assertions)
            lines.append(
                f"Based on {len(pkg.supporting_assertions)} supporting "
                f"assertion(s) (confidence {pkg.confidence:.2f}): {ids}."
            )
        else:
            lines.append("I don't have accepted memory that answers that.")

        # Surface each conflict separately (never merged).
        for conflict in pkg.conflicts:
            label = conflict.text or conflict.memory_id or "unknown item"
            reason = conflict.reason or "conflict"
            lines.append(f"Conflict ({reason}): {label}")

        # Attach provenance when high-stakes / decision-support.
        if high_stakes and pkg.supporting_sources:
            refs = ", ".join(
                str(getattr(src, "source_ref", getattr(src, "id", "?")))
                for src in pkg.supporting_sources
            )
            lines.append(f"Sources: {refs}")

        # State missing evidence.
        for note in pkg.missing_information:
            lines.append(f"Missing: {note}")

        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Turn result
# --------------------------------------------------------------------------- #
@dataclass
class TurnResult:
    """The outcome of a single :meth:`AgentLoop.run_turn` (Req 20.1).

    Carries the agent's ``answer``, the ``evidence`` that produced it, and the
    ``write_result`` from persisting any new memory (``None`` when the turn
    yielded nothing to write).
    """

    turn: int
    user_message: str
    source_ref: str
    answer: str
    evidence: EvidencePackage
    extracted_memory: Optional[str] = None
    committed: bool = False
    write_result: Optional["WriteResult"] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the turn outcome as a plain ``dict`` (answer + evidence + write)."""
        return {
            "turn": self.turn,
            "user_message": self.user_message,
            "source_ref": self.source_ref,
            "answer": self.answer,
            "evidence": self.evidence,
            "extracted_memory": self.extracted_memory,
            "committed": self.committed,
            "write_result": self.write_result,
        }


# Type of the optional extract hook: (user_message, evidence) -> new memory|None.
ExtractFn = Callable[[str, EvidencePackage], Optional[str]]

# Leading tokens that mark a turn as a question/command to memory rather than a
# statement of new memory. Such turns are retrieved against but NOT written back
# (writing a question would let the extractor mine spurious facts from it, e.g.
# "Who owns Project Orion?" -> a bogus Person "Who" OWNS Orion).
_QUESTION_LEADERS: frozenset[str] = frozenset(
    {
        "who", "what", "when", "where", "why", "how", "which", "whom", "whose",
        "is", "are", "was", "were", "do", "does", "did", "can", "could",
        "should", "would", "will", "has", "have", "had",
        "tell", "show", "list", "find", "give", "describe", "summarize",
        "explain", "search",
    }
)


def _is_question(text: str) -> bool:
    """Heuristic: whether ``text`` is a question/command rather than a statement.

    A turn is treated as a query (not new memory) when it ends with ``?`` or
    opens with an interrogative/imperative leader word. This keeps the agent
    loop from writing the user's *questions* into memory, where the extractor
    could mine spurious assertions from interrogative phrasing.
    """
    stripped = (text or "").strip()
    if not stripped:
        return True
    if stripped.endswith("?"):
        return True
    first = stripped.split(maxsplit=1)[0].strip(".,!:;\"'").lower()
    return first in _QUESTION_LEADERS


def _default_extract(user_message: str, evidence: EvidencePackage) -> Optional[str]:
    """Default extraction: persist statement turns; skip question/command turns.

    Returns the trimmed turn text as candidate memory for declarative input, and
    ``None`` for empty or question/command input so retrieval-only turns never
    pollute memory (the authoritative write-time governance still runs on what
    is returned).
    """
    text = (user_message or "").strip()
    if not text or _is_question(text):
        return None
    return text


class AgentLoop:
    """A dependency-free, LangGraph-style loop over the :class:`MemoryTool` (Req 20.4).

    The loop's node names and edges mirror a LangGraph graph so a port is
    mechanical, but the default runtime is a plain Python state machine with no
    LangGraph import (Req 20.4).
    """

    #: The ordered node sequence (Req 20.1).
    NODES: tuple[str, ...] = (
        "receive",
        "retrieve",
        "answer",
        "extract",
        "validate",
        "commit",
    )

    #: The directed edges between nodes (LangGraph-style; ``commit`` loops back
    #: to ``receive`` for the next turn). Kept as data so a LangGraph port can
    #: read it directly.
    EDGES: tuple[tuple[str, str], ...] = (
        ("receive", "retrieve"),
        ("retrieve", "answer"),
        ("answer", "extract"),
        ("extract", "validate"),
        ("validate", "commit"),
        ("commit", "receive"),
    )

    def __init__(
        self,
        memory: MemoryTool | Any,
        *,
        answer_policy: Optional[Any] = None,
        extract_fn: Optional[ExtractFn] = None,
        high_stakes: bool = False,
        top_k: int = 5,
        source_prefix: str = "agent-turn",
    ) -> None:
        """Build the loop.

        Args:
            memory: The :class:`MemoryTool` seam (Req 20.1). For convenience a
                :class:`CoreContainer` may be passed instead; it is wrapped in a
                :class:`MemoryTool` automatically.
            answer_policy: Optional object with ``render(pkg, high_stakes) -> str``.
                When omitted, the P1–P5 :class:`AnswerPolicy` (task 16.2) is used
                if available; otherwise a simple deterministic renderer.
            extract_fn: Optional hook mapping ``(user_message, evidence)`` to the
                new memory text to write (or ``None`` to skip the write). Defaults
                to treating the turn content as candidate memory.
            high_stakes: When ``True``, the answer node requests provenance in the
                rendered answer (P4 behavior when the Answer Policy is present).
            top_k: ``top_k`` forwarded to ``memory.query`` each turn.
            source_prefix: Prefix for the per-turn ``source_ref`` (Req 20.3).
        """
        self.memory = self._coerce_memory(memory)
        self.answer_policy = answer_policy or _load_answer_policy() or _SimpleAnswerRenderer()
        self.extract_fn = extract_fn or _default_extract
        self.high_stakes = high_stakes
        self.top_k = top_k
        self.source_prefix = source_prefix
        self._turn = 0

    @staticmethod
    def _coerce_memory(memory: MemoryTool | Any) -> MemoryTool:
        """Accept either a :class:`MemoryTool` or a raw container for convenience."""
        if isinstance(memory, MemoryTool):
            return memory
        # Duck-typed: anything exposing the two pipelines is a container.
        if hasattr(memory, "retrieval_pipeline") and hasattr(memory, "write_pipeline"):
            return MemoryTool(memory)
        raise TypeError(
            "AgentLoop expects a MemoryTool or a CoreContainer-like object "
            "exposing retrieval_pipeline and write_pipeline."
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run_turn(self, user_message: str, source_ref: Optional[str] = None) -> TurnResult:
        """Execute one turn through the node sequence and return its outcome.

        Walks receive → retrieve → answer → extract → validate → commit, calling
        ``memory.query`` (Req 20.2) and, when the turn yields new memory,
        ``memory.write`` (Req 20.3).

        Args:
            user_message: The incoming user message for this turn.
            source_ref: Optional explicit provenance ref; a per-turn default
                (``"{source_prefix}-{n}"``) is assigned when omitted (Req 20.3).

        Returns:
            A :class:`TurnResult` with the answer, the evidence, and the write
            result (``None`` when nothing was committed).
        """
        state: Dict[str, Any] = {"user_message": user_message, "source_ref": source_ref}

        # The node sequence mirrors the LangGraph graph (NODES/EDGES) but runs
        # as a plain, ordered state machine (Req 20.1, 20.4).
        state = self._receive(state)
        state = self._retrieve(state)
        state = self._answer(state)
        state = self._extract(state)
        state = self._validate(state)
        state = self._commit(state)

        return TurnResult(
            turn=state["turn"],
            user_message=state["user_message"],
            source_ref=state["source_ref"],
            answer=state["answer"],
            evidence=state["evidence"],
            extracted_memory=state.get("extracted_memory"),
            committed=state.get("committed", False),
            write_result=state.get("write_result"),
        )

    def run_session(self, messages: List[str]) -> List[TurnResult]:
        """Run a sequence of user messages as successive turns (Req 20.1).

        Demonstrates the ``commit → receive`` next-turn edge: each message is a
        full turn, and memory written on earlier turns is retrievable on later
        ones.
        """
        return [self.run_turn(message) for message in messages]

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #
    def _receive(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """receive — accept input and assign a per-turn ``source_ref`` (Req 20.1)."""
        self._turn += 1
        state["turn"] = self._turn
        if not state.get("source_ref"):
            state["source_ref"] = f"{self.source_prefix}-{self._turn}"
        return state

    def _retrieve(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """retrieve — query memory with the user input (Req 20.2)."""
        state["evidence"] = self.memory.query(state["user_message"], top_k=self.top_k)
        return state

    def _answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """answer — render a response from the evidence (P1–P5 when available)."""
        state["answer"] = self.answer_policy.render(
            state["evidence"], high_stakes=self.high_stakes
        )
        return state

    def _extract(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """extract — derive candidate new memory from the turn content."""
        state["extracted_memory"] = self.extract_fn(
            state["user_message"], state["evidence"]
        )
        return state

    def _validate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """validate — decide whether the turn yielded memory worth persisting.

        This is a cheap pre-check; the authoritative validation/quarantine
        decision happens inside the write pipeline at commit (Req 20.1).
        """
        memory_text = state.get("extracted_memory")
        state["should_commit"] = bool(memory_text and memory_text.strip())
        return state

    def _commit(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """commit — write new memory with the turn's ``source_ref`` (Req 20.3)."""
        if state.get("should_commit"):
            state["write_result"] = self.memory.write(
                state["extracted_memory"], state["source_ref"]
            )
            state["committed"] = True
        else:
            state["write_result"] = None
            state["committed"] = False
        return state
