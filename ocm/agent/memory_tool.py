"""MemoryTool — the single seam between an agent and OCM (Req 20.1, 20.2, 20.3).

The :class:`MemoryTool` is the only object the :class:`~ocm.agent.loop.AgentLoop`
(task 16.1) talks to. It maps 1:1 onto the two memory operations the agent needs:

* :meth:`query` → the Retrieval Pipeline (R0→R4), returning an
  :class:`~ocm.retrieval.evidence_packager.EvidencePackage` (Req 20.2).
* :meth:`write` → the Write Pipeline (W1→W8), returning a
  :class:`~ocm.memory.write_pipeline.WriteResult` (Req 20.3).

Because the tool delegates to the wired :class:`~ocm.core.container.CoreContainer`
pipelines — the very same objects behind the HTTP endpoints (``POST /memory/query``
and ``POST /memory/write``) — the agent can run either in-process (direct
container) or over HTTP without any code change. This keeps OCM pluggable: the
loop never imports storage, graph, or retrieval internals (Req 20.1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from ocm.ontology.enums import WriteIntent
from ocm.retrieval.evidence_packager import EvidencePackage

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from ocm.core.container import CoreContainer
    from ocm.memory.write_pipeline import WriteResult


class MemoryTool:
    """The agent-facing memory interface over a wired :class:`CoreContainer`.

    The tool holds no state of its own; it forwards to the container's
    ``retrieval_pipeline`` and ``write_pipeline`` so the agent depends only on
    this thin seam (Req 20.1).
    """

    def __init__(self, container: "CoreContainer") -> None:
        """Bind the tool to a wired container.

        Args:
            container: The :class:`CoreContainer` holding the wired
                ``retrieval_pipeline`` (R0–R4) and ``write_pipeline`` (W1–W8).
        """
        self.container = container

    # ------------------------------------------------------------------ #
    # Read path (Req 20.2)
    # ------------------------------------------------------------------ #
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        include_conflicts: bool = False,
    ) -> EvidencePackage:
        """Retrieve memory for ``query_text`` (Req 20.2).

        Runs the full retrieval pipeline (R0 classify → R1 symbolic →
        R2 semantic → R3 rerank → R4 package) and returns the structured
        evidence the agent reasons over.

        Args:
            query_text: The natural-language query (typically the user input).
            top_k: Number of nearest semantic items to retrieve.
            include_conflicts: Force inclusion of quarantined items even for a
                non-conflict query.

        Returns:
            The assembled :class:`EvidencePackage`.
        """
        return self.container.retrieval_pipeline.query(
            query_text,
            top_k=top_k,
            include_conflicts=include_conflicts,
        )

    # ------------------------------------------------------------------ #
    # Write path (Req 20.3)
    # ------------------------------------------------------------------ #
    def write(
        self,
        text: str,
        source_ref: str,
        write_intent: Union[WriteIntent, str, None] = None,
    ) -> "WriteResult":
        """Write ``text`` to memory with a ``source_ref`` (Req 20.3).

        Runs the full write pipeline (W1 extract → … → W8 commit/quarantine).
        Governance (schema validation, constraints, the contradiction gate, and
        quarantine routing) happens inside the pipeline; the tool simply returns
        the aggregate outcome.

        Args:
            text: The turn content / new information to persist.
            source_ref: Provenance reference for the write (Req 20.3). The
                :class:`AgentLoop` supplies a per-turn ref.
            write_intent: Optional :class:`WriteIntent`; ``None`` lets the
                pipeline apply its default (``new_fact``).

        Returns:
            The :class:`WriteResult` (accepted / superseded / quarantined /
            rejected outcomes plus the summary).
        """
        return self.container.write_pipeline.run(
            text,
            source_ref,
            write_intent=write_intent,
        )
