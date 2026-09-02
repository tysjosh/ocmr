"""Run identity: fold the extraction stack's fingerprint into checkpoint keys.

Two caches sit behind every LLM-driven evaluation run, and until now only one of
them was protected.

The **extraction cache**
(:class:`~ocm.extraction.caching_extractor.CachingExtractor`) is already safe: it
persists the wrapped extractor's ``fingerprint`` under ``__meta__`` and *raises*
on load when the stored fingerprint differs from the current one, because its
entry keys are only ``(source_ref, text)`` and would otherwise serve generations
from a different model.

The **result checkpoints** written by
:func:`~ocm.evaluation.experiment.run_multiseed` were not. Their keys are
``ms__{method}__seed{seed}__pc{per_category}{key_suffix}`` — the *experiment*
configuration, with nothing identifying the extraction stack that produced the
writes. So after changing model, token budget, or prompt, the extraction cache
correctly refuses to load (or is pointed at a fresh path) and re-extracts, while
the per-``(method, seed)`` results from the previous stack are silently reused.
The run then reports a mixture of two configurations, which is the failure mode
behind the inconsistent LongMemEval Arm-B artifacts (see
``docs/evaluation_methodology.md``).

This module builds a short digest of the extraction stack and formats it as a
``key_suffix`` fragment, so a changed stack recomputes instead of resuming onto
stale results. It degrades gracefully: an extractor exposing no identity yields a
digest of its class name, which is still stable and still distinguishes a mock
run from a real one.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional

__all__ = [
    "json_digest",
    "extractor_identity",
    "embeddings_identity",
    "run_identity",
    "run_fingerprint",
    "fingerprint_suffix",
]


def json_digest(value: Any, *, length: int = 12) -> str:
    """Stable short digest of any JSON-encodable value.

    Keys are sorted and separators normalized so logically equal identities
    produce equal digests regardless of construction order.
    """
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return digest[:length] if length else digest


def _unwrap(component: Any) -> Any:
    """Follow ``_base`` decorator chains to the innermost component.

    ``CachingExtractor`` and ``StrictExtractor`` both wrap another extractor; the
    identity that matters is the one that actually generates.
    """
    seen: set[int] = set()
    current = component
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        base = getattr(current, "_base", None)
        if base is None:
            break
        current = base
    return current


def extractor_identity(extractor: Any) -> dict[str, Any]:
    """Everything about ``extractor`` that changes what it emits.

    Prefers an explicit ``fingerprint`` mapping (``TransformersExtractor`` exposes
    model id, token budget, and prompt hash). Falls back to ``version``, then to
    the class name — always returning something stable rather than nothing, so a
    missing fingerprint cannot silently collapse two stacks onto one key.
    """
    if extractor is None:
        return {"extractor": None}

    for candidate in (extractor, _unwrap(extractor)):
        value = getattr(candidate, "fingerprint", None)
        if isinstance(value, Mapping) and value:
            return {"extractor": dict(value)}

    inner = _unwrap(extractor)
    identity: dict[str, Any] = {"extractor_class": type(inner).__name__}
    version = getattr(inner, "version", None)
    if version:
        identity["extractor_version"] = str(version)
    for attr in ("model_id", "max_new_tokens"):
        attr_value = getattr(inner, attr, None)
        if attr_value is not None:
            identity[attr] = attr_value
    return {"extractor": identity}


def embeddings_identity(embeddings: Any) -> dict[str, Any]:
    """Identity of the embedding provider (retrieval changes what is scored)."""
    if embeddings is None:
        return {"embeddings": None}
    inner = _unwrap(embeddings)
    identity: dict[str, Any] = {"embeddings_class": type(inner).__name__}
    for attr in ("model_name", "dim"):
        value = getattr(inner, attr, None)
        if value is not None:
            identity[attr] = value
    return {"embeddings": identity}


def run_identity(
    *,
    extractor: Any = None,
    embeddings: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    """Assemble the full identity mapping for a run.

    ``extra`` carries anything the caller knows that is not introspectable from
    the components themselves — dataset digest, prompt variant, code revision.
    """
    identity: dict[str, Any] = {}
    identity.update(extractor_identity(extractor))
    identity.update(embeddings_identity(embeddings))
    for key, value in extra.items():
        if value is not None:
            identity[key] = value
    return identity


def run_fingerprint(
    *,
    extractor: Any = None,
    embeddings: Any = None,
    length: int = 12,
    **extra: Any,
) -> str:
    """Short digest of the extraction stack, for use as a checkpoint-key fragment."""
    return json_digest(
        run_identity(extractor=extractor, embeddings=embeddings, **extra),
        length=length,
    )


def fingerprint_suffix(fingerprint: Optional[str]) -> str:
    """Format a fingerprint as a checkpoint ``key_suffix`` fragment.

    Returns ``""`` for ``None`` so existing checkpoints stay addressable when no
    fingerprint is supplied — passing one is opt-in, and omitting it reproduces
    the previous key layout exactly.
    """
    if not fingerprint:
        return ""
    return f"__fp{fingerprint}"
