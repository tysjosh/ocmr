"""Fail-fast wrapper that separates environment faults from model faults (W1).

Why this exists
---------------
:class:`~ocm.memory.write_pipeline.WritePipeline` deliberately treats an
:class:`~ocm.extraction.base.ExtractionError` as a *recorded validation failure*
(Req 3.3): it logs a warning, records zero candidates, and carries on. That is
correct when the **model** misbehaves — a local model wrapping JSON in prose is a
normal, measurable outcome that belongs in the results.

It is dangerous when the **environment** is broken. A missing CUDA library, an
unbuildable Triton kernel, or an OOM makes *every* ``generate`` call raise. The
pipeline then swallows all of them, every write produces no candidates, and the
harness prints a complete, plausible-looking table computed over an empty memory
store. The arms would appear to differ only by governance while in fact no arm
had any memory at all.

So the two cases need different handling, and the distinction is available:
:class:`~ocm.extraction.transformers_extractor.TransformersExtractor` re-raises
any non-``ExtractionError`` exception from generation as
``"transformers generation failed: ..."``, whereas parse and schema failures
carry their own distinct prefixes. This wrapper routes on that:

* **Environment fault** -> raise :class:`ExtractionEnvironmentError`, which is a
  ``RuntimeError`` and *not* an ``ExtractionError``, so the pipeline's ``except``
  clause does not catch it and the run aborts loudly at the first occurrence.
* **Model fault** -> re-raise the original ``ExtractionError`` unchanged, so
  existing recorded-failure semantics are preserved, and count it so the harness
  can report how often it happened.

Wiring
------
Sits between the model and the cache, so failures are never memoized::

    base   = TransformersExtractor(model=mdl, tokenizer=tok)
    strict = StrictExtractor(base)
    extractor = CachingExtractor(strict, cache_path=...)
"""

from __future__ import annotations

from typing import Any

from ocm.extraction.base import ExtractionError
from ocm.memory.contracts import ExtractionResult

#: Message prefix that ``TransformersExtractor`` uses for runtime/generation
#: faults, as opposed to parse or schema faults.
_ENVIRONMENT_MARKER = "generation failed"

#: Substrings that identify an environment fault even inside a generic message.
#: These are the failure modes that make *every* subsequent call fail too, so
#: continuing the run would only produce more empty extractions.
_ENVIRONMENT_SIGNATURES: tuple[str, ...] = (
    "Python.h",
    "CalledProcessError",
    "cuda_utils",
    "CUDA out of memory",
    "CUDA error",
    "libcuda",
    "no kernel image is available",
    "Triton",
    "triton",
)

_REMEDIES = """
This is an environment fault, not a model behaviour, so the run was aborted
rather than continuing with empty extractions.

If the cause is a missing Python.h, Triton cannot JIT-compile its CUDA helper
because the interpreter's development headers are absent. Fix whichever applies:

  1. Install the headers (needs root; matches your interpreter version):
       sudo apt-get install -y python3.12-dev
     Verify with:
       python -c "import sysconfig,os;p=sysconfig.get_paths()['include'];print(p, os.path.exists(p+'/Python.h'))"
     That must print True before re-running.

  2. No root? Use an interpreter that ships its own headers, then reinstall:
       uv python install 3.12
       uv venv --python 3.12 .venv && . .venv/bin/activate
       pip install -r requirements.txt 'transformers>=4.45' accelerate
     (a conda env works equally well)

Re-run the identical command afterwards. The extraction cache is keyed on
(source_ref, text) and persists to disk, so nothing already extracted is redone.
""".strip()


class ExtractionEnvironmentError(RuntimeError):
    """An extraction failed for an environmental reason, not a model one.

    Deliberately **not** a subclass of :class:`ExtractionError`: the write
    pipeline catches that type and degrades to a recorded failure, which is
    exactly the behaviour we need to bypass here.
    """


def classify(message: str) -> str:
    """Return ``"environment"`` or ``"model"`` for an ``ExtractionError`` message.

    Args:
        message: The ``str`` of the raised :class:`ExtractionError`.

    Returns:
        ``"environment"`` when the failure looks like broken tooling, drivers, or
        exhausted memory — conditions that will recur on every subsequent call —
        and ``"model"`` when it looks like an unparseable or schema-invalid
        generation, which is a legitimate per-input outcome.
    """
    if _ENVIRONMENT_MARKER in message:
        return "environment"
    return (
        "environment"
        if any(sig in message for sig in _ENVIRONMENT_SIGNATURES)
        else "model"
    )


class StrictExtractor:
    """Aborts the run on environment faults; counts model faults and passes them on.

    Parameters
    ----------
    base:
        The wrapped extractor, exposing ``extract(text, source_ref)``.
    tolerate_environment_errors:
        When ``True``, environment faults are counted and re-raised as ordinary
        :class:`ExtractionError`, restoring the degrade-and-continue behaviour.
        Off by default; only useful for deliberately testing the degraded path.
    """

    def __init__(self, base: Any, *, tolerate_environment_errors: bool = False) -> None:
        self._base = base
        self.version = getattr(base, "version", "strict-extractor")
        #: Forwarded so a cache further out can still see the model's identity.
        self.fingerprint = getattr(base, "fingerprint", None)
        self.tolerate_environment_errors = tolerate_environment_errors
        self.calls = 0
        self.model_failures = 0
        self.environment_failures = 0
        #: Distinct inputs seen and distinct inputs that failed. A failing input
        #: is re-extracted by every arm unless the cache memoizes failures, so
        #: ``model_failures / calls`` counts retries and overstates how much of
        #: the corpus is actually unparseable. These sets give the honest ratio.
        self._seen: set[str] = set()
        self._failed: set[str] = set()
        #: First few model-fault messages, for the end-of-run report.
        self.model_failure_examples: list[str] = []

    @staticmethod
    def _key(text: str, source_ref: str) -> str:
        return f"{source_ref}\x00{text}"

    def extract(self, text: str, source_ref: str) -> ExtractionResult:
        """Extract, routing failures by cause."""
        self.calls += 1
        key = self._key(text, source_ref)
        self._seen.add(key)
        try:
            return self._base.extract(text, source_ref)
        except ExtractionError as exc:
            if classify(str(exc)) == "environment":
                self.environment_failures += 1
                if not self.tolerate_environment_errors:
                    raise ExtractionEnvironmentError(
                        f"W1 extraction hit an environment fault on {source_ref!r} "
                        f"after {self.calls} call(s): {exc}\n\n{_REMEDIES}"
                    ) from exc
            else:
                self.model_failures += 1
                if key not in self._failed:
                    self._failed.add(key)
                    if len(self.model_failure_examples) < 5:
                        self.model_failure_examples.append(f"{source_ref}: {exc}")
            raise

    @property
    def stats(self) -> dict[str, Any]:
        """Call and failure counters for the end-of-run report.

        ``unparseable_input_rate`` is the figure to quote: it is over *distinct*
        inputs, so it is not inflated by per-arm retries of the same failure.
        """
        n_inputs = len(self._seen)
        return {
            "calls": self.calls,
            "distinct_inputs": n_inputs,
            "distinct_unparseable_inputs": len(self._failed),
            "unparseable_input_rate": (len(self._failed) / n_inputs) if n_inputs else 0.0,
            "model_failures": self.model_failures,
            "environment_failures": self.environment_failures,
            "model_failure_examples": list(self.model_failure_examples),
        }
