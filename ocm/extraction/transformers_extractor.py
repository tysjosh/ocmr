"""In-process Hugging Face ``transformers`` extractor (W1).

``TransformersExtractor`` runs a local instruction-tuned model (e.g.
``Qwen/Qwen2.5-32B-Instruct``) **in the same process** via ``model.generate`` \u2014
no HTTP server, no vLLM. It reuses the exact JSON extraction prompt from
:mod:`ocm.extraction.llm_extractor`, so it is a drop-in alternative to the
OpenAI-compatible :class:`~ocm.extraction.llm_extractor.LLMExtractor` for fully
local runs (Colab/A100, on-prem GPUs).

Wiring
------
The model + tokenizer are constructed by the caller (so this module never
imports ``torch`` / ``transformers`` and stays import-safe in hermetic tests),
then injected into the container::

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
    mdl = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-32B-Instruct", torch_dtype="bfloat16", device_map="auto")

    from ocm.extraction.transformers_extractor import TransformersExtractor
    from ocm.core.container import CoreContainer
    from ocm.core.config import Settings

    extractor = TransformersExtractor(model=mdl, tokenizer=tok)
    container = CoreContainer(
        Settings(deterministic_test_mode=True, chroma_mode="memory"),
        extractor=extractor,
    )

Robustness
----------
Local models often wrap JSON in prose or Markdown fences, so the raw generation
is parsed leniently (the first balanced ``{...}`` object is extracted). A failure
to produce schema-valid JSON raises :class:`~ocm.extraction.base.ExtractionError`,
which the write pipeline turns into a recorded validation failure (Req 3.3).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from pydantic import ValidationError

from ocm.extraction.base import ExtractionError
from ocm.extraction.llm_extractor import SYSTEM_PROMPT
from ocm.memory.contracts import ExtractionResult


def _loads_lenient(content: str) -> dict:
    """Parse the first balanced JSON object out of a model generation.

    Tolerates Markdown code fences and surrounding prose by extracting the
    substring from the first ``{`` to the matching last ``}`` and ``json.loads``-ing
    it. Raises :class:`ExtractionError` when no valid JSON object is present.
    """
    if not isinstance(content, str) or not content.strip():
        raise ExtractionError("transformers extractor returned empty output")
    text = content.strip()
    # Strip a leading ```json / ``` fence if present.
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ExtractionError("transformers extractor output contained no JSON object")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"transformers extractor produced invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ExtractionError("transformers extractor JSON was not an object")
    return data


class TransformersExtractor:
    """W1 extractor that generates with a local HF model in-process.

    Parameters
    ----------
    model, tokenizer:
        A loaded ``transformers`` causal-LM and its tokenizer. Required unless a
        ``complete`` callable is supplied.
    complete:
        Optional ``complete(messages: list[dict]) -> str`` override that returns
        the model's text completion for a chat-message list. When omitted a
        default implementation drives ``tokenizer.apply_chat_template`` +
        ``model.generate``. Supplying it keeps the class unit-testable without a
        real model.
    max_new_tokens:
        Generation budget for the JSON output.
    system_prompt:
        The extraction system prompt (defaults to the shared
        :data:`~ocm.extraction.llm_extractor.SYSTEM_PROMPT`).
    version:
        Provenance tag recorded as ``extractor_version``.
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        *,
        complete: Optional[Callable[[list[dict]], str]] = None,
        max_new_tokens: int = 1024,
        system_prompt: str = SYSTEM_PROMPT,
        version: str = "qwen-transformers-v1",
    ) -> None:
        if complete is None and (model is None or tokenizer is None):
            raise ValueError(
                "TransformersExtractor needs either a `complete` callable or both "
                "`model` and `tokenizer`."
            )
        self.model = model
        self.tokenizer = tokenizer
        self._complete = complete or self._default_complete
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.version = version

    # -- public API --------------------------------------------------------
    def extract(self, text: str, source_ref: str) -> ExtractionResult:
        """Extract candidate memory items from ``text`` via local generation."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"source_ref={source_ref}\n<<<{text}>>>"},
        ]
        try:
            content = self._complete(messages)
        except ExtractionError:
            raise
        except Exception as exc:  # generation/runtime errors
            raise ExtractionError(f"transformers generation failed: {exc!r}") from exc

        data = _loads_lenient(content)
        data.setdefault("extractor_version", self.version)
        try:
            return ExtractionResult.model_validate(data)
        except ValidationError as exc:
            raise ExtractionError(
                f"transformers extractor output failed validation: {exc}"
            ) from exc

    # -- default HF generation --------------------------------------------
    def _default_complete(self, messages: list[dict]) -> str:
        """Drive ``apply_chat_template`` + ``model.generate`` (greedy decoding)."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
        )
        # Drop the prompt tokens; decode only the newly generated continuation.
        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
