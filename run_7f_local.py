#!/usr/bin/env python3
"""Run notebook Cell 7f locally: LongMemEval Arm B / end-to-end.

This mirrors OCM_Colab.ipynb Cell 7f, but uses local filesystem defaults and
persistent caches. The real Qwen fact extractor reads the full LongMemEval
haystack file and emits noisy durable facts; an optional Qwen slot-linking pass
can canonicalize paraphrased attributes before governance is evaluated.

Examples:
    python run_7f_local.py --e2e-limit 5 --abst-limit 5 --embeddings deterministic
    python run_7f_local.py
    python run_7f_local.py --full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Optional


LME_S_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
    "resolve/main/longmemeval_s_cleaned.json"
)

SYSTEM_PROMPT = "You extract structured data and output only JSON."


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _parse_csv(value))


def _parse_limit(value: str | None) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("none", "full", "all", "-1"):
        return None
    n = int(text)
    if n < 0:
        return None
    return n


def _limit_segment(value: Optional[int]) -> str:
    return "full" if value is None else f"limit_{value}"


def _safe_segment(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return cleaned.strip("_") or "run"


def _ci(row: dict, key: str) -> str:
    x = row[key]
    return f"{x['mean']:.1f} [{x['low']:.1f},{x['high']:.1f}]"


def _download_if_missing(path: Path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, path)


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _build_openai_chat_fn(
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    timeout_s: float,
    max_tokens: int,
):
    url = _chat_completions_url(base_url)

    def _chat(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected chat-completions response: {data!r}") from exc

    return _chat


def _build_transformers_chat_fn(
    *,
    model_id: str,
    max_new_tokens: int,
    allow_offload: bool,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Transformers extraction backend needs a CUDA GPU. Use an existing "
            "cache or --extract-backend openai with a remote endpoint."
        )

    print(f"loading {model_id} with transformers bf16/device_map=auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Loaded {model_id} (bf16)")

    device_map = getattr(model, "hf_device_map", {}) or {}
    offloaded = [name for name, device in device_map.items() if device in ("cpu", "disk")]
    print("offloaded modules:", offloaded or "none (all on GPU)")
    if offloaded and not allow_offload:
        raise RuntimeError(
            "Model offloaded to CPU/disk, which will make extraction very slow. "
            "Use a smaller model, reduce memory pressure, or pass --allow-offload."
        )

    def _chat(prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    return _chat


class CachedChat:
    """Disk-backed prompt cache that lazily builds the expensive chat backend."""

    def __init__(
        self,
        path: Path,
        chat_factory,
        *,
        flush_every: int = 50,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, str] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                self.cache = {str(k): str(v) for k, v in raw.items()}
        self._chat_factory = chat_factory
        self._chat = None
        self._dirty = 0
        self.flush_every = max(1, int(flush_every))

    def __call__(self, prompt: str) -> str:
        key = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        if key in self.cache:
            return self.cache[key]
        if self._chat is None:
            self._chat = self._chat_factory()
        response = self._chat(prompt)
        self.cache[key] = response
        self._dirty += 1
        if self._dirty % self.flush_every == 0:
            self.flush()
        return response

    def flush(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self.cache, fh, ensure_ascii=False)
        tmp.replace(self.path)
        self._dirty = 0


def _print_knowledge_table(report: dict[str, Any], intent_mode: str) -> None:
    print(
        f"\n=== LongMemEval knowledge-update END-TO-END "
        f"(intent_mode={intent_mode}; mean [95% CI]) ==="
    )
    print(f"{'Method':<8}{'TaskSuccess up':<22}{'Contradiction dn':<22}{'ConstraintViol dn':<22}")
    for method in report["methods"]:
        row = report["decisive_metrics"][method]
        print(
            f"{method:<8}"
            f"{_ci(row, 'task_success'):<22}"
            f"{_ci(row, 'contradiction_rate'):<22}"
            f"{_ci(row, 'constraint_violations'):<22}"
        )
    print(
        "\nwrite outcomes:",
        {m: report["write_outcomes"][m] for m in report["methods"]},
    )


def _print_abstention_table(report: dict[str, Any], intent_mode: str) -> None:
    print(
        f"\n=== LongMemEval abstention END-TO-END "
        f"(intent_mode={intent_mode}; mean [95% CI]) ==="
    )
    print(f"{'Method':<8}{'Abstention up':<22}{'False answer dn':<22}{'Support diag':<22}")
    for method in report["methods"]:
        row = report["abstention_metrics"][method]
        false_key = (
            "false_answer_rate"
            if "false_answer_rate" in row
            else "false_support_or_answer_rate"
        )
        support = (
            _ci(row, "supporting_response_rate")
            if "supporting_response_rate" in row
            else "n/a"
        )
        print(
            f"{method:<8}"
            f"{_ci(row, 'abstention_accuracy'):<22}"
            f"{_ci(row, false_key):<22}"
            f"{support:<22}"
        )
    print(
        "\nabstention counts:",
        {m: report["counts"][m] for m in report["methods"]},
    )
    print(
        "abstention write outcomes:",
        {m: report["write_outcomes"][m] for m in report["methods"]},
    )


def main() -> int:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run LongMemEval Cell 7f locally using cached real extraction."
    )
    parser.add_argument("--repo-dir", type=Path, default=repo_dir)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--extract-cache", type=Path, default=None)
    parser.add_argument("--link-cache", type=Path, default=None)
    parser.add_argument("--e2e-limit", default="30", help="Knowledge-update cap; use full/none/all for full split.")
    parser.add_argument("--abst-limit", default="30", help="Abstention cap; use full/none/all for full split.")
    parser.add_argument("--full", action="store_true", help="Set both limits to full.")
    parser.add_argument("--seeds", default="1337,7,42,99,2024")
    parser.add_argument("--baselines", default="B0,B2,B3")
    parser.add_argument("--intent-mode", choices=("auto", "new_fact"), default="auto")
    parser.add_argument(
        "--slot-linker",
        choices=("none", "deterministic", "qwen"),
        default="none",
        help="Canonicalize extracted attributes before Slot creation.",
    )
    parser.add_argument(
        "--link-threshold",
        type=float,
        default=0.75,
        help="Minimum model confidence for Qwen slot-link decisions.",
    )
    parser.add_argument(
        "--extract-backend",
        choices=("transformers", "openai"),
        default="transformers",
        help="Extraction backend for cache misses.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.environ.get("LME_E2E_BASE_URL"),
        help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1.",
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.environ.get("LME_E2E_API_KEY"),
        help="API key for the extraction endpoint, if required.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("LME_E2E_MODEL", "Qwen/Qwen2.5-14B-Instruct"),
        help="Model id/name used by the extraction backend.",
    )
    parser.add_argument("--llm-timeout-s", type=float, default=120.0)
    parser.add_argument("--llm-max-tokens", type=int, default=256)
    parser.add_argument("--allow-offload", action="store_true")
    parser.add_argument("--flush-every", type=int, default=50)
    parser.add_argument(
        "--embeddings",
        choices=("local", "deterministic"),
        default="local",
        help="Use local sentence-transformers embeddings or cheap deterministic vectors.",
    )
    parser.add_argument("--skip-knowledge-update", action="store_true")
    parser.add_argument("--skip-abstention", action="store_true")
    args = parser.parse_args()

    repo_dir = args.repo_dir.resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from ocm.evaluation.datasets.longmemeval_adapter import (
        build_fact_extract_fn,
        build_slot_link_fn,
        evaluate_abstention_e2e,
        load_longmemeval,
        run_longmemeval_e2e,
    )
    from ocm.retrieval.embeddings import (
        DeterministicEmbeddingProvider,
        LocalEmbeddingProvider,
    )

    if args.full:
        e2e_limit = None
        abst_limit = None
    else:
        e2e_limit = _parse_limit(args.e2e_limit)
        abst_limit = _parse_limit(args.abst_limit)

    data_dir = (args.data_dir or repo_dir / "data").resolve()
    output_dir = (args.output_dir or repo_dir / "local_results").resolve()
    extract_cache = (
        args.extract_cache or output_dir / "lme_e2e_extract_cache.json"
    ).resolve()
    link_cache = (
        args.link_cache or output_dir / "lme_e2e_link_cache.json"
    ).resolve()
    if args.checkpoint_dir is not None:
        checkpoint_root = args.checkpoint_dir.resolve()
    else:
        link_segment = ""
        if args.slot_linker != "none":
            link_segment = (
                f"__link_{_safe_segment(args.slot_linker)}"
                f"_t{int(round(args.link_threshold * 100))}"
            )
        run_segment = (
            f"intent_{args.intent_mode}__ku_{_limit_segment(e2e_limit)}"
            f"__abs_{_limit_segment(abst_limit)}{link_segment}"
        )
        checkpoint_root = (
            output_dir / "checkpoints" / "qwen_e2e" / _safe_segment(run_segment)
        ).resolve()
    lme_s_path = data_dir / "longmemeval_s.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    _download_if_missing(lme_s_path, LME_S_URL)
    print(f"LongMemEval (full haystack): {lme_s_path}")
    print(f"checkpoint root: {checkpoint_root}")
    print(f"extraction cache: {extract_cache}")
    if args.slot_linker == "qwen":
        print(f"slot-link cache: {link_cache}")
    elif args.slot_linker == "deterministic":
        print("slot linker: deterministic aliases")

    def chat_factory():
        if args.extract_backend == "openai":
            if not args.llm_base_url:
                raise RuntimeError(
                    "--extract-backend openai requires --llm-base-url, for example "
                    "http://localhost:8000/v1"
                )
            print(
                "using OpenAI-compatible extraction endpoint "
                f"{_chat_completions_url(args.llm_base_url)} model={args.llm_model}"
            )
            return _build_openai_chat_fn(
                base_url=args.llm_base_url,
                model=args.llm_model,
                api_key=args.llm_api_key,
                timeout_s=args.llm_timeout_s,
                max_tokens=args.llm_max_tokens,
            )
        return _build_transformers_chat_fn(
            model_id=args.llm_model,
            max_new_tokens=args.llm_max_tokens,
            allow_offload=args.allow_offload,
        )

    shared_chat: dict[str, Any] = {"fn": None}

    def shared_chat_factory():
        if shared_chat["fn"] is None:
            shared_chat["fn"] = chat_factory()
        return shared_chat["fn"]

    cached_chat = CachedChat(
        extract_cache,
        shared_chat_factory,
        flush_every=args.flush_every,
    )
    fact_extract_fn = build_fact_extract_fn(cached_chat)
    link_cached_chat = None
    slot_link_fn = None
    if args.slot_linker == "deterministic":
        slot_link_fn = build_slot_link_fn()
    elif args.slot_linker == "qwen":
        link_cached_chat = CachedChat(
            link_cache,
            shared_chat_factory,
            flush_every=args.flush_every,
        )
        slot_link_fn = build_slot_link_fn(
            link_cached_chat,
            confidence_threshold=args.link_threshold,
        )

    if args.embeddings == "deterministic":
        embeddings = DeterministicEmbeddingProvider()
    else:
        embeddings = LocalEmbeddingProvider()

    seeds = _parse_seeds(args.seeds)
    baselines = _parse_csv(args.baselines)
    result: dict[str, Any] = {}

    try:
        if not args.skip_knowledge_update:
            instances = load_longmemeval(
                str(lme_s_path),
                question_type="knowledge-update",
                limit=e2e_limit,
            )
            print(
                f"{len(instances)} knowledge-update questions "
                "(end-to-end extraction over full haystack)"
            )
            e2e_report = run_longmemeval_e2e(
                instances,
                fact_extract_fn,
                intent_mode=args.intent_mode,
                slot_link_fn=slot_link_fn,
                slot_linker_name=args.slot_linker,
                baselines=baselines,
                seeds=seeds,
                embeddings=embeddings,
                checkpoint_dir=str(checkpoint_root / "knowledge_update"),
            )
            result["knowledge_update"] = e2e_report
            _print_knowledge_table(e2e_report, args.intent_mode)

        if not args.skip_abstention:
            abst_instances = load_longmemeval(
                str(lme_s_path),
                question_type=None,
                abstention=True,
                limit=abst_limit,
            )
            print(
                f"{len(abst_instances)} abstention questions "
                "(end-to-end extraction over full haystack)"
            )
            abst_report = evaluate_abstention_e2e(
                abst_instances,
                fact_extract_fn,
                intent_mode=args.intent_mode,
                slot_link_fn=slot_link_fn,
                slot_linker_name=args.slot_linker,
                baselines=baselines,
                seeds=seeds,
                embeddings=embeddings,
                checkpoint_dir=str(checkpoint_root / "abstention"),
            )
            result["abstention"] = abst_report
            _print_abstention_table(abst_report, args.intent_mode)
    finally:
        cached_chat.flush()
        if link_cached_chat is not None:
            link_cached_chat.flush()
        print(f"extraction cache entries: {len(cached_chat.cache)} -> {extract_cache}")
        if link_cached_chat is not None:
            print(f"slot-link cache entries: {len(link_cached_chat.cache)} -> {link_cache}")

    out_path = output_dir / "results_longmemeval_e2e.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
