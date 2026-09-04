#!/usr/bin/env python3
"""Run notebook Cell 7f locally: LongMemEval Arm B / end-to-end.

This mirrors OCM_Colab.ipynb Cell 7f, but uses local filesystem defaults and
persistent caches. The real Qwen fact extractor reads the full LongMemEval
haystack file and emits noisy memory facts; an optional Qwen slot-linking pass
can canonicalize paraphrased attributes before governance is evaluated.

Examples:
    python run_7f_local.py --e2e-limit 5 --abst-limit 5 --embeddings deterministic
    python run_7f_local.py
    python run_7f_local.py --full
    python run_7f_local.py --full --baselines B0,B2,Bsup,B3 --slot-linker qwen
    python run_7f_local.py --full --legacy-cache-keys  # deliberate old artifacts
    python run_7f_local.py --full --manager memgpt   # MemGPT-style Bmemgpt row
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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


def _json_digest(value: Any, *, length: int = 16) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def _text_sha256(value: str, *, length: int | None = 16) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def _file_sha256(path: Path, *, length: int | None = 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    return digest if length is None else digest[:length]


def _git_revision(repo_dir: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _git_diff_sha256(repo_dir: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "diff", "--binary", "HEAD"],
            check=True,
            capture_output=True,
        )
    except Exception:
        return "unknown"
    if not proc.stdout:
        return "clean"
    return hashlib.sha256(proc.stdout).hexdigest()


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
        namespace: dict[str, Any] | None = None,
        flush_every: int = 50,
        legacy_keys: bool = False,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, str] = {}
        self.namespace = namespace or {}
        self.legacy_keys = bool(legacy_keys)
        self.unversioned_cache = False
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                meta = raw.get("__meta__")
                if isinstance(meta, dict):
                    stored_mode = meta.get("key_mode")
                    current_mode = self._key_mode
                    if stored_mode and stored_mode != current_mode:
                        raise ValueError(
                            "Prompt cache key mode mismatch.\n"
                            f"  cache:   {stored_mode}\n"
                            f"  current: {current_mode}\n"
                            f"  file:    {self.path}"
                        )
                    stored_namespace = meta.get("namespace")
                    if (
                        stored_namespace
                        and self.namespace
                        and stored_namespace != self.namespace
                    ):
                        raise ValueError(
                            "Prompt cache was produced by a different run identity.\n"
                            f"  cache:   {stored_namespace}\n"
                            f"  current: {self.namespace}\n"
                            f"  file:    {self.path}\n"
                            "Use a different cache path, delete the cache, or "
                            "pass --legacy-cache-keys only for deliberate old "
                            "md5(prompt) reuse."
                        )
                    entries = raw.get("entries") or {}
                    if isinstance(entries, dict):
                        self.cache = {str(k): str(v) for k, v in entries.items()}
                elif self.legacy_keys:
                    self.cache = {str(k): str(v) for k, v in raw.items()}
                    self.unversioned_cache = True
                else:
                    raise ValueError(
                        "Prompt cache predates identity metadata and would be "
                        "unsafe to reuse by default.\n"
                        f"  file: {self.path}\n"
                        "Use a new cache path, delete the old cache, or pass "
                        "--legacy-cache-keys to intentionally reproduce the old "
                        "md5(prompt)-only behavior."
                    )
        self._chat_factory = chat_factory
        self._chat = None
        self._dirty = 0
        self.flush_every = max(1, int(flush_every))

    @property
    def _key_mode(self) -> str:
        return "legacy-md5-prompt" if self.legacy_keys else "identity-sha256-v1"

    def _key(self, prompt: str) -> str:
        if self.legacy_keys:
            return hashlib.md5(prompt.encode("utf-8")).hexdigest()
        return _json_digest(
            {"namespace": self.namespace, "prompt": prompt},
            length=64,
        )

    def __call__(self, prompt: str) -> str:
        key = self._key(prompt)
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
            json.dump(
                {
                    "__meta__": {
                        "format": 2,
                        "key_mode": self._key_mode,
                        "namespace": self.namespace,
                        "n_entries": len(self.cache),
                    },
                    "entries": self.cache,
                },
                fh,
                ensure_ascii=False,
            )
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
    parser.add_argument("--baselines", default="B0,B2,Bsup,Bevi,B3")
    parser.add_argument(
        "--manager",
        choices=("governed", "memgpt"),
        default="governed",
        help=(
            "Write manager. 'governed' runs the OCMR baselines given by "
            "--baselines. 'memgpt' runs the MemGPT-style LLM-managed baseline "
            "(Bmemgpt): the same extraction/retrieval, but each per-fact memory "
            "edit is an LLM insert/update/skip decision with no OCMR governance, "
            "so a wrong 'insert' on a changed fact leaves a durable violation. "
            "Knowledge-update only; abstention is skipped."
        ),
    )
    parser.add_argument("--intent-mode", choices=("auto", "new_fact"), default="auto")
    parser.add_argument(
        "--extract-prompt",
        choices=("durable", "longmemeval", "generic"),
        default="longmemeval",
        help=(
            "Fact extraction prompt. 'durable' is the original user-profile "
            "extractor; 'longmemeval' also extracts counts, schedules, events, "
            "third-party facts, and yes/no memory facts."
        ),
    )
    parser.add_argument(
        "--slot-linker",
        choices=("none", "deterministic", "qwen"),
        default="qwen",
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
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=None,
        help=(
            "Generation budget for cache misses. Defaults to 256 for the old "
            "durable prompt and 512 for the LongMemEval/generic memory prompts."
        ),
    )
    parser.add_argument(
        "--legacy-cache-keys",
        action="store_true",
        help=(
            "Deliberately use the old md5(prompt)-only prompt cache keys. This "
            "is only for reproducing legacy artifacts; new paper runs should "
            "leave it off."
        ),
    )
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
        FACT_EXTRACTION_PROMPTS,
        build_fact_extract_fn,
        build_memgpt_decide_fn,
        build_slot_link_fn,
        evaluate_abstention_e2e,
        load_longmemeval,
        run_longmemeval_e2e,
        run_longmemeval_memgpt,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    llm_max_tokens = (
        args.llm_max_tokens
        if args.llm_max_tokens is not None
        else (512 if args.extract_prompt in ("longmemeval", "generic") else 256)
    )
    seeds = _parse_seeds(args.seeds)
    baselines = _parse_csv(args.baselines)
    lme_s_path = data_dir / "longmemeval_s.json"
    _download_if_missing(lme_s_path, LME_S_URL)

    dataset_sha256 = _file_sha256(lme_s_path, length=None)
    extract_prompt_sha256 = _text_sha256(
        FACT_EXTRACTION_PROMPTS[args.extract_prompt],
        length=None,
    )
    code_revision = _git_revision(repo_dir)
    code_diff_sha256 = _git_diff_sha256(repo_dir)
    cache_base_identity: dict[str, Any] = {
        "dataset": "longmemeval_s",
        "dataset_sha256": dataset_sha256,
        "extract_backend": args.extract_backend,
        "llm_model": args.llm_model,
        "llm_max_tokens": llm_max_tokens,
        "decoding": {"temperature": 0, "do_sample": False},
        "system_prompt_sha256": _text_sha256(SYSTEM_PROMPT, length=None),
        "code_revision": code_revision,
        "code_diff_sha256": code_diff_sha256,
    }
    extract_cache_identity = {
        **cache_base_identity,
        "task": "fact_extraction",
        "extract_prompt": args.extract_prompt,
        "extract_prompt_sha256": extract_prompt_sha256,
    }
    link_cache_identity = {
        **cache_base_identity,
        "task": "slot_linking",
        "slot_linker": args.slot_linker,
        "slot_link_prompt": "longmemeval-slot-link-v1",
    }
    cache_suffix = (
        ""
        if args.extract_prompt == "durable"
        else f"_{_safe_segment(args.extract_prompt)}"
    )
    extract_cache_id = _json_digest(extract_cache_identity, length=12)
    link_cache_id = _json_digest(link_cache_identity, length=12)
    if args.legacy_cache_keys:
        extract_cache_name = f"lme_e2e_extract_cache{cache_suffix}.json"
        link_cache_name = f"lme_e2e_link_cache{cache_suffix}.json"
    else:
        extract_cache_name = (
            f"lme_e2e_extract_cache{cache_suffix}__{extract_cache_id}.json"
        )
        link_cache_name = f"lme_e2e_link_cache{cache_suffix}__{link_cache_id}.json"
    extract_cache = (args.extract_cache or output_dir / extract_cache_name).resolve()
    link_cache = (args.link_cache or output_dir / link_cache_name).resolve()

    link_segment = ""
    if args.slot_linker != "none":
        link_segment = (
            f"__link_{_safe_segment(args.slot_linker)}"
            f"_t{int(round(args.link_threshold * 100))}"
        )
    manager_segment = "" if args.manager == "governed" else "__mgr_memgpt"
    run_identity: dict[str, Any] = {
        "dataset": "longmemeval_s",
        "dataset_sha256": dataset_sha256,
        "extract_prompt": args.extract_prompt,
        "extract_prompt_sha256": extract_prompt_sha256,
        "extract_backend": args.extract_backend,
        "llm_model": args.llm_model,
        "llm_max_tokens": llm_max_tokens,
        "slot_linker": args.slot_linker,
        "link_threshold": args.link_threshold,
        "intent_mode": args.intent_mode,
        "knowledge_update_limit": e2e_limit,
        "abstention_limit": abst_limit,
        "baselines": baselines,
        "seeds": seeds,
        "embeddings": args.embeddings,
        "manager": args.manager,
        "code_revision": code_revision,
        "code_diff_sha256": code_diff_sha256,
        "legacy_cache_keys": args.legacy_cache_keys,
    }
    run_fingerprint = _json_digest(run_identity, length=12)
    if args.checkpoint_dir is not None:
        checkpoint_root = args.checkpoint_dir.resolve()
    else:
        run_segment = (
            f"extract_{args.extract_prompt}__intent_{args.intent_mode}"
            f"__ku_{_limit_segment(e2e_limit)}"
            f"__abs_{_limit_segment(abst_limit)}"
            f"{link_segment}{manager_segment}__run_{run_fingerprint}"
        )
        checkpoint_root = (
            output_dir / "checkpoints" / "qwen_e2e" / _safe_segment(run_segment)
        ).resolve()

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_fingerprint": run_fingerprint,
        "run_identity": run_identity,
        "extract_cache": str(extract_cache),
        "extract_cache_identity": extract_cache_identity,
        "link_cache": str(link_cache),
        "link_cache_identity": link_cache_identity,
        "checkpoint_root": str(checkpoint_root),
    }
    manifest_path = checkpoint_root / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"LongMemEval (full haystack): {lme_s_path}")
    print(f"extract prompt: {args.extract_prompt}; max tokens: {llm_max_tokens}")
    print(f"run fingerprint: {run_fingerprint}")
    print(f"checkpoint root: {checkpoint_root}")
    print(f"run manifest: {manifest_path}")
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
                max_tokens=llm_max_tokens,
            )
        return _build_transformers_chat_fn(
            model_id=args.llm_model,
            max_new_tokens=llm_max_tokens,
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
        namespace=extract_cache_identity,
        flush_every=args.flush_every,
        legacy_keys=args.legacy_cache_keys,
    )
    if cached_chat.unversioned_cache:
        print(
            "WARNING: extraction cache is legacy md5(prompt)-only; "
            "do not cite as a new reproducible paper run."
        )
    fact_extract_fn = build_fact_extract_fn(
        cached_chat,
        prompt_template=FACT_EXTRACTION_PROMPTS[args.extract_prompt],
    )
    link_cached_chat = None
    slot_link_fn = None
    if args.slot_linker == "deterministic":
        slot_link_fn = build_slot_link_fn()
    elif args.slot_linker == "qwen":
        link_cached_chat = CachedChat(
            link_cache,
            shared_chat_factory,
            namespace=link_cache_identity,
            flush_every=args.flush_every,
            legacy_keys=args.legacy_cache_keys,
        )
        if link_cached_chat.unversioned_cache:
            print(
                "WARNING: slot-link cache is legacy md5(prompt)-only; "
                "do not cite as a new reproducible paper run."
            )
        slot_link_fn = build_slot_link_fn(
            link_cached_chat,
            confidence_threshold=args.link_threshold,
        )

    # MemGPT-style per-fact memory decisions (insert/update/skip). Uses the
    # same shared Qwen backend but a distinct disk cache so decisions are reused
    # across seeds and never collide with the extraction/link caches.
    decide_cached_chat = None
    memgpt_decide_fn = None
    if args.manager == "memgpt":
        decide_cache_identity = {
            **cache_base_identity,
            "task": "memgpt_decision",
            "extract_prompt": args.extract_prompt,
        }
        decide_cache_id = _json_digest(decide_cache_identity, length=12)
        decide_cache_name = (
            f"lme_e2e_memgpt_decide_cache_{_safe_segment(args.extract_prompt)}.json"
            if args.legacy_cache_keys
            else (
                f"lme_e2e_memgpt_decide_cache_{_safe_segment(args.extract_prompt)}"
                f"__{decide_cache_id}.json"
            )
        )
        decide_cache = (
            output_dir / decide_cache_name
        ).resolve()
        decide_cached_chat = CachedChat(
            decide_cache,
            shared_chat_factory,
            namespace=decide_cache_identity,
            flush_every=args.flush_every,
            legacy_keys=args.legacy_cache_keys,
        )
        if decide_cached_chat.unversioned_cache:
            print(
                "WARNING: MemGPT decision cache is legacy md5(prompt)-only; "
                "do not cite as a new reproducible paper run."
            )
        memgpt_decide_fn = build_memgpt_decide_fn(decide_cached_chat)
        print(f"manager: MemGPT-style (Bmemgpt); decision cache: {decide_cache}")
        if not args.skip_abstention:
            print(
                "note: --manager memgpt is a write-policy baseline for updates; "
                "skipping abstention (governed B0/B2/B3 already cover it)."
            )
            args.skip_abstention = True

    if args.embeddings == "deterministic":
        embeddings = DeterministicEmbeddingProvider()
    else:
        embeddings = LocalEmbeddingProvider()

    result: dict[str, Any] = {"_run_manifest": manifest}

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
            if args.manager == "memgpt":
                e2e_report = run_longmemeval_memgpt(
                    instances,
                    fact_extract_fn,
                    memgpt_decide_fn,
                    extract_prompt_name=args.extract_prompt,
                    slot_link_fn=slot_link_fn,
                    slot_linker_name=args.slot_linker,
                    seeds=seeds,
                    embeddings=embeddings,
                    checkpoint_dir=str(checkpoint_root / "knowledge_update"),
                )
            else:
                e2e_report = run_longmemeval_e2e(
                    instances,
                    fact_extract_fn,
                    intent_mode=args.intent_mode,
                    extract_prompt_name=args.extract_prompt,
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
                extract_prompt_name=args.extract_prompt,
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
        if decide_cached_chat is not None:
            decide_cached_chat.flush()
        manifest["extract_cache_entries"] = len(cached_chat.cache)
        if link_cached_chat is not None:
            manifest["link_cache_entries"] = len(link_cached_chat.cache)
        if decide_cached_chat is not None:
            manifest["memgpt_decision_cache_entries"] = len(decide_cached_chat.cache)
        if "knowledge_update" in result:
            manifest["knowledge_update_examples_fingerprint"] = result[
                "knowledge_update"
            ].get("examples_fingerprint")
        if "abstention" in result:
            manifest["abstention_examples_fingerprint"] = result["abstention"].get(
                "examples_fingerprint"
            )
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
        print(f"extraction cache entries: {len(cached_chat.cache)} -> {extract_cache}")
        if link_cached_chat is not None:
            print(f"slot-link cache entries: {len(link_cached_chat.cache)} -> {link_cache}")
        if decide_cached_chat is not None:
            print(
                f"memgpt decision cache entries: {len(decide_cached_chat.cache)} "
                f"-> {decide_cached_chat.path}"
            )

    out_path = output_dir / "results_longmemeval_e2e.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
