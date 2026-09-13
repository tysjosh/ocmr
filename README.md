# Ontology-Constrained Memory (OCM)

A **write-time-governed, ontology-constrained memory layer** for long-horizon LLM
agents. Incoming facts become typed graph assertions, are checked against an
ontology and constraints, run through a contradiction gate, and stamped with
provenance before they commit. Conflicting or low-trust writes are quarantined or
rejected rather than silently overwriting good data.

Implementation lives in [`ocm/`](ocm/). Evaluation entry points are the
`run_*.py` scripts at the repository root.

**Already-produced output is in [`results/`](results/)** — see
[`results/README.md`](results/README.md) for what each file is and which command
produces it, so the reported numbers can be inspected without re-running anything.

## Setup

Python **3.11+**.

```bash
pip install -e ".[dev]"
```

Runtime dependencies come from [`pyproject.toml`](pyproject.toml): `fastapi`,
`uvicorn`, `pydantic` v2, `networkx`, `chromadb`, `sentence-transformers`. The
`dev` extra adds `pytest`, `hypothesis`, `httpx`.

Everything runs **offline by default** — no API key, no network — using the mock
extractor and a local embedding model.

## Check the install

```bash
python -m pytest                                  # full suite, hermetic
python run_lmo_durable_state.py --fixture         # scorer on a built-in fixture
python run_multiwoz_durable_state.py --fixture    # ditto, 3 dialogues
```

The `--fixture` runs need no corpus and no downloads. They don't reproduce
reported numbers; they confirm the pipeline works before you fetch data.

Expect **9 skipped** tests on a fresh clone — they need `data/longmemeval_s.json`
(below). Skips are not failures. One test is `xfail(strict=True)` and records a
known defect deliberately.

## Get the data

`data/` is not shipped. LongMemEval comes from the public
[`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
dataset. `run_lmo_durable_state.py` fetches its corpus automatically on first use,
or fetch manually:

```bash
mkdir -p data
curl -L -o data/longmemeval_oracle.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json

# only needed for LM-R and the 9 skipped tests
curl -L -o data/longmemeval_s.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

| File | Size | Needed for |
| --- | --- | --- |
| `data/longmemeval_oracle.json` | 15 MB | LM-O |
| `data/longmemeval_s.json` | 277 MB | LM-R, and the 9 skipped tests |

The two share question ids, so pointing a script at the wrong one raises no error
— it only changes how many distractor sessions are ingested. Each result file
records `dataset`, `dataset_sha256` and `dataset_bytes` in its `_meta` so runs
cannot be confused after the fact.

MultiWOZ dialogues are fetched per split at run time and need access to
`raw.githubusercontent.com`.

### Gold trajectories for LM-O

LM-O is annotation-scoped: it needs a JSON file giving the sequence of values each
`knowledge-update` question passes through. **The 47 trajectories behind the
reported row are included** at
[`results/longmemeval_kupdate_annotations__Qwen_Qwen2.5-14B-Instruct.json`](results/longmemeval_kupdate_annotations__Qwen_Qwen2.5-14B-Instruct.json),
so no annotation pass is needed to reproduce it.

To regenerate them instead, use `annotate_file` from
[`ocm/evaluation/datasets/longmemeval_annotate.py`](ocm/evaluation/datasets/longmemeval_annotate.py),
passing a chat callable:

```python
from ocm.evaluation.datasets.longmemeval_annotate import annotate_file

annotate_file(
    "data/longmemeval_oracle.json",
    "local_results/longmemeval_kupdate_annotations__<model-slug>.json",
    chat_fn,            # prompt -> response; any instruct model
)
```

This is an LLM pass, so it needs a model and is the one part of LM-O that is not
CPU-only. Every annotation is validated against the corpus's own `answer` field
before being written. Stamp the annotating model into the filename: the
trajectories are model-generated, so mixing files produced by different models
silently blends two configurations, and `run_lmo_durable_state.py` records the
filename slug and a content digest in its output for exactly that reason.

Cell **7e** of [`OCM_Colab.ipynb`](OCM_Colab.ipynb) runs this pass end to end.

## Run the evaluations

Three of the four surfaces need **no GPU** and finish in minutes — they replay
governance decisions over oracle-supplied or cached facts, so no language model is
invoked at scoring time.

### LM-O — LongMemEval durable state

```bash
python run_lmo_durable_state.py \
  --annotations results/longmemeval_kupdate_annotations__Qwen_Qwen2.5-14B-Instruct.json \
  --data data/longmemeval_oracle.json \
  --arms B0,B2,Bsup,B3 \
  --embeddings local \
  --out local_results/lmo_durable_state.json
```

Compare against [`results/lmo_durable_state.json`](results/lmo_durable_state.json).

Keep `--embeddings local`. The `deterministic` default hashes text into vectors,
which is fine for the durable-state buckets but makes answer-recall numbers
meaningless.

Two flags reproduce the write-policy stress cells, which hold the gold value fixed
and change only how the trajectory is presented:

```bash
--order permuted      # a STALE value is written last: last-writer-wins keeps it
--confidence inverted # the current value gets LOW confidence: confidence-weighting keeps a stale value
--no-authoritative-supersede   # do not bypass the contradiction gate's margin test
```

### MultiWOZ — dialogue-state slots

```bash
python run_multiwoz_durable_state.py \
  --arms B0,B2,Bsup,B3 \
  --out local_results/multiwoz_durable_state.json
```

Full validation split (1,000 dialogues) by default; `--limit N` for a smoke run.

### Entity-linking evasion — security suite

```bash
python run_entity_linking_evasion.py --paper-suite
```

Five seeds over seven attack axes, with original and mutated attacks, a
configuration-off ablation, and a benign false-positive workload. `--help` lists
the individual axes and intensities.

### LM-R — LongMemEval end-to-end (needs a GPU)

Real extraction from raw text with Qwen2.5-14B over the 277 MB haystack. This is
the expensive one: roughly **two days on one GPU** without cached extractions.

```bash
python run_7f_local.py --full \
  --extract-prompt longmemeval \
  --llm-model Qwen/Qwen2.5-14B-Instruct \
  --slot-linker qwen \
  --extract-cache local_results/lme_e2e_extract_cache_longmemeval.json \
  --link-cache local_results/lme_e2e_link_cache_longmemeval.json
```

Point `--extract-cache` / `--link-cache` at existing cache files to replay without
re-running the model. Cache identity includes the code revision and working-tree
diff, so an edit anywhere invalidates every key — see the module docstring before
reusing caches across revisions.

### Synthetic benchmark

Seeded and fully offline:

```bash
python -m ocm.scripts.build_benchmark --out benchmark.jsonl --seed 1337

python -m ocm.scripts.run_benchmark \
  --benchmark benchmark.jsonl --baselines B0,B1,B2,B3 \
  --out results.jsonl --log research_log.jsonl --top-k 10

python -m ocm.scripts.report_metrics --results results.jsonl --json metrics.json
```

`run_benchmark` generates a benchmark on the fly if `--benchmark` is omitted.
`report_metrics` takes exactly one of `--results` or `--benchmark`.
`python -m ocm.scripts.run_experiments` runs the whole suite.

## Arms

Selected by name via `--arms` / `--baselines`, defined in
[`ocm/evaluation/arms/`](ocm/evaluation/arms/):

| Arm | Governance |
| --- | --- |
| `B0` | vector retrieval only, none |
| `B1` | symbolic retrieval only, none |
| `B2` | graph + semantic, none |
| `B3` | full: contradiction gate, quarantine, provenance |
| `B4` | `B3` plus the answer policy |
| `Bsup` | latest-value supersession only |
| `Bmemgpt` | MemGPT-style self-editing baseline |
| `Brag`, `Brtcf` | retrieval-augmented / read-time conflict filtering |

## Metrics

The durable-state runners report the buckets `correct`, `stale`, `split`,
`abstained` and `missing`, alongside `task_success` and durable violation counts.
Definitions and the rationale for each bucket are in the module docstring of
[`ocm/evaluation/durable_state.py`](ocm/evaluation/durable_state.py).

## Configuration

Centralized in [`ocm/core/config.py`](ocm/core/config.py) (`Settings`, Pydantic
v2). Defaults are offline. The ones that change results:

| Setting | Default | Effect |
| --- | --- | --- |
| `extractor` | `"mock"` | `"llm"` opts in to a real model |
| `embedding_mode` | `"local"` | `sentence-transformers`; deterministic mode needs no dependency |
| `chroma_mode` | `"persistent"` | `"memory"` keeps the vector index in-process |
| `deterministic_test_mode` | `False` | reproducible IDs, in-memory storage, deterministic embeddings |
| `authoritative_update_supersede` | `False` | when set, an `update` supersedes unconditionally and **bypasses the contradiction gate's margin test** |

## Repository layout

| Path | Contents |
| --- | --- |
| `ocm/core/` | settings, container, IDs, logging |
| `ocm/ontology/` | entity/relation types, enums |
| `ocm/extraction/` | mock and LLM extractors |
| `ocm/resolution/` | normalization, entity resolution |
| `ocm/memory/` | write pipeline, commit manager, SQLite repo, graph, provenance, quarantine |
| `ocm/validation/` | schema validator, constraints C1–C10, contradiction checker |
| `ocm/governance/` | risk router, conditions, policy registry, review queue |
| `ocm/retrieval/` | classifier, retrievers, reranker, evidence packager, vector index |
| `ocm/agent/` | agent loop, memory tool, answer policy |
| `ocm/evaluation/` | benchmark generation, dataset adapters, arms, runner, metrics |
| `ocm/scripts/` | synthetic-benchmark CLI entry points |
| `ocm/tests/` | unit and property-based (Hypothesis) tests |

## Notebook (the GPU path)

[`OCM_Colab.ipynb`](OCM_Colab.ipynb) is the practical route for anything needing a
GPU, and it produced most of what is in [`results/`](results/). Open it in Colab
by uploading it directly (**File → Upload notebook**), or set `OCM_REPO_URL` in the
clone cell to a checkout of this repository.

Sections 1–5 need no GPU. Section 7 onward loads a local Qwen model through
`transformers` and runs the real-extraction arms.

| Section | What it does | Output |
| --- | --- | --- |
| 1–2 | clone, install, optional Drive mount for persistent output | — |
| 3 | sanity tests | — |
| 4 | offline governance demo, no GPU or API key | — |
| 5a–5b | benchmark, metrics, then the full offline suite (multi-seed CIs, significance, τ-sweep, stress) | `results_offline.json` |
| 6 | switch to real `sentence-transformers` embeddings | — |
| 7 | GPU check and Qwen2.5-14B-Instruct load in bf16 (~28 GB) | — |
| 7-alt | **T4 (16 GB) fallback**: Qwen2.5-7B-Instruct in 4-bit NF4 | — |
| 7b | full research experiment with the Qwen extractor and real embeddings | `results_qwen.json` |
| 7c | governed-write replay: qualitative evidence and false-quarantine reconciliation | `governance_examples.json` |
| 7d | MultiWOZ 2.2 real-data run (oracle extraction, so no LLM calls) | — |
| 7e | LongMemEval knowledge-update oracle arm, including the annotation pass | `results_longmemeval.json`, gold trajectories |
| 7f | LongMemEval end-to-end (LM-R): real extraction from raw text | extraction and slot-link caches |

The 7-alt cell matters if you only have a free-tier T4: it swaps the 14B bf16 load
for Qwen2.5-7B-Instruct in 4-bit so section 7 onward still runs, at a different
model scale than the reported numbers.

Section 7f is the expensive one — roughly two days on a single GPU. Its caches are
written to the output directory so re-runs replay instead of re-generating.

## License

MIT (see [`pyproject.toml`](pyproject.toml)).
