# CLAUDE.md — Kosha

**Kosha: how close can a self-hosted 4B model get to the frontier?**

*Kosha* (Sanskrit: sheath / treasury / vault). Two meanings, both deliberate:
- **Vault** — the model and the data stay inside our own infrastructure and region (sovereignty).
- **Sheaths** — like the *pancha kosha* (five nested layers), the pipeline wraps a base model in
  layers — fine-tuning, quantization, inference optimisation, deployment — measuring each one.

Use `kosha` as the Python package name, CLI name and resource prefix (S3 buckets, EC2 tags,
MLflow experiment names), e.g. `kosha-ap-south-1-weights`, `kosha eval`, `kosha bench`.

## What this project is
Self-host a small open-weight LLM, specialise it for one narrow task, and push it through a
staged optimisation pipeline (fine-tune → quantize → inference-optimise), measuring quality,
latency and cost at **every** stage against a frontier API baseline.

The question the project answers:
> "For a narrow, data-sensitive task, how close can a self-hosted 4B model on one GPU get to a
> frontier API — and at what cost per 1k queries?"

Read `DECISIONS.md` before changing any model, dataset, metric or infra choice. If you change
one, add a new decision entry there (don't silently edit an old one — supersede it).

## Task (see DECISIONS.md D-001)
**PubMedQA** — given a question + a PubMed abstract (context), answer `yes | no | maybe`,
optionally with a one-sentence rationale.
- Train: `pqa_artificial` (~211k, auto-labelled, yes/no only) + `pqa_labeled` train split (450)
- Dev: `pqa_labeled` dev split (50)
- Test: `pqa_labeled` test split (500) — **frozen, never used for tuning or prompt iteration**
- OOD check: MedMCQA dev subset (1,000 items) — catches over-specialisation

## Models (see D-002, D-003)
| Role | Model | Notes |
|---|---|---|
| Primary student | `google/gemma-4-E4B-it` | Apache 2.0, official QAT + MTP drafter |
| Challenger student | `Qwen/Qwen3.5-4B` | hybrid Gated DeltaNet attention — KV-cache contrast |
| Domain reference | `google/medgemma-1.5-4b-it` | "does domain pretraining beat our LoRA?" |
| Frontier ceiling | Claude Sonnet 5.5 (API) | quality ceiling |
| Frontier cost peer | Claude Haiku 5.5 (API) | cheapest frontier option — the real competitor |

Verify exact HF repo IDs and API model strings in Phase 0; don't trust this table blindly.

## Pipeline stages — every stage produces a row in `results/leaderboard.csv`
| Stage | ID | What changes |
|---|---|---|
| Frontier baseline | F0 | Sonnet 5.5 + Haiku 5.5, zero-shot and 3-shot |
| Base zero-shot | S0 | student, HF transformers, bf16, no tuning |
| Base few-shot | S0b | same, 3-shot prompt |
| LoRA (bf16) | S1 | LoRA on attention+MLP, r=16 |
| QLoRA (4-bit NF4) | S2 | same recipe, 4-bit base — compare to S1 |
| Quantized serving | S3 | merge best adapter → AWQ/GPTQ W4A16, FP8 (W8A8), GGUF Q4_K_M |
| Inference-optimised | S4 | vLLM: continuous batching, prefix caching, FP8 KV cache, speculative decoding (MTP drafter) |
| Deployed | S5 | S4 config behind an endpoint on AWS, load-tested |

Change **one variable per stage**. If two things change, split into two rows.

## Metrics captured per stage (see D-006)
- Quality: accuracy, macro-F1, per-class F1, ECE (calibration), invalid-output rate
- Latency: TTFT p50/p95, TPOT p50/p95, end-to-end p95 at concurrency 1/8/32
- Throughput: requests/s and output tokens/s at fixed concurrency
- Resource: peak VRAM, model size on disk
- Cost: $ per 1k queries (GPU $/hr ÷ measured throughput; API = token usage × list price)
- Training (S1/S2): wall-clock, peak VRAM, GPU-hours, $

## Repo layout
```
configs/            # one YAML per stage/run; runs are fully reproducible from config
data/               # download + preprocessing scripts (raw data gitignored)
src/kosha/          # installable package; CLI entry point `kosha`
  data/             # loaders, prompt templates, splits
  train/            # LoRA / QLoRA (TRL + PEFT, or Unsloth)
  quantize/         # AWQ, GPTQ, FP8 (llm-compressor), GGUF export
  serve/            # vLLM launch configs, HF baseline server
  eval/             # quality evals, answer parsing, stats (bootstrap, McNemar)
  bench/            # latency/throughput load generator
  frontier/         # API clients with caching + cost tracking
infra/              # Terraform/CDK or shell for EC2, S3, IAM
results/
  runs/<run_id>/    # predictions.jsonl, metrics.json, config snapshot, env info
  leaderboard.csv   # one row per run — the project's headline artifact
notebooks/          # analysis/plots only, never source of truth
```

## Commands (fill in as they're built)
```bash
make setup          # uv venv + deps
make data           # download + split PubMedQA, MedMCQA subset
make eval RUN=configs/s0_gemma4_e4b.yaml
make train RUN=configs/s1_lora.yaml
make bench RUN=configs/s4_vllm.yaml
make leaderboard    # regenerate results/leaderboard.csv + plots
```

## Conventions
- Python 3.11, `uv` for env, `ruff` for lint/format, type hints on public functions.
- Every run writes `results/runs/<run_id>/` with: resolved config, git SHA, package versions,
  GPU type, seed, predictions.jsonl (one line per item incl. raw output), metrics.json.
- Seeds fixed (`42`); greedy decoding (temperature 0) for all quality evals.
- Same prompt template for every model within a stage; templates live in `src/data/prompts.py`.
- Answer parsing is shared code; unparseable outputs count as wrong and are reported separately.
- Frontier API calls are cached on disk keyed by (model, prompt hash) — never pay twice.
- Experiment tracking: MLflow (local file store, synced to S3) or W&B — pick once, record in DECISIONS.md.

## Guardrails
- **Never** look at test-set predictions to tune prompts, hyperparameters or thresholds. Dev only.
- **Cost:** every EC2 launch script must set an auto-stop (idle shutdown or `shutdown -h +N`).
  Prefer spot for training with checkpoints to S3 every N steps. Print estimated $ before any run > 1 GPU-hour.
- Secrets via AWS Secrets Manager / env vars only; nothing in configs or git.
- Keep data and model weights in the chosen sovereign region (D-005); no cross-region S3 replication.
- Don't upgrade vLLM/transformers mid-comparison; pin versions in `uv.lock`. If you must, re-run affected baselines.

## Definition of done
1. `leaderboard.csv` has F0, S0–S5 rows for the primary model (challenger at least S0–S3).
2. A Pareto plot: accuracy vs $/1k queries and accuracy vs p95 latency, all stages + frontier.
3. Statistical significance stated for every "X beats Y" claim (McNemar or bootstrap CI).
4. A short write-up (`REPORT.md`) answering the project question with numbers.
