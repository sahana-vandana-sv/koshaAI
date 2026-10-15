# DECISIONS.md — Kosha

Format: each decision has a status (**Proposed** → **Accepted** → **Superseded by D-xxx**).
Never edit an Accepted decision's rationale; add a new entry that supersedes it.
D-001–D-006 came from the initial brainstorm (2026-10-15) and were reviewed in Phase 0 the same day:
all are Accepted, with amendments recorded as D-007–D-018 (7-day sprint plan, see PLAN.md).

---

## D-000 — Project name: Kosha
**Status:** Accepted (2026-10-15)

**Decision:** The project is called **Kosha**: Sanskrit for sheath, treasury or vault.

**Why:** "Vault" carries the sovereignty idea (weights and data stay in our own infra and region).
"Sheath" maps to the pipeline: the *pancha kosha* concept of nested layers around a core mirrors
the base model wrapped in fine-tuning, quantization, inference optimisation and deployment.
Short, pronounceable, and works as a package, CLI and resource prefix.

**Alternatives considered:** Swaraj (too historically weighty for a benchmark repo), Homegrown
(clear but generic), Ceiling & Floor (good write-up title, weak repo name), Pocket Oracle (demo-ish).

---

## D-001 — Task & dataset: PubMedQA (primary), MedMCQA (OOD check)
**Status:** Accepted (2026-10-15) — split and training-mix details amended by D-013

**Options considered**
| Dataset | Task | Pros | Cons |
|---|---|---|---|
| **PubMedQA** | question + abstract → yes/no/maybe | Crisp metric (accuracy/F1); long context (abstract ~250–400 tokens) makes prefill, prefix-caching and KV-cache work actually matter; strong sovereignty story (clinical/biomedical data shouldn't leave your infra) | Only 1k expert-labelled items; the 211k artificial set is noisy and has no "maybe"; likely seen by frontier models in pretraining |
| BillSum | US bill → summary | Very long inputs (great for KV-cache stress); 20k+ training pairs | Summarisation eval is noisy (ROUGE weak, needs LLM judge); US-centric — odd fit for a "sovereign" pitch |
| MedMCQA | Indian medical entrance MCQs (4 options) | 180k+ train items, crisp metric, India relevance | Short inputs → little to optimise in prefill/KV; knowledge-recall more than reading comprehension, so LoRA gains are smaller |

**Decision:** PubMedQA as the primary task; MedMCQA dev subset as a held-out out-of-distribution
regression check. BillSum kept as a stretch "long-context" benchmark for the S4 KV-cache experiments only.

**Why:** An exact-match metric keeps every stage comparison clean, and the context-heavy input is
what makes the inference-optimisation stage meaningful. The "maybe" class is the interesting
hard part — it's where frontier models and small models diverge.

**Risks / mitigations**
- Label noise in `pqa_artificial` → ablation: train on labelled-only vs artificial+labelled.
- No "maybe" in artificial set → class-weighted loss or upsample the 450 labelled examples.
- Contamination → report it as a caveat; optionally build a 100-item fresh set from 2026 PubMed abstracts
  (labelled with frontier-model draft + human check) as a clean test.

---

## D-002 — Student model: Gemma 4 E4B (primary), Qwen3.5-4B (challenger), MedGemma 1.5 4B (reference)
**Status:** Accepted (2026-10-15) — sprint scope for Qwen and MedGemma amended by D-007

**Context:** Must fit LoRA training *and* serving on a single 24–48 GB GPU, permissive licence,
first-class vLLM support, and enough architectural features to make the optimisation stages interesting.

**Options considered**
- **Gemma 4 E4B** — Apache 2.0 (Gemma 3 had a custom licence; this matters for a sovereignty pitch).
  ~18 GB in bf16, ~4.5 GB at 4-bit. Google ships official **QAT** checkpoints (incl. W4A16 for vLLM)
  and **MTP drafter** models for speculative decoding — so S3 can compare post-training quantization
  vs QAT, and S4 gets speculative decoding "for free".
- **Qwen3.5-4B** — strongest sub-5B model on general benchmarks; Apache 2.0. Uses a hybrid
  Gated DeltaNet (linear-attention) architecture, so most layers have **no growing KV cache** — a
  great contrast for the KV-cache experiments, but tooling (QLoRA kernels, AWQ, vLLM features)
  may lag standard transformers. Thinking mode off by default on small models — keep it off for eval.
- **Qwen3.5-9B / Gemma 4 12B** — better quality, but slower iteration; keep as a "scale-up" stretch.
- **MedGemma 1.5 4B** — medical-domain Gemma. Not a student; it's the control that answers
  "is our cheap LoRA as good as Google's domain pretraining?"
- Llama / Mistral small models — weaker at this size class in 2026 and more restrictive licences; skipped.

**Decision:** Run the full S0→S5 pipeline on Gemma 4 E4B. Run Qwen3.5-4B through S0→S3 (S4 if
vLLM support is solid). Evaluate MedGemma 1.5 4B at S0 and S3 only.

**Phase 0 verification items:** confirm exact HF IDs; confirm PEFT/LoRA works with Gemma 4's
"effective parameter" (per-layer embedding) design; confirm vLLM version supports both architectures.

---

## D-003 — Frontier baseline: Claude Sonnet 5.5 (ceiling) + Claude Haiku 5.5 (cost peer)
**Status:** Accepted (2026-10-15) — "temperature 0" in the protocol superseded by D-014

**Why two:** Sonnet answers "how far below the ceiling are we?"; Haiku answers the business question
"would anyone actually self-host instead of calling the cheapest frontier model?" Haiku is the
competitor that matters for the $/1k-queries chart.

**Protocol:** identical prompt template, temperature 0, zero-shot and 3-shot, structured output
(single label + rationale). Log input/output tokens for exact cost. Cache all responses.

**Optional third arm:** one non-Anthropic frontier model (e.g. a GPT or Gemini model) for vendor-neutrality.

**Sovereignty note:** also record whether the frontier model is reachable *in-region* (e.g. via
Amazon Bedrock in the chosen AWS region). "Frontier via in-region Bedrock" vs "self-hosted" is a
fairer sovereignty comparison than "frontier via public US endpoint".

---

## D-004 — Optimisation stages and what each must measure
**Status:** Accepted (2026-10-15) — sprint subset of S3 variants and ablations in D-007; selection rules in D-017

1. **Fine-tuning (S1 LoRA vs S2 QLoRA)**
   - Library: TRL + PEFT (or Unsloth for speed — record which). r=16, α=32, dropout 0.05,
     targets: all linear layers; lr 2e-4, cosine, 1–2 epochs, max seq 1024.
   - Output format trained: `{"answer": "yes|no|maybe", "rationale": "..."}`.
   - Compare: accuracy, training VRAM, wall-clock, $ — does 4-bit training cost quality?
   - Ablations (cheap, high signal): data size (450 vs 10k vs 50k vs all), rank (8/16/64).
2. **Quantization (S3)** — on the merged best adapter
   - bf16 (reference) → FP8 W8A8 (L40S/Ada supports FP8 natively) → AWQ W4A16 → GPTQ W4A16 → GGUF Q4_K_M (CPU/edge story).
   - For Gemma: also compare PTQ vs Google's QAT checkpoint + re-applied LoRA.
   - Measure quality delta *and* speed; 4-bit isn't always faster at small batch sizes.
3. **Inference optimisation (S4)** — vLLM, add one feature at a time:
   - HF `generate()` naive baseline → vLLM default (continuous batching, PagedAttention)
   - + automatic **prefix caching** (shared system prompt + few-shot examples → big TTFT win)
   - + **FP8 KV cache** (more concurrent sequences per GB)
   - + **speculative decoding** (Gemma MTP drafter) — note: outputs are short (1 label + rationale), so
     gains may be small; that's a valid finding.
   - + guided/structured decoding for the JSON output (also kills invalid-output rate)
   - KV-cache study: VRAM and max concurrency vs context length (PubMedQA ~500 tok; BillSum 2–8k tok),
     Gemma 4 vs Qwen3.5 hybrid attention.

---

## D-005 — Deployment: AWS ap-south-1 (Mumbai), EC2 g6e + vLLM in Docker
**Status:** Accepted (2026-10-15) — region and stack stand; training rows superseded by D-008, serving instance by D-009

**Region:** `ap-south-1` (Mumbai) — data residency in India is the point of "sovereign".
Cost: ~20% pricier than us-east-1 for g6e (≈$2.69 vs $2.24/hr on-demand for g6e.2xlarge); accept it,
or do *training only* in a cheaper region with public data and keep *serving* in Mumbai — record which.
Hyderabad (`ap-south-2`) is the DR / second-region option.

**Instances**
| Use | Instance | GPU | Why |
|---|---|---|---|
| Training (S1/S2) | g6e.xlarge / g6e.2xlarge, **spot** | 1× L40S 48 GB | bf16 LoRA on a 4B fits easily; spot + S3 checkpoints |
| Serving (S4/S5) | g6e.xlarge | 1× L40S 48 GB | FP8 support, headroom for KV cache |
| Cheap serving variant | g6.xlarge | 1× L4 24 GB | if 4-bit model + KV fits — the "how cheap can we go" row |
| Stretch | Inferentia2 (inf2) via Neuron | — | AWS-native accelerator comparison; only if time allows |

**Stack:** EC2 + Docker + vLLM OpenAI-compatible server, model weights in S3 (same region),
ALB or API Gateway in front, CloudWatch for GPU metrics, auto-stop on idle.
**Not SageMaker initially** — EC2 gives full visibility into what each optimisation does; SageMaker
endpoint can be a later "productionised" variant.

**Open question:** confirm g6e/g6 availability and spot capacity in ap-south-1 in Phase 0.

---

## D-006 — Evaluation protocol
**Status:** Accepted (2026-10-15) — refined by D-015 (engine), D-016 (output/ECE), D-018 (cost model); IFEval and LLM-judge out of sprint scope (D-007)

- **Primary metric:** accuracy and macro-F1 on PubMedQA test (500). Macro-F1 because "maybe" is rare
  and accuracy hides failure on it.
- **Secondary quality:** per-class F1, confusion matrix, ECE (calibration from label logprobs where
  available), invalid-output rate.
- **Regression checks:** MedMCQA dev subset (OOD medical); a small general-capability set
  (e.g. IFEval subset) to detect catastrophic forgetting after LoRA.
- **Rationale quality (optional):** LLM-as-judge (Sonnet 5.5) on 100 sampled rationales —
  faithfulness to abstract, 1–5 scale — with ~30 human-checked to validate the judge.
- **Statistics:** 500 items ⇒ ~±4 pt 95% CI on accuracy. Report bootstrap CIs; use McNemar's test for
  paired model comparisons. Don't claim wins inside the noise.
- **Systems benchmarks:** fixed request set (dev+test prompts, no labels needed), concurrency 1/8/32,
  warm-up excluded, 3 repeats, report p50/p95. Same harness for self-hosted and API.
- **Cost model:** self-hosted $/1k = (instance $/hr) ÷ (measured requests/hr at target concurrency) × 1000;
  include idle time assumption explicitly (e.g. 30% utilisation) as a second figure.
- **Leakage rules:** test set touched only by `make eval` on frozen configs; all tuning on dev.

---

## Phase 0 open questions — resolved 2026-10-15
1. Experiment tracker → MLflow (D-010).
2. Unsloth vs TRL/PEFT → TRL + PEFT, Unsloth as fallback (D-011).
3. Where to train → hourly GPU cloud on public data; serving stays in Mumbai (D-008).
4. Budget cap → $45 total (D-008).
5. Contamination-free test set → deferred until after the sprint; reported as a caveat instead.

---

## D-007 — 7-day sprint scope
**Status:** Accepted (2026-10-15)

**Decision:** Gemma 4 E4B goes through every stage S0→S5. Qwen3.5-4B and MedGemma 1.5 4B are evaluated at
S0 (zero-shot) only. S3 covers FP8 W8A8 and GPTQ W4A16. Out of sprint scope: AWQ, GGUF, the Gemma QAT comparison,
rank and data-size ablations, IFEval, LLM-as-judge on rationales, a non-Anthropic frontier arm, BillSum.

**Why:** 7 days and a $45 cap. One model taken through the full pipeline answers the project question; three
models half-done does not. Amends D-002, D-004 and CLAUDE.md definition-of-done item 1 (challenger at S0).

---

## D-008 — Compute: experiments on an hourly GPU cloud, AWS for deployment only
**Status:** Accepted (2026-10-15) — supersedes the training rows of D-005

**Decision:** S0–S4 run on an hourly GPU cloud: L4 24 GB pods (the same GPU as the AWS deployment) plus one
L40S/A100 session (~4 h) for S1/S2 training. AWS is used only for S5 deployment. Budget: $45 total — AWS Budget
$25 with alerts, prepaid GPU-cloud credit (a natural hard cap), $10 Anthropic Console spend limit.

**Why:** running dev boxes, GPU quotas and two instance types on AWS made the plan cumbersome for a 7-day sprint.

**Sovereignty note:** only public data (PubMedQA, MedMCQA) is used off-AWS. Final model weights live in S3
ap-south-1 and serving stays in Mumbai. With private data, training would move into ap-south-1.

**Alternatives considered:** everything on AWS (more setup, ~2× cost); Colab Pro (notebook-only, session limits);
free Kaggle T4s (no bf16/FP8, breaks hardware consistency).

---

## D-009 — Serving instance: EC2 g6.xlarge (L4 24 GB) in ap-south-1
**Status:** Accepted (2026-10-15) — supersedes the serving rows of D-005

**Decision:** S5 serves the best S4 config on g6.xlarge (1× L4 24 GB, ~$0.97/hr on-demand in Mumbai).

**Why:** the quantized 4B model plus KV cache fits in 24 GB, the L4 supports FP8, and it is the cheapest GPU that
serves the model — the "how cheap can sovereign be" row. Experiments use the same GPU model, so S4 numbers transfer.

---

## D-010 — Experiment tracker: MLflow
**Status:** Accepted (2026-10-15)

**Decision:** MLflow local store, mirrored to S3. `results/runs/` remains the source of truth; MLflow holds curves
(training loss, dev accuracy during training).

**Why:** self-hosted and sovereign-friendly; no run metadata sent to a SaaS.

---

## D-011 — Training library: TRL + PEFT
**Status:** Accepted (2026-10-15)

**Decision:** S1/S2 use TRL `SFTTrainer` + PEFT (+ bitsandbytes for QLoRA). Unsloth is the fallback if Gemma 4's
multimodal class breaks PEFT; if used, record it as a separate variable.

**Why:** standard, transparent, and better for learning what each knob does.

---

## D-012 — Infrastructure as code: Terraform for the deploy stack
**Status:** Accepted (2026-10-15)

**Decision:** `infra/` Terraform creates the S5 stack: S3 bucket, IAM role, security groups, EC2 g6.xlarge (Deep
Learning AMI), ALB, Secrets Manager secret, AWS Budget. Default VPC, no NAT gateway (cost), SSM Session Manager
instead of SSH. ALB on HTTP with an IP allowlist plus a vLLM API key from Secrets Manager (secret value set via CLI,
never in Terraform state). HTTPS deferred (needs a domain + ACM).

**Why:** reproducible apply/destroy, no forgotten resources, and a transferable skill.

---

## D-013 — Data protocol
**Status:** Accepted (2026-10-15) — amends D-001

**Decision:**
- Test = the **official 500 PubMedQA test PMIDs** (`test_ground_truth.json` in the PubMedQA repo).
- Remaining 500 labelled items → 450 train / 50 dev, stratified, seed 42.
- Training mix: 20k stratified from `pqa_artificial` (yes ≤ 65%) + the 450 labelled items ×4 upsampled
  (the only source of "maybe").
- Rationale target = first sentence of `long_answer`.
- OOD check = 1,000 MedMCQA validation items (seed 42).
- Quantization calibration data comes from train only, never dev/test.

**Why:** the official test split keeps results comparable with the literature; upsampling protects the "maybe" class.

---

## D-014 — Frontier protocol amendment
**Status:** Accepted (2026-10-15) — supersedes "temperature 0" in D-003

**Decision:** Models `claude-sonnet-5-5` ($2/$10 per MTok) and `claude-haiku-5-5` ($0.10/$0.50 per MTok). Default
sampling. Thinking off (Sonnet `{"type": "between_tools"}`, Haiku `{"type": "disabled"}`), effort `low`,
JSON-schema structured output with an enum label. Quality evals via the Message Batches API (50% off); latency
via the real-time API using the same load harness as self-hosted. Dev set run twice → agreement rate reported as
frontier non-determinism. ECE not reported for frontier (no logprobs).

**Why:** Sonnet 5.5 and Haiku 5.5 reject non-default sampling parameters, so temperature 0 is not possible.

---

## D-015 — Eval engine
**Status:** Accepted (2026-10-15) — amends the S0 row in CLAUDE.md

**Decision:** All quality evals S0–S3 use the vLLM offline engine with greedy decoding; LoRA adapters are merged
before eval. HF `generate()` is used only as the S4a naive latency baseline.

**Why:** ~10× faster evals (budget), and one engine across stages isolates the effect of weight changes.

**Check:** HF vs vLLM agreement ≥ 95% on 50 dev items at S0.

---

## D-016 — Output format and calibration
**Status:** Accepted (2026-10-15)

**Decision:** Output `{"answer": "yes|no|maybe", "rationale": "..."}` with the answer first. ECE uses the
renormalised P(yes/no/maybe) at the answer token, 10 bins. Unparseable outputs count as wrong and are reported
separately (invalid rate, JSON-valid rate).

**Why:** answer-first puts the label logprob at a fixed early position and lets the label be read before the rationale.

---

## D-017 — Pre-registered selection rules (dev only)
**Status:** Accepted (2026-10-15)

**Decision:** Always use the final checkpoint. S3 input = the S1/S2 adapter with higher dev macro-F1 (S1 if within
2 pts). S4 input = the S3 variant within 1 pt of bf16 dev macro-F1 with the highest c=32 throughput. In S4, a rung
that doesn't help is dropped from the cumulative config and the drop is recorded.

**Why:** fixing selection rules before seeing results keeps the test set clean.

---

## D-018 — Cost model
**Status:** Accepted (2026-10-15) — refines D-006

**Decision:** Self-hosted $/1k queries = instance $/hr ÷ (measured rps at c=32 × 3600 × utilisation) × 1000,
reported at utilisation 1.0 and 0.3. Frontier = measured tokens × list price, at real-time and batch rates.
Training $ = GPU-hours × $/hr actually paid.
