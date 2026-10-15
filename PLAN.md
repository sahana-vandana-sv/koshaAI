# PLAN.md — Kosha 7-day sprint

**Question:** for PubMedQA, how close can a self-hosted 4B model on one GPU get to a frontier API, and at what
cost per 1k queries? Decisions behind this plan are in `DECISIONS.md` (D-007–D-018 cover the sprint).

## Decisions at a glance
| Area | Decision | Ref |
|---|---|---|
| Dataset | PubMedQA (question + abstract → yes/no/maybe), official 500-item test set; MedMCQA 1k as OOD check | D-001, D-013 |
| Model | Gemma 4 E4B through every stage; Qwen3.5-4B + MedGemma 1.5 4B zero-shot only | D-002, D-007 |
| Frontier baseline | Claude Sonnet 5.5 (quality ceiling) + Claude Haiku 5.5 (cost competitor) | D-003, D-014 |
| Optimisations | LoRA vs QLoRA → FP8 + GPTQ 4-bit → vLLM with prefix/KV caching + speculative decoding | D-004, D-007 |
| Compute | Hourly GPU cloud for experiments; AWS for deployment only; $45 cap | D-008 |
| Deployment | AWS ap-south-1 (Mumbai), EC2 g6.xlarge (L4 24 GB), Docker vLLM, ALB, Terraform | D-005, D-009, D-012 |
| Evals | accuracy, macro-F1, per-class F1, ECE, invalid rate · TTFT/TPOT/p95 · $/1k · McNemar + bootstrap CI | D-006, D-015–D-018 |

## Architecture
```
 Mac ($0)                     GPU cloud (hourly, experiments)        AWS ap-south-1 Mumbai (deployment)
 data, eval code      ──►     L4 pod: S0, S3, S4                     S3: final model weights
 frontier baseline            L40S/A100 pod (~4h): S1, S2  ──────►   ALB ─► EC2 g6.xlarge L4, Docker vLLM
 report, load tests  ─────────────────────────────────────────────►  Secrets Manager · CloudWatch · Budget · auto-stop
```

## Pipeline — one leaderboard row per stage
| Row | What changes | Day |
|---|---|---|
| B0 | majority-class floor | 1 |
| F0 | Sonnet 5.5, Haiku 5.5 × {0-shot, 3-shot} | 2 |
| S0 / S0b | Gemma 0-shot / 3-shot (bf16); Qwen, MedGemma 0-shot | 2 |
| S1 | + LoRA bf16, r16 | 3 |
| S2 | LoRA → QLoRA NF4 | 3 |
| S3 | merged best adapter → FP8 W8A8, GPTQ W4A16 | 4 |
| S4a–f | HF generate → vLLM → +prefix cache → +FP8 KV cache → +MTP spec decode → +structured output | 5 |
| S5 | best S4 config deployed in Mumbai, load-tested | 6–7 |

## Day by day
| Day | Theme | Key evals / exit gate | Est. $ | Status |
|---|---|---|---|---|
| 0 | Prerequisites: AWS G/VT quota (ap-south-1, ≥4 vCPU), GPU-cloud credit + volume, HF access (Gemma 4, MedGemma), Anthropic key | accounts and access ready | 0 | ☐ |
| 1 | Foundations: repo, data splits, prompts, parsing, metrics, run folders | unit tests green; zero PMID overlap; B0 accuracy 0.50–0.60 | 0 | ☐ |
| 2 | Baselines: F0 frontier (Batch API) + S0/S0b students (vLLM) | F0 invalid = 0; HF-vs-vLLM parity ≥ 95%; Gemma S0 > B0 | ~6 | ☐ |
| 3 | Fine-tuning: S1 LoRA vs S2 QLoRA | dev F1(maybe) > 0 during training; S1 > S0b on dev; MedMCQA drop ≤ 3 pts; S2 VRAM < 24 GB | ~5 | ☐ |
| 4 | Quantization S3 + benchmark harness | FP8 ≈ bf16 (McNemar); GPTQ within ~2 pts; size and speed logged | ~2 | ☐ |
| 5 | Inference optimisation + KV caching S4 | quality unchanged per rung; latency/throughput at c=1/8/32; cache hit and MTP acceptance rates; KV table | ~2 | ☐ |
| 6 | AWS deployment S5 (Terraform) | health 200; 401 without key; dev parity ≥ 98% via ALB; auto-stop works | ~6 | ☐ |
| 7 | Load test, leaderboard, Pareto plots, REPORT.md, teardown | significance for every claim; `terraform destroy`; total ≤ $45 | ~5 | ☐ |

Working rule: no code is written for a day until it is approved ("go Day N"). Each day ends by appending its
decisions (D-020+) to `DECISIONS.md` and ticking the status column here.

## Learning resources
| Day | Resources |
|---|---|
| 1 | [PubMedQA paper](https://arxiv.org/abs/1909.06146) · [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) · [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) |
| 2 | [Claude Batch processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing) · [Claude Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) · [vLLM quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) |
| 3 | [LoRA paper](https://arxiv.org/abs/2106.09685) · [QLoRA paper](https://arxiv.org/abs/2305.14314) · [Raschka: Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) · [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) |
| 4 | [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) · [GPTQ paper](https://arxiv.org/abs/2210.17323) · [llm-compressor](https://github.com/vllm-project/llm-compressor) |
| 5 | [PagedAttention / vLLM paper](https://arxiv.org/abs/2309.06180) · [Raschka: Coding the KV Cache](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) · [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) · [vLLM MTP speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/) |
| 6 | [Terraform AWS tutorial](https://developer.hashicorp.com/terraform/tutorials/aws-get-started) · [vLLM Docker deployment](https://docs.vllm.ai/en/latest/deployment/docker.html) · [ALB user guide](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/introduction.html) |
| 7 | [LLM Inference Performance Engineering (Databricks)](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) · [Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing) |

## Definition of done
1. `results/leaderboard.csv` with B0, F0, S0–S5 (Gemma) and S0 (Qwen, MedGemma).
2. Pareto plots: macro-F1 vs $/1k queries, and vs p95 latency.
3. McNemar or bootstrap CI behind every "X beats Y" claim.
4. `REPORT.md` answering the project question with numbers.
5. `terraform destroy` leaves nothing billable.

## Risks
| Risk | Fallback |
|---|---|
| AWS GPU quota not approved by Day 6 | request on Day 0 |
| TRL/PEFT issues with Gemma 4's multimodal model class | Unsloth (D-011) |
| MTP drafter incompatible with the fine-tuned or quantized model | report it, skip the rung |
