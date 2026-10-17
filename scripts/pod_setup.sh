#!/usr/bin/env bash
# Run inside the GPU pod (vllm/vllm-openai image). Needs HF_TOKEN for the gated Gemma/MedGemma repos.
set -euo pipefail
: "${HF_TOKEN:?export HF_TOKEN first}"
: "${REPO_URL:?export REPO_URL (your private GitHub repo)}"
cd /workspace
[ -d koshaAI ] || git clone "$REPO_URL" koshaAI
cd koshaAI
pip install -q -e .                        # vllm, torch, transformers come from the image
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
kosha data                                  # rebuild splits deterministically...
git diff --exit-code data/processed/manifest.json && echo "data identical to the Mac build"   # ...and prove it
