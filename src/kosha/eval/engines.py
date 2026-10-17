"""Model engines for quality evals. vllm/torch are imported lazily, so the Mac never needs them."""

import math
from collections.abc import Sequence

import numpy as np

from kosha import PUBMEDQA_LABELS, SEED


def label_probs(steps, labels: Sequence[str] = PUBMEDQA_LABELS) -> list[float] | None:
    """Renormalised P(label) at the first generated position whose top token is a label (D-016).

    `steps` is vLLM's per-position logprobs: a list of {token_id: Logprob(logprob, decoded_token, ...)}.
    """
    lower = [label.lower() for label in labels]

    def norm(lp) -> str:
        return (lp.decoded_token or "").strip().strip('"').lower()

    for step in steps or []:
        if norm(max(step.values(), key=lambda lp: lp.logprob)) in lower:
            p = np.zeros(len(labels))
            for lp in step.values():
                if (tok := norm(lp)) in lower:
                    p[lower.index(tok)] += math.exp(lp.logprob)  # "yes" and "Yes" add up
            return (p / p.sum()).tolist()
    return None


def generate_vllm(cfg: dict, conversations: list[list[dict]], labels: Sequence[str]) -> list[dict]:
    """Greedy decoding with the vLLM offline engine (D-015)."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=cfg["model_path"], dtype=cfg.get("dtype", "auto"), seed=SEED, **cfg.get("vllm_kwargs", {})
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=cfg.get("max_tokens", 256), logprobs=5)
    outs = llm.chat(
        conversations, sampling, chat_template_kwargs=cfg.get("chat_template_kwargs"), use_tqdm=True
    )
    return [
        {
            "text": o.outputs[0].text,
            "out_tokens": len(o.outputs[0].token_ids),
            "finish_reason": o.outputs[0].finish_reason,
            "probs": label_probs(o.outputs[0].logprobs, labels),
        }
        for o in outs
    ]


def generate_hf(cfg: dict, conversations: list[list[dict]], labels: Sequence[str]) -> list[dict]:
    """Plain transformers generate(): the parity check today, the naive latency baseline (S4a) on Day 5."""
    import torch
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(cfg["model_path"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model_cls = getattr(transformers, cfg.get("model_class", "AutoModelForCausalLM"))
    model = model_cls.from_pretrained(cfg["model_path"], torch_dtype=torch.bfloat16, device_map="cuda").eval()
    torch.manual_seed(SEED)

    outputs, bs = [], cfg.get("batch_size", 8)
    for i in range(0, len(conversations), bs):
        enc = tok.apply_chat_template(
            conversations[i : i + bs],
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
            **(cfg.get("chat_template_kwargs") or {}),
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=cfg.get("max_tokens", 256), do_sample=False)
        for row in gen[:, enc["input_ids"].shape[1] :]:
            outputs.append({"text": tok.decode(row, skip_special_tokens=True)})
    return outputs
