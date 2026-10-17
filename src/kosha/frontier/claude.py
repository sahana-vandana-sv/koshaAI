"""Frontier baseline (F0): Claude via the Message Batches API, with a disk cache and cost tracking (D-014)."""

import hashlib
import json
import time
from collections.abc import Sequence

import anthropic
import diskcache

from kosha import PUBMEDQA_LABELS

# USD per million tokens (input, output) at list price for prompts <= 100K tokens. Batch API = 50% off.
PRICES = {"claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50)}
# The 5.5 models reject temperature/top_p; "thinking off + low effort" is the closest to greedy (D-014).
THINKING_OFF = {"claude-sonnet-5-5": {"type": "between_tools"}, "claude-haiku-5-5": {"type": "disabled"}}
CACHE = diskcache.Cache(".cache/frontier")  # never pay twice for the same request (CLAUDE.md)


def answer_schema(labels: Sequence[str]) -> dict:
    return {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": list(labels)}, "rationale": {"type": "string"}},
        "required": ["answer", "rationale"],
        "additionalProperties": False,
    }


def params_for(model: str, messages: list[dict], labels: Sequence[str] = PUBMEDQA_LABELS) -> dict:
    """Turn our shared chat messages into a Messages API request (system prompt goes top-level)."""
    return {
        "model": model,
        "max_tokens": 1024,
        "system": messages[0]["content"],
        "messages": messages[1:],
        "thinking": THINKING_OFF[model],
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": answer_schema(labels)},
        },
    }


def cost_usd(model: str, usage: dict, batch: bool) -> float:
    price_in, price_out = PRICES[model]
    tokens_in = (
        usage["input_tokens"]
        + 1.25 * (usage.get("cache_creation_input_tokens") or 0)
        + 0.1 * (usage.get("cache_read_input_tokens") or 0)
    )
    return (0.5 if batch else 1.0) * (tokens_in * price_in + usage["output_tokens"] * price_out) / 1e6


def estimate_usd(model: str, params: list[dict]) -> float:
    """Rough pre-submit estimate: ~4 characters per token in, <=100 tokens out, batch price."""
    price_in, price_out = PRICES[model]
    chars = sum(len(p["system"]) + sum(len(m["content"]) for m in p["messages"]) for p in params)
    return 0.5 * (chars / 4 * price_in + 100 * len(params) * price_out) / 1e6


def _key(params: dict, repeat: int) -> str:
    return hashlib.sha256(json.dumps([params, repeat], sort_keys=True).encode()).hexdigest()


def run_batch(
    model: str,
    ids: list[str],
    conversations: list[list[dict]],
    labels: Sequence[str] = PUBMEDQA_LABELS,
    repeat: int = 0,
    poll_s: int = 30,
) -> list[dict]:
    """One output per id, in input order. Cached answers are reused; only new requests are sent."""
    params = [params_for(model, c, labels) for c in conversations]
    keys = [_key(p, repeat) for p in params]
    todo = {cid: (k, p) for cid, k, p in zip(ids, keys, params, strict=True) if k not in CACHE}
    errors: dict[str, str] = {}

    if todo:
        est = estimate_usd(model, [p for _, p in todo.values()])
        print(f"[{model}] {len(todo)} uncached requests, est. ~${est:.2f} at batch price", flush=True)
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        batch = client.messages.batches.create(
            requests=[{"custom_id": cid, "params": p} for cid, (_, p) in todo.items()]
        )
        print(f"[{model}] submitted batch {batch.id}", flush=True)
        while (status := client.messages.batches.retrieve(batch.id)).processing_status != "ended":
            c = status.request_counts
            print(f"  processing={c.processing} succeeded={c.succeeded} errored={c.errored}", flush=True)
            time.sleep(poll_s)
        for res in client.messages.batches.results(batch.id):  # results arrive in any order
            if res.result.type != "succeeded":
                errors[res.custom_id] = res.result.type  # not cached, so a re-run retries it
                continue
            msg = res.result.message
            usage = msg.usage.to_dict()
            CACHE[todo[res.custom_id][0]] = {
                "text": next((b.text for b in msg.content if b.type == "text"), ""),
                "stop_reason": msg.stop_reason,
                "usage": usage,
                "usd": cost_usd(model, usage, batch=True),
            }

    return [
        CACHE.get(k, {"text": "", "error": errors.get(cid, "missing")})
        for cid, k in zip(ids, keys, strict=True)
    ]
