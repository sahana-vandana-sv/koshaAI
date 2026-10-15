"""Shared answer parsing: one parser for every model and stage (CLAUDE.md)."""

import json
import re
from collections.abc import Sequence

from kosha import PUBMEDQA_LABELS

_decoder = json.JSONDecoder()


def parse_answer(text: str, labels: Sequence[str] = PUBMEDQA_LABELS) -> tuple[str | None, str]:
    """Return (label, mode). mode is "json" (clean), "fallback" (label found in free text) or "invalid".

    Invalid outputs count as wrong; the mode is logged so format failures are reported separately.
    """
    canon = {label.lower(): label for label in labels}

    # 1) First JSON object anywhere in the text with a valid "answer" field.
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            obj, _ = _decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            answer = str(obj.get("answer", "")).strip().lower()
            if answer in canon:
                return canon[answer], "json"

    # 2) Fallback: first standalone label word. Single letters (A-D) are matched case-sensitively,
    #    otherwise the article "a" would count as an answer.
    flags = 0 if all(len(label) == 1 for label in labels) else re.IGNORECASE
    m = re.search(rf"\b({'|'.join(map(re.escape, labels))})\b", text, flags)
    if m:
        return canon[m.group(1).lower()], "fallback"
    return None, "invalid"
