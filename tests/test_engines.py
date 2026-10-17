import math
from types import SimpleNamespace

import pytest

from kosha import MEDMCQA_LABELS
from kosha.eval.engines import label_probs


def lp(token: str, p: float):
    return SimpleNamespace(decoded_token=token, logprob=math.log(p))


def test_reads_probs_at_the_answer_token():
    steps = [
        {1: lp('{"', 0.99)},
        {2: lp("answer", 0.99)},
        {3: lp('":', 0.99)},
        {4: lp(' "', 0.99)},
        {5: lp("yes", 0.70), 6: lp("no", 0.20), 7: lp("maybe", 0.05), 8: lp("Yes", 0.05)},
    ]
    assert label_probs(steps) == pytest.approx([0.75, 0.20, 0.05])


def test_handles_quote_glued_tokens_and_letters():
    steps = [{1: lp(' "B', 0.6), 2: lp(' "C', 0.4)}]
    assert label_probs(steps, MEDMCQA_LABELS) == pytest.approx([0.0, 0.6, 0.4, 0.0])


def test_none_when_no_label_is_generated():
    assert label_probs([{1: lp("I", 0.9)}, {2: lp(" cannot", 0.9)}]) is None
