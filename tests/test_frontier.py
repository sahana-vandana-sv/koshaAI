import pytest

from kosha import MEDMCQA_LABELS
from kosha.data.prompts import SYSTEM_PUBMEDQA, pubmedqa_messages
from kosha.frontier.claude import cost_usd, params_for

ITEM = {"context": "ctx", "question": "q?", "label": "yes", "rationale": "r."}


def test_params_put_system_top_level_and_turn_thinking_off():
    p = params_for("claude-haiku-5-5", pubmedqa_messages(ITEM, shots=[ITEM]))
    assert p["system"] == SYSTEM_PUBMEDQA
    assert [m["role"] for m in p["messages"]] == ["user", "assistant", "user"]
    assert p["thinking"] == {"type": "disabled"}
    assert "temperature" not in p  # rejected by the 5.5 models (D-014)


def test_schema_enum_follows_labels():
    p = params_for("claude-sonnet-5-5", pubmedqa_messages(ITEM), MEDMCQA_LABELS)
    assert p["output_config"]["format"]["schema"]["properties"]["answer"]["enum"] == ["A", "B", "C", "D"]
    assert p["thinking"] == {"type": "between_tools"}


def test_batch_is_half_price():
    usage = {"input_tokens": 1_000_000, "output_tokens": 100_000}
    assert cost_usd("claude-sonnet-5-5", usage, batch=False) == pytest.approx(3.0)
    assert cost_usd("claude-sonnet-5-5", usage, batch=True) == pytest.approx(1.5)
