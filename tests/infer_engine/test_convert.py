"""Contract tests for ``infer_engine.convert`` (the journal <-> inference seam)."""

import pytest

from patchbay_llm.events import Media as JournalMedia
from patchbay_llm.events import Thought as JournalThought
from patchbay_llm.events import ToolCall as JournalToolCall
from patchbay_llm.events import ToolResult as JournalToolResult
from patchbay_llm.infer_engine.convert import from_part, to_part
from patchbay_llm.infer_engine.prompt import Breakpoint, Call, Result
from patchbay_llm.infer_engine.prompt import Thought as InferThought


def test_media_passes_through_unchanged() -> None:
    """Media is the one shared type; conversion is an identity in both directions."""
    m = JournalMedia("text/plain", b"hello")
    assert to_part(m) is m
    assert from_part(m) is m


def test_thought_round_trips() -> None:
    """A journal Thought and an inference Thought carry the same fields."""
    jt = JournalThought("reasoning...", signature="sig")
    part = to_part(jt)
    assert isinstance(part, InferThought)
    assert part.text == "reasoning..." and part.signature == "sig"
    back = from_part(part)
    assert isinstance(back, JournalThought)
    assert back == jt


def test_tool_call_args_are_parsed_only_by_to_part() -> None:
    """``to_part`` parses ``args``; ``from_part`` re-serialises it as a string."""
    jc = JournalToolCall(id="c1", tool="get_weather", args='{"city": "Paris"}')
    part = to_part(jc)
    assert isinstance(part, Call)
    assert part.id == "c1" and part.tool == "get_weather"
    assert part.args == {"city": "Paris"}

    back = from_part(part)
    assert isinstance(back, JournalToolCall)
    assert back.id == "c1" and back.tool == "get_weather"
    assert back.args == '{"city": "Paris"}'


def test_tool_call_with_empty_args() -> None:
    """An empty ``args`` string parses to an empty mapping, not an error."""
    jc = JournalToolCall(id="c1", tool="noop", args="")
    part = to_part(jc)
    assert isinstance(part, Call)
    assert part.args == {}


def test_tool_result_round_trips() -> None:
    """A journal ToolResult and an inference Result carry the same fields."""
    jr = JournalToolResult(
        call="c1", content=(JournalMedia("text/plain", b"18C, cloudy"),), failed=False
    )
    part = to_part(jr)
    assert isinstance(part, Result)
    assert part.call == "c1" and part.failed is False

    back = from_part(part)
    assert isinstance(back, JournalToolResult)
    assert back == jr


def test_breakpoint_has_no_journal_equivalent() -> None:
    """Breakpoint is a prompt-assembly concept, not recorded content."""
    with pytest.raises(TypeError):
        from_part(Breakpoint())
