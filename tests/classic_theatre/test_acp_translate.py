"""Contract tests for ACP translation helpers."""

from patchbay_llm.classic_theatre.acp_theatre import (
    _event_to_acp,  # pyright: ignore[reportPrivateUsage]
)
from patchbay_llm.classic_theatre.acp_theatre import (
    _live_update_to_acp,  # pyright: ignore[reportPrivateUsage]
)
from patchbay_llm.classic_theatre.conversation import ASSISTANT, HUMAN
from patchbay_llm.events import Event, EventId, Media, new_id
from patchbay_llm.live import (
    MessageUpdate,
    ThoughtUpdate,
    ToolCallUpdate,
    ToolResultUpdate,
)


def test_live_update_message_chunk() -> None:
    """MessageChunk with text becomes agent_message_chunk."""
    result = _live_update_to_acp(MessageUpdate(id=EventId("mid1"), text="hello"))
    assert result is not None
    assert result["sessionUpdate"] == "agent_message_chunk"
    assert result["messageId"] == "mid1"
    assert result["content"]["type"] == "text"
    assert result["content"]["text"] == "hello"


def test_live_update_thought_chunk() -> None:
    """ThoughtChunk with text becomes agent_thought_chunk."""
    result = _live_update_to_acp(ThoughtUpdate(id=EventId("mid2"), text="thinking"))
    assert result is not None
    assert result["sessionUpdate"] == "agent_thought_chunk"
    assert result["messageId"] == "mid2"
    assert result["content"]["type"] == "text"
    assert result["content"]["text"] == "thinking"


def test_live_update_message_chunk_empty_text() -> None:
    """MessageChunk with empty text returns None."""
    result = _live_update_to_acp(MessageUpdate(id=EventId("mid3"), text=""))
    assert result is None


def test_live_update_tool_call_chunk() -> None:
    """ToolCallChunk returns None (not surfaced)."""
    result = _live_update_to_acp(
        ToolCallUpdate(id=EventId("mid4"), call_id="call1", tool="test")
    )
    assert result is None


def test_live_update_tool_result_chunk() -> None:
    """ToolResultChunk returns None (not surfaced)."""
    result = _live_update_to_acp(
        ToolResultUpdate(id=EventId("mid5"), call_id="call1", text="result")
    )
    assert result is None


def test_event_message_assistant() -> None:
    """Assistant message becomes agent_message."""
    eid = new_id()
    event = Event(
        id=eid,
        ts=0.0,
        kind="message",
        content=(Media("text/plain", b"hello"),),
        meta={"author": ASSISTANT},
        complete=True,
    )
    result = _event_to_acp(event)
    assert result is not None
    assert result["sessionUpdate"] == "agent_message"
    assert result["messageId"] == eid
    assert len(result["content"]) == 1
    assert result["content"][0]["type"] == "text"


def test_event_thought_assistant() -> None:
    """Assistant thought becomes agent_thought."""
    eid = new_id()
    event = Event(
        id=eid,
        ts=0.0,
        kind="thought",
        content=(Media("text/plain", b"thinking"),),
        meta={"author": ASSISTANT},
        complete=True,
    )
    result = _event_to_acp(event)
    assert result is not None
    assert result["sessionUpdate"] == "agent_thought"
    assert result["messageId"] == eid


def test_event_human_message() -> None:
    """Human message returns None (nothing to report)."""
    eid = new_id()
    event = Event(
        id=eid,
        ts=0.0,
        kind="message",
        content=(Media("text/plain", b"hello"),),
        meta={"author": HUMAN},
        complete=True,
    )
    result = _event_to_acp(event)
    assert result is None


def test_event_usage() -> None:
    """Usage event returns None (not surfaced)."""
    event = Event(
        id=new_id(),
        ts=0.0,
        kind="usage",
        meta={"author": ASSISTANT},
        complete=True,
    )
    result = _event_to_acp(event)
    assert result is None


def test_event_turn_end() -> None:
    """turn_end event returns None (not surfaced)."""
    event = Event(
        id=new_id(),
        ts=0.0,
        kind="turn_end",
        meta={"author": ASSISTANT, "reason": "relinquished"},
        complete=True,
    )
    result = _event_to_acp(event)
    assert result is None
