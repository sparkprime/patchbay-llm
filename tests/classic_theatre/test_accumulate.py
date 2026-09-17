"""Contract tests for ``accumulate`` (DESIGN2 §6.1, §8).

No model, no network, no journal, no engine -- exactly the modularity claim
DESIGN2 §8 asks a standalone test to demonstrate: ``accumulate`` is exercised
with nothing but a synthetic ``Delta`` sequence.
"""

import json
from typing import AsyncIterator

import pytest

from patchbay_llm.classic_theatre.conversation import ASSISTANT
from patchbay_llm.classic_theatre.repl_theatre import (
    accumulate,
)
from patchbay_llm.events import Event, EventId, Media, Thought, ToolCall, ToolResult
from patchbay_llm.infer_engine.delta import (
    CallDelta,
    Delta,
    Finish,
    Stop,
    TextDelta,
    ThoughtDelta,
    Usage,
)
from patchbay_llm.live import (
    LiveUpdate,
    MessageChunk,
    ThoughtChunk,
    ToolCallChunk,
    ToolResultChunk,
)
from patchbay_llm.reify import reify_block


def _usage(model: str = "m") -> Usage:
    return Usage(
        model=model, input=10, cache_read=0, cache_write=0, output=5, thought=0
    )


async def _stream(deltas: list[Delta]) -> AsyncIterator[Delta]:
    for d in deltas:
        yield d


async def _drain(
    deltas: list[Delta],
) -> list[Event | LiveUpdate]:
    return [e async for e in accumulate(_stream(deltas))]


# ── reify_block: the free-standing join function ───────────────────────────


def _update(eid: EventId, kind: str, **kwargs: object) -> LiveUpdate:
    if kind == "message":
        return MessageChunk(id=eid, **kwargs)  # type: ignore[arg-type]
    if kind == "thought":
        return ThoughtChunk(id=eid, **kwargs)  # type: ignore[arg-type]
    if kind == "tool_call":
        return ToolCallChunk(id=eid, **kwargs)  # type: ignore[arg-type]
    if kind == "tool_result":
        return ToolResultChunk(id=eid, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown kind: {kind!r}")


def test_reify_block_text() -> None:
    """Text chunks join into a Media block."""
    eid = EventId("e1")
    updates = [
        _update(eid, "message", text="Let me"),
        _update(eid, "message", text=" check the"),
        _update(eid, "message", text=" config."),
    ]
    block = reify_block("message", updates)
    assert isinstance(block, Media)
    assert block.data == b"Let me check the config."


def test_reify_block_thought() -> None:
    """Thought chunks join into a Thought block, with the last signature."""
    eid = EventId("e1")
    updates = [
        _update(eid, "thought", text="hmm"),
        _update(eid, "thought", text=" let's see", signature="sig123"),
    ]
    block = reify_block("thought", updates)
    assert isinstance(block, Thought)
    assert block.text == "hmm let's see"
    assert block.signature == "sig123"


def test_reify_block_tool_call() -> None:
    """Call chunks join into a ToolCall with accumulated args (still a string)."""
    eid = EventId("e1")
    updates = [
        _update(eid, "tool_call", call_id="call_1", tool="get_weather"),
        _update(eid, "tool_call", args='{"city": "Pa'),
        _update(eid, "tool_call", args='ris"}'),
    ]
    block = reify_block("tool_call", updates)
    assert isinstance(block, ToolCall)
    assert block.id == "call_1"
    assert block.tool == "get_weather"
    assert isinstance(block.args, str)
    assert json.loads(block.args) == {"city": "Paris"}


def test_reify_block_empty() -> None:
    """An empty update list still produces a valid block."""
    block = reify_block("message", [])
    assert isinstance(block, Media)
    assert block.data == b""


def test_reify_block_unknown_kind_raises() -> None:
    """An unknown kind raises ValueError."""
    with pytest.raises(ValueError, match="unknown slot kind"):
        reify_block("nope", [])


def test_reify_block_tool_result() -> None:
    """Result chunks join into a ToolResult with accumulated text and failed flag."""
    eid = EventId("e1")
    updates = [
        _update(eid, "tool_result", call_id="call_1", text="hello "),
        _update(eid, "tool_result", text="world"),
        _update(eid, "tool_result", failed=True),
    ]
    block = reify_block("tool_result", updates)
    assert isinstance(block, ToolResult)
    assert block.call == "call_1"
    assert block.failed is True
    assert len(block.content) == 1
    assert isinstance(block.content[0], Media)
    assert block.content[0].data == b"hello world"


def test_reify_block_tool_result_empty() -> None:
    """An empty result list produces a ToolResult with empty content."""
    block = reify_block("tool_result", [])
    assert isinstance(block, ToolResult)
    assert block.failed is False
    assert block.content[0].data == b""


# ── accumulate: the basic flow ─────────────────────────────────────────────


async def test_text_yields_updates_then_final_event_on_finish() -> None:
    """One LiveUpdate per delta, then one complete Event + usage on Finish."""
    items = await _drain(
        [
            TextDelta(0, "Let me"),
            TextDelta(0, " check the"),
            TextDelta(0, " config."),
            Finish(Stop.END, _usage()),
        ]
    )
    # 3 LiveUpdates + 1 final Event + 1 usage Event
    assert len(items) == 5
    for i in range(3):
        assert isinstance(items[i], MessageChunk)
        assert items[i].text

    final = items[3]
    assert isinstance(final, Event)
    assert final.kind == "message"
    assert final.complete is True
    block = final.content[0]
    assert isinstance(block, Media)
    assert block.data == b"Let me check the config."

    usage = items[4]
    assert isinstance(usage, Event)
    assert usage.kind == "usage"
    assert usage.complete is True


async def test_updates_share_id_with_final_event() -> None:
    """LiveUpdate.id matches the final Event.id for the same slot."""
    items = await _drain(
        [
            TextDelta(0, "hello"),
            Finish(Stop.END, _usage()),
        ]
    )
    updates = [i for i in items if isinstance(i, LiveUpdate)]
    finals = [i for i in items if isinstance(i, Event) and i.kind != "usage"]
    assert len(updates) == 1
    assert len(finals) == 1
    assert updates[0].id == finals[0].id


async def test_distinct_slots_get_distinct_ids_and_variants() -> None:
    """A thought slot and a text slot get separate ids and distinct variants."""
    items = await _drain(
        [
            ThoughtDelta(0, "hmm"),
            TextDelta(1, "ok"),
            ThoughtDelta(0, " let's see", signature="sig123"),
            Finish(Stop.END, _usage()),
        ]
    )
    updates = [i for i in items if isinstance(i, LiveUpdate)]
    finals = [i for i in items if isinstance(i, Event) and i.kind != "usage"]

    # 3 LiveUpdates: ThoughtChunk(0), MessageChunk(1), ThoughtChunk(0)
    assert len(updates) == 3
    assert isinstance(updates[0], ThoughtChunk)
    assert isinstance(updates[1], MessageChunk)
    assert updates[0].id != updates[1].id

    # 2 final Events
    assert len(finals) == 2
    thought_final = [f for f in finals if f.kind == "thought"][0]
    assert thought_final.complete is True
    block = thought_final.content[0]
    assert isinstance(block, Thought)
    assert block.text == "hmm let's see"
    assert block.signature == "sig123"


async def test_tool_call_args_are_a_string_and_never_parsed_here() -> None:
    """Partial arguments must never become executable (DESIGN2 §3)."""
    deltas: list[Delta] = [
        CallDelta(0, id="call_1", tool="get_weather"),
        CallDelta(0, args='{"city": "Pa'),
        CallDelta(0, args='ris"}'),
    ]

    # A prefix produces fragment args (not valid JSON).
    eid = EventId("e1")
    prefix_updates = [
        _update(eid, "tool_call", call_id="call_1", tool="get_weather"),
        _update(eid, "tool_call", args='{"city": "Pa'),
    ]
    snap = reify_block("tool_call", prefix_updates)
    assert isinstance(snap, ToolCall)
    with pytest.raises(json.JSONDecodeError):
        json.loads(snap.args)

    # The full stream produces valid JSON, but accumulate never parses it.
    items = await _drain(deltas + [Finish(Stop.TOOLS, _usage())])
    finals = [i for i in items if isinstance(i, Event) and i.kind == "tool_call"]
    final = finals[0]
    assert final.complete is True
    block = final.content[0]
    assert isinstance(block, ToolCall)
    assert isinstance(block.args, str)
    assert json.loads(block.args) == {"city": "Paris"}


async def test_meta_author_is_assistant_on_every_event() -> None:
    """The assistant is ``meta.author`` on every emitted event."""
    items = await _drain([TextDelta(0, "hi"), Finish(Stop.END, _usage())])
    for item in items:
        if isinstance(item, Event):
            assert item.meta["author"] == ASSISTANT


async def test_usage_event_carries_raw_provider_counts() -> None:
    """The terminal usage event carries the provider's raw counts, unmodified."""
    items = await _drain([TextDelta(0, "hi"), Finish(Stop.END, _usage(model="claude"))])
    usage = [i for i in items if isinstance(i, Event) and i.kind == "usage"][0]
    assert usage.meta["model"] == "claude"
    assert usage.meta["input"] == 10
    assert usage.meta["output"] == 5
    assert "stop" not in usage.meta
    assert "detail" not in usage.meta
    assert usage.complete is True


async def test_stop_and_detail_are_on_content_events_not_usage() -> None:
    """The stop reason and detail land on the content events, not the usage event."""
    items = await _drain(
        [TextDelta(0, "hi"), Finish(Stop.END, _usage(), detail="stop_sequence")]
    )
    final = [i for i in items if isinstance(i, Event) and i.kind == "message"][0]
    assert final.meta["stop"] == "END"
    assert final.meta["detail"] == "stop_sequence"


async def test_stream_exhausted_without_finish_raises() -> None:
    """infer_engine's own contract forbids a silent exhaustion."""
    with pytest.raises(RuntimeError):
        await _drain([TextDelta(0, "hi")])


# ── Reify after any prefix is structurally valid ───────────────────────────


def test_reify_block_after_any_prefix_is_structurally_valid() -> None:
    """Joining updates after any prefix produces a structurally valid block."""
    eid = EventId("e1")
    all_updates = [
        _update(eid, "thought", text="thinking"),
        _update(eid, "message", text="The weather in "),
        _update(eid, "tool_call", call_id="call_1", tool="get_weather"),
        _update(eid, "tool_call", args='{"city": "Pa'),
        _update(eid, "tool_call", args='ris"}'),
        _update(eid, "tool_result", call_id="call_1", text="Paris: 18°C"),
        _update(eid, "tool_result", failed=False),
        _update(eid, "message", text="Paris is nice."),
    ]
    for n in range(1, len(all_updates) + 1):
        prefix = all_updates[:n]
        by_kind: dict[str, list[LiveUpdate]] = {}
        for u in prefix:
            kind = {
                MessageChunk: "message",
                ThoughtChunk: "thought",
                ToolCallChunk: "tool_call",
                ToolResultChunk: "tool_result",
            }[type(u)]
            by_kind.setdefault(kind, []).append(u)
        for kind, kind_updates in by_kind.items():
            block = reify_block(kind, kind_updates)
            if kind == "message":
                assert isinstance(block, Media)
            elif kind == "thought":
                assert isinstance(block, Thought)
            elif kind == "tool_call":
                assert isinstance(block, ToolCall)
            elif kind == "tool_result":
                assert isinstance(block, ToolResult)


# ── LiveUpdate carries per-variant fields, not infer_engine types ──────────


async def test_message_chunk_carries_text_increment() -> None:
    """A TextDelta becomes a MessageChunk with text populated."""
    items = await _drain([TextDelta(0, "hello"), Finish(Stop.END, _usage())])
    u = [i for i in items if isinstance(i, MessageChunk)][0]
    assert u.text == "hello"


async def test_thought_chunk_carries_signature() -> None:
    """A ThoughtDelta becomes a ThoughtChunk with text and signature."""
    items = await _drain(
        [ThoughtDelta(0, "hmm", signature="sig"), Finish(Stop.END, _usage())]
    )
    u = [i for i in items if isinstance(i, ThoughtChunk)][0]
    assert u.text == "hmm"
    assert u.signature == "sig"


async def test_tool_call_chunk_carries_call_id_tool_args() -> None:
    """A CallDelta becomes a ToolCallChunk with call_id, tool, args."""
    items = await _drain(
        [
            CallDelta(0, id="c1", tool="run", args='{"x":1}'),
            Finish(Stop.TOOLS, _usage()),
        ]
    )
    u = [i for i in items if isinstance(i, ToolCallChunk)][0]
    assert u.call_id == "c1"
    assert u.tool == "run"
    assert u.args == '{"x":1}'
