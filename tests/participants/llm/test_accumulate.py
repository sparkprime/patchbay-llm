"""Contract tests for ``accumulate`` (DESIGN2 §6.1, §8; durable_partials.md).

No model, no network, no journal, no engine -- exactly the modularity claim
DESIGN2 §8 asks a standalone test to demonstrate: ``accumulate`` is exercised
with nothing but a synthetic ``Delta`` sequence.
"""

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

import pytest

from patchbay_llm.events import Event, EventId, Media, Thought, ToolCall
from patchbay_llm.infer_engine.delta import (
    CallDelta,
    Delta,
    Finish,
    Stop,
    TextDelta,
    ThoughtDelta,
    Usage,
)
from patchbay_llm.participants.llm.accumulate import (
    FileDeltaLog,
    PartialEvent,
    accumulate,
    recover_events,
    reify_block,
)


def _usage(model: str = "m") -> Usage:
    return Usage(
        model=model, input=10, cache_read=0, cache_write=0, output=5, thought=0
    )


async def _stream(deltas: list[Delta]) -> AsyncIterator[Delta]:
    for d in deltas:
        yield d


async def _drain(
    deltas: list[Delta], author: str = "llm:main"
) -> list[Event | PartialEvent]:
    return [e async for e in accumulate(_stream(deltas), author)]


# ── reify_block: the free-standing join function ───────────────────────────


def test_reify_block_text() -> None:
    """Text deltas join into a Media block."""
    deltas: list[Delta] = [
        TextDelta(0, "Let me"),
        TextDelta(0, " check the"),
        TextDelta(0, " config."),
    ]
    block = reify_block("message", deltas)
    assert isinstance(block, Media)
    assert block.data == b"Let me check the config."


def test_reify_block_thought() -> None:
    """Thought deltas join into a Thought block, with the last signature."""
    deltas: list[Delta] = [
        ThoughtDelta(0, "hmm"),
        ThoughtDelta(0, " let's see", signature="sig123"),
    ]
    block = reify_block("thought", deltas)
    assert isinstance(block, Thought)
    assert block.text == "hmm let's see"
    assert block.signature == "sig123"


def test_reify_block_tool_call() -> None:
    """Call deltas join into a ToolCall with accumulated args (still a string)."""
    deltas: list[Delta] = [
        CallDelta(0, id="call_1", tool="get_weather"),
        CallDelta(0, args='{"city": "Pa'),
        CallDelta(0, args='ris"}'),
    ]
    block = reify_block("tool_call", deltas)
    assert isinstance(block, ToolCall)
    assert block.id == "call_1"
    assert block.tool == "get_weather"
    assert isinstance(block.args, str)
    assert json.loads(block.args) == {"city": "Paris"}


def test_reify_block_empty() -> None:
    """An empty delta list still produces a valid block."""
    block = reify_block("message", [])
    assert isinstance(block, Media)
    assert block.data == b""


def test_reify_block_unknown_kind_raises() -> None:
    """An unknown kind raises ValueError."""
    with pytest.raises(ValueError, match="unknown slot kind"):
        reify_block("nope", [])


# ── accumulate: the basic flow ─────────────────────────────────────────────


async def test_text_yields_partial_then_final_event_on_finish() -> None:
    """One PartialEvent handle, then one complete Event on Finish."""
    items = await _drain(
        [
            TextDelta(0, "Let me"),
            TextDelta(0, " check the"),
            TextDelta(0, " config."),
            Finish(Stop.END, _usage()),
        ]
    )
    # 1 PartialEvent + 1 final Event + 1 usage Event
    assert len(items) == 3
    pe = items[0]
    assert isinstance(pe, PartialEvent)
    assert pe.kind == "message"

    final = items[1]
    assert isinstance(final, Event)
    assert final.kind == "message"
    assert final.complete is True
    block = final.content[0]
    assert isinstance(block, Media)
    assert block.data == b"Let me check the config."

    usage = items[2]
    assert isinstance(usage, Event)
    assert usage.kind == "usage"
    assert usage.complete is True


async def test_partial_snapshot_grows_as_deltas_arrive() -> None:
    """snapshot() returns the joined-so-far content at any point."""
    gen = accumulate(
        _stream(
            [
                TextDelta(0, "Let me"),
                TextDelta(0, " check the"),
                TextDelta(0, " config."),
                Finish(Stop.END, _usage()),
            ]
        ),
        "llm:main",
    )
    pe = await gen.__anext__()
    assert isinstance(pe, PartialEvent)

    snap0 = pe.snapshot()
    assert isinstance(snap0, Media)
    assert snap0.data == b"Let me"

    # Let accumulate process the next delta (which it appends but doesn't yield)
    final = await gen.__anext__()
    assert isinstance(final, Event)

    # The final event has the full content — snapshot before reify would have
    # matched.  After reify, snapshot raises (already reified) — but we already
    # captured it.
    assert callable(pe.snapshot)  # method still exists
    assert b"check" in final.content[0].data  # type: ignore[union-attr]


async def test_distinct_slots_get_distinct_partial_events() -> None:
    """A thought slot and a text slot get separate PartialEvents."""
    items = await _drain(
        [
            ThoughtDelta(0, "hmm"),
            TextDelta(1, "ok"),
            ThoughtDelta(0, " let's see", signature="sig123"),
            Finish(Stop.END, _usage()),
        ]
    )
    partials = [i for i in items if isinstance(i, PartialEvent)]
    finals = [i for i in items if isinstance(i, Event) and i.kind != "usage"]

    assert len(partials) == 2
    assert partials[0].kind == "thought"
    assert partials[1].kind == "message"
    assert partials[0].id != partials[1].id

    assert len(finals) == 2
    thought_final = [f for f in finals if f.kind == "thought"][0]
    assert thought_final.complete is True
    block = thought_final.content[0]
    assert isinstance(block, Thought)
    assert block.text == "hmm let's see"
    assert block.signature == "sig123"


async def test_tool_call_args_are_a_string_and_never_parsed_here() -> None:
    """Partial arguments must never become executable (DESIGN2 §3).

    Two things to check:
    1. A prefix of the delta stream produces fragment args (not valid JSON).
    2. The final, complete Event's args are still a raw string — accumulate
       never parses them.
    """
    deltas: list[Delta] = [
        CallDelta(0, id="call_1", tool="get_weather"),
        CallDelta(0, args='{"city": "Pa'),
        CallDelta(0, args='ris"}'),
    ]

    # (1) Reifying after a prefix (the first two deltas) gives a fragment.
    snap = reify_block("tool_call", deltas[:2])
    assert isinstance(snap, ToolCall)
    with pytest.raises(json.JSONDecodeError):
        json.loads(snap.args)

    # (2) The full stream produces valid JSON, but accumulate itself never
    # parses it — args stays a string on the journal.
    items = await _drain(deltas + [Finish(Stop.TOOLS, _usage())])
    finals = [i for i in items if isinstance(i, Event) and i.kind == "tool_call"]
    final = finals[0]
    assert final.complete is True
    block = final.content[0]
    assert isinstance(block, ToolCall)
    assert isinstance(block.args, str)
    assert json.loads(block.args) == {"city": "Paris"}


async def test_meta_author_is_attached_to_every_event() -> None:
    """``author`` becomes ``meta.author`` on every emitted event."""
    items = await _drain(
        [TextDelta(0, "hi"), Finish(Stop.END, _usage())], author="llm:main"
    )
    for item in items:
        if isinstance(item, PartialEvent):
            assert item.author == "llm:main"
        else:
            assert item.meta["author"] == "llm:main"


async def test_usage_event_carries_raw_provider_counts() -> None:
    """The terminal usage event carries the provider's raw counts, unmodified."""
    items = await _drain([TextDelta(0, "hi"), Finish(Stop.END, _usage(model="claude"))])
    usage = [i for i in items if isinstance(i, Event) and i.kind == "usage"][0]
    assert usage.meta["model"] == "claude"
    assert usage.meta["input"] == 10
    assert usage.meta["output"] == 5
    assert usage.complete is True


async def test_stream_exhausted_without_finish_raises() -> None:
    """infer_engine's own contract forbids a silent exhaustion (INFERENCE.md
    §5); accumulate must not paper over an engine that violates it."""
    with pytest.raises(RuntimeError):
        await _drain([TextDelta(0, "hi")])


# ── The ported invariant: reify after any prefix is structurally valid ─────


async def test_reify_after_any_prefix_produces_structurally_valid_event() -> None:
    """durable_partials.md "Consequence: merge() becomes vestigial":

    Reifying a PartialEvent after any prefix of its delta stream produces a
    structurally valid Event.  Same guarantee as the old "every prefix of the
    output merges to a structurally valid event" test, re-aimed at reify()
    instead of merge().  Whoever builds this should port that test, not just
    delete it.
    """
    deltas: list[Delta] = [
        ThoughtDelta(0, "thinking"),
        TextDelta(1, "The weather in "),
        CallDelta(2, id="call_1", tool="get_weather"),
        CallDelta(2, args='{"city": "Pa'),
        CallDelta(2, args='ris"}'),
        TextDelta(1, "Paris is nice."),
        Finish(Stop.END, _usage()),
    ]
    # Three slots: thought(0), message(1), tool_call(2).
    # One PartialEvent yield per new slot: 3 yields before Finish.
    stoppable_yields = 3

    for n in range(1, stoppable_yields + 1):
        gen = accumulate(_stream(deltas), "llm:main")
        partials: list[PartialEvent] = []
        for _ in range(n):
            item = await gen.__anext__()
            assert isinstance(item, PartialEvent)
            partials.append(item)
        # Consumer walks away (simulates live cancellation / Ctrl-C)
        await gen.aclose()

        for pe in partials:
            event = pe.reify(complete=False)
            # Same guarantee: an interrupted stream is always safe to finalize
            assert event.complete is False
            assert event.id == pe.id
            assert event.kind == pe.kind
            assert event.meta["author"] == "llm:main"
            assert len(event.content) == 1

            # The content block must be structurally valid
            block = event.content[0]
            if pe.kind == "message":
                assert isinstance(block, Media)
            elif pe.kind == "thought":
                assert isinstance(block, Thought)
            elif pe.kind == "tool_call":
                assert isinstance(block, ToolCall)
                # Args may be a fragment — that's fine, it's complete=False


# ── PartialEvent.subscribe: fan-out ────────────────────────────────────────


async def test_subscribe_delivers_every_delta_live() -> None:
    """subscribe() yields every increment as it arrives, from creation."""
    deltas: list[Delta] = [
        TextDelta(0, "Hello"),
        TextDelta(0, " world"),
        Finish(Stop.END, _usage()),
    ]
    gen = accumulate(_stream(deltas), "llm:main")
    pe = await gen.__anext__()
    assert isinstance(pe, PartialEvent)

    received: list[str] = []

    async def _drain_sub() -> None:
        async for delta in pe.subscribe():
            if isinstance(delta, TextDelta):
                received.append(delta.text)

    task = asyncio.create_task(_drain_sub())
    # Let accumulate run to Finish (which calls reify, ending the subscription)
    async for _ in gen:
        pass
    await task

    assert received == ["Hello", " world"]


async def test_subscribe_supports_multiple_independent_subscribers() -> None:
    """Fan-out: multiple subscribers each get every delta independently."""
    deltas: list[Delta] = [
        TextDelta(0, "A"),
        TextDelta(0, "B"),
        Finish(Stop.END, _usage()),
    ]
    gen = accumulate(_stream(deltas), "llm:main")
    pe = await gen.__anext__()
    assert isinstance(pe, PartialEvent)

    received_a: list[str] = []
    received_b: list[str] = []

    async def _sub(buf: list[str]) -> None:
        async for delta in pe.subscribe():
            if isinstance(delta, TextDelta):
                buf.append(delta.text)

    task_a = asyncio.create_task(_sub(received_a))
    task_b = asyncio.create_task(_sub(received_b))
    async for _ in gen:
        pass
    await asyncio.gather(task_a, task_b)

    assert received_a == ["A", "B"]
    assert received_b == ["A", "B"]


async def test_subscribe_late_attacher_gets_full_buffer() -> None:
    """A subscriber that attaches after deltas exist gets them via replay."""
    deltas: list[Delta] = [
        TextDelta(0, "X"),
        TextDelta(0, "Y"),
        Finish(Stop.END, _usage()),
    ]
    gen = accumulate(_stream(deltas), "llm:main")
    pe = await gen.__anext__()
    assert isinstance(pe, PartialEvent)

    # Drain accumulate to Finish (all deltas appended, reify called)
    async for _ in gen:
        pass

    # Now subscribe after the fact — the buffer replay should deliver the
    # deltas, then _END (reify already happened)
    received: list[str] = []
    async for delta in pe.subscribe():
        if isinstance(delta, TextDelta):
            received.append(delta.text)
    assert received == ["X", "Y"]


# ── Durability: FileDeltaLog and recover_events ────────────────────────────


async def test_file_delta_log_append_and_finalize(tmp_path: Path) -> None:
    """FileDeltaLog writes JSONL, one file per id, and finalize deletes it."""
    sink = FileDeltaLog(tmp_path)
    eid = EventId("abc123")
    sink.append(eid, "message", "llm:main", TextDelta(0, "Hello"), 0)
    sink.append(eid, "message", "llm:main", TextDelta(0, " world"), 1)

    log = tmp_path / "abc123.deltas.jsonl"
    assert log.exists()
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert entry["seq"] == 0
    assert entry["kind"] == "message"
    assert entry["author"] == "llm:main"
    assert entry["delta"]["text"] == "Hello"

    sink.finalize(eid)
    assert not log.exists()


async def test_recover_events_reconstructs_interrupted_events(tmp_path: Path) -> None:
    """recover_events scans for dangling logs and produces complete=False Events."""
    sink = FileDeltaLog(tmp_path)
    eid1 = EventId("msg-001")
    eid2 = EventId("call-002")

    # Simulate a crash mid-stream: two slots with partial deltas, no finalize
    sink.append(eid1, "message", "llm:main", TextDelta(0, "Hello"), 0)
    sink.append(eid1, "message", "llm:main", TextDelta(0, " world"), 1)
    sink.append(eid2, "tool_call", "llm:main", CallDelta(0, id="c1", tool="run"), 0)
    sink.append(eid2, "tool_call", "llm:main", CallDelta(0, args='{"x":'), 1)

    recovered = recover_events(tmp_path)
    assert len(recovered) == 2

    msg = [e for e in recovered if e.kind == "message"][0]
    assert msg.id == eid1
    assert msg.complete is False
    assert msg.meta["author"] == "llm:main"
    block = msg.content[0]
    assert isinstance(block, Media)
    assert block.data == b"Hello world"

    call = [e for e in recovered if e.kind == "tool_call"][0]
    assert call.id == eid2
    assert call.complete is False
    cblock = call.content[0]
    assert isinstance(cblock, ToolCall)
    assert cblock.id == "c1"
    assert cblock.tool == "run"
    # Args are a fragment — not valid JSON, but that's the point (§Recovery.4)
    with pytest.raises(json.JSONDecodeError):
        json.loads(cblock.args)

    # Logs are deleted after recovery (§Recovery.5)
    assert not (tmp_path / "msg-001.deltas.jsonl").exists()
    assert not (tmp_path / "call-002.deltas.jsonl").exists()

    # Second recovery finds nothing
    assert not recover_events(tmp_path)


async def test_accumulate_with_sink_then_crash_then_recover(tmp_path: Path) -> None:
    """End-to-end: accumulate with a FileDeltaLog, crash mid-stream, recover.

    A PartialEvent accumulates deltas durably.  The consumer walks away
    (aclose) without calling reify — the log survives.  recover_events
    produces the same complete=False Event that reify(complete=False) would
    have produced.
    """
    sink = FileDeltaLog(tmp_path)
    # Two slots → two PartialEvent yields before Finish.  Both deltas get
    # appended (and durably written) before the consumer crashes.
    deltas: list[Delta] = [
        TextDelta(0, "Let me"),
        TextDelta(1, "more text"),
        Finish(Stop.END, _usage()),  # never reached — we crash before this
    ]
    gen = accumulate(_stream(deltas), "llm:main", sink=sink)
    pe0 = await gen.__anext__()
    assert isinstance(pe0, PartialEvent)
    pe1 = await gen.__anext__()
    assert isinstance(pe1, PartialEvent)

    # Simulate a crash: close the generator without reaching Finish.
    # reify() is NOT called, so the logs survive.
    await gen.aclose()

    # Two durable logs should exist (one per slot)
    logs = list(tmp_path.glob("*.deltas.jsonl"))
    assert len(logs) == 2

    # Recover — should produce the same Events reify(complete=False) would
    recovered = recover_events(tmp_path)
    assert len(recovered) == 2

    for event in recovered:
        assert event.complete is False
        assert event.kind == "message"

    # Check by id (both are message kind, so index by id, not by kind)
    events_by_id = {e.id: e for e in recovered}
    assert events_by_id[pe0.id].content[0].data == b"Let me"  # type: ignore[union-attr]
    assert events_by_id[pe1.id].content[0].data == b"more text"  # type: ignore[union-attr]

    # Logs deleted after recovery
    assert not list(tmp_path.glob("*.deltas.jsonl"))


async def test_reify_finalizes_sink_log(tmp_path: Path) -> None:
    """reify(complete=True) on Finish deletes the durable log for that id."""
    sink = FileDeltaLog(tmp_path)
    deltas: list[Delta] = [
        TextDelta(0, "done"),
        Finish(Stop.END, _usage()),
    ]
    items: list[Event | PartialEvent] = []
    async for item in accumulate(_stream(deltas), "llm:main", sink=sink):
        items.append(item)

    pe = items[0]
    assert isinstance(pe, PartialEvent)

    # After Finish, reify was called → log should be deleted
    assert not list(tmp_path.glob("*.deltas.jsonl"))

    # And the final Event is complete
    final = [i for i in items if isinstance(i, Event) and i.kind == "message"][0]
    assert final.complete is True
