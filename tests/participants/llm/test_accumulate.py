"""Contract tests for ``accumulate`` (DESIGN2 §6.1, §8).

No model, no network, no journal, no engine -- exactly the modularity claim
DESIGN2 §8 asks a standalone test to demonstrate: ``accumulate`` is exercised
with nothing but a synthetic ``Delta`` sequence.
"""

import json
from typing import AsyncGenerator, AsyncIterator, cast

import pytest

from patchbay_llm.events import Event, EventId, Media, Thought, ToolCall, merge
from patchbay_llm.infer_engine.delta import (
    CallDelta,
    Delta,
    Finish,
    Stop,
    TextDelta,
    ThoughtDelta,
    Usage,
)
from patchbay_llm.participants.llm.accumulate import accumulate


def _usage(model: str = "m") -> Usage:
    return Usage(
        model=model, input=10, cache_read=0, cache_write=0, output=5, thought=0
    )


async def _stream(deltas: list[Delta]) -> AsyncIterator[Delta]:
    for d in deltas:
        yield d


async def _drain(deltas: list[Delta], author: str = "llm:main") -> list[Event]:
    return [e async for e in accumulate(_stream(deltas), author)]


async def test_text_accumulates_cumulatively_and_completes_on_finish() -> None:
    """Text deltas grow one cumulative ``message`` event, promoted on Finish."""
    events = await _drain(
        [
            TextDelta(0, "Let me"),
            TextDelta(0, " check the"),
            TextDelta(0, " config."),
            Finish(Stop.END, _usage()),
        ]
    )
    msgs = [e for e in events if e.kind == "message"]
    texts: list[bytes] = []
    for m in msgs:
        block = m.content[0]
        assert isinstance(block, Media)
        texts.append(block.data)
    # 3 growing partials, then one more event when Finish promotes the slot
    # to complete=True -- same content, different completeness (DESIGN2 §3:
    # "the journal receives one event per id, when its stream ends").
    assert texts == [
        b"Let me",
        b"Let me check the",
        b"Let me check the config.",
        b"Let me check the config.",
    ]
    assert [m.complete for m in msgs] == [False, False, False, True]
    assert len({m.id for m in msgs}) == 1  # one id for the whole slot
    assert events[-1].kind == "usage"
    assert events[-1].complete is True


async def test_distinct_slots_get_distinct_ids_and_kinds() -> None:
    """A thought slot and a text slot get separate ids, kinds and content."""
    events = await _drain(
        [
            ThoughtDelta(0, "hmm"),
            TextDelta(1, "ok"),
            ThoughtDelta(0, " let's see", signature="sig123"),
            Finish(Stop.END, _usage()),
        ]
    )
    thoughts = [e for e in events if e.kind == "thought"]
    messages = [e for e in events if e.kind == "message"]
    assert len({e.id for e in thoughts}) == 1
    assert len({e.id for e in messages}) == 1
    assert thoughts[0].id != messages[0].id

    final = thoughts[-1]
    assert final.complete is True
    block = final.content[0]
    assert isinstance(block, Thought)
    assert block.text == "hmm let's see"
    assert block.signature == "sig123"


async def test_tool_call_args_are_a_string_and_never_parsed_here() -> None:
    """Partial arguments must never become executable (DESIGN2 §3)."""
    events = await _drain(
        [
            CallDelta(0, id="call_1", tool="get_weather"),
            CallDelta(0, args='{"city": "Pa'),
            CallDelta(0, args='ris"}'),
            Finish(Stop.TOOLS, _usage()),
        ]
    )
    calls = [e for e in events if e.kind == "tool_call"]
    blocks: list[ToolCall] = []
    for e in calls:
        block = e.content[0]
        assert isinstance(block, ToolCall)
        blocks.append(block)

    # The middle partial's args is a fragment -- not valid JSON on its own.
    with pytest.raises(json.JSONDecodeError):
        json.loads(blocks[1].args)

    # Only the final, complete event carries the whole, valid fragment --
    # and even then, accumulate itself never parses it.
    final = calls[-1]
    assert final.complete is True
    final_block = final.content[0]
    assert isinstance(final_block, ToolCall)
    assert isinstance(final_block.args, str)
    assert json.loads(final_block.args) == {"city": "Paris"}


async def test_meta_author_is_attached_to_every_event() -> None:
    """``author`` becomes ``meta.author`` on every emitted event."""
    events = await _drain(
        [TextDelta(0, "hi"), Finish(Stop.END, _usage())], author="llm:main"
    )
    for e in events:
        assert e.meta["author"] == "llm:main"


async def test_usage_event_carries_raw_provider_counts() -> None:
    """The terminal usage event carries the provider's raw counts, unmodified."""
    events = await _drain(
        [TextDelta(0, "hi"), Finish(Stop.END, _usage(model="claude"))]
    )
    usage = events[-1]
    assert usage.kind == "usage"
    assert usage.meta["model"] == "claude"
    assert usage.meta["input"] == 10
    assert usage.meta["output"] == 5
    assert usage.complete is True


async def test_stream_exhausted_without_finish_raises() -> None:
    """infer_engine's own contract forbids a silent exhaustion (INFERENCE.md
    §5); accumulate must not paper over an engine that violates it."""
    with pytest.raises(RuntimeError):
        await _drain([TextDelta(0, "hi")])


async def test_consumer_walking_away_leaves_every_partial_structurally_valid() -> None:
    """DESIGN2 §8 (the accumulate row) + 'Surrender is free': interruption is
    the *consumer* stopping mid-stream (``aclose()``), not the producer
    exhausting early -- that second case is a distinct engine bug, already
    covered by ``test_stream_exhausted_without_finish_raises``.

    Stop consuming at every possible point strictly before ``Finish`` and
    assert that whatever partials were seen for each id fold via
    ``events.merge`` with no error and end ``complete=False`` -- exactly what
    a dying stream leaves behind, with nothing to flush and nothing to clean
    up."""
    deltas: list[Delta] = [
        ThoughtDelta(0, "thinking"),
        TextDelta(1, "The weather in "),
        CallDelta(2, id="call_1", tool="get_weather"),
        CallDelta(2, args='{"city": "Pa'),
        CallDelta(2, args='ris"}'),
        TextDelta(1, "Paris is nice."),
        Finish(Stop.END, _usage()),
    ]
    # One event per non-Finish delta, in order; stop consuming at every point
    # strictly before the generator would reach Finish's own yields.
    stoppable_yields = len(deltas) - 1

    for n in range(1, stoppable_yields + 1):
        gen = accumulate(_stream(deltas), "llm:main")
        seen: list[Event] = []
        for _ in range(n):
            seen.append(await gen.__anext__())
        await cast(AsyncGenerator[Event, None], gen).aclose()  # consumer walks away

        by_id: dict[EventId, list[Event]] = {}
        for e in seen:
            by_id.setdefault(e.id, []).append(e)

        for partials in by_id.values():
            merged = partials[0]
            for update in partials[1:]:
                merged = merge(merged, update)  # must never raise
            assert merged is partials[-1]
            assert merged.complete is False  # Finish was never reached
