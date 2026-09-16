"""``accumulate``: provider deltas -> PartialEvent / Event.

``StreamEvent``/``Delta`` and ``Event`` are one step apart, and it is tempting
to delete the gap by having the model layer yield partial events directly. It
should not: assigning ids and accumulating cumulative content is *participant
policy*, whereas ``infer_engine`` exists purely to normalise provider chunk
shapes into ``Delta`` -- keeping this out of it is what makes swapping the
model layer a one-file change (DESIGN2 §6.1).

This module implements the ``durable_partials.md`` proposal.  The in-flight
phase -- between a producer's first delta for an id and the moment that id's
stream ends -- is a distinct type, :class:`PartialEvent`, which is genuinely
cheap to ignore and genuinely cheap to subscribe to.  Exactly one
:class:`Event` is ever produced for an id, at ``reify()`` time.

Three properties, one mechanism (durable_partials.md §"Three properties"):

1. **Simple, append-only accumulation, resilient to a crash.**  Whatever is
   received from the provider gets recorded durably as it arrives, not
   reconstructed after the fact.  An optional :class:`DurabilitySink` is
   constructor-injected here; ``None``/no-op for tests and for theatres that
   accept losing an in-flight turn.
2. **Consumers that want live partial updates get them fast.**  ``subscribe()``
   yields every increment as it arrives -- the actual delta, with no cumulative
   Event ever constructed in between and no diffing anywhere.
3. **Consumers that don't want partials pay nothing for their existence.**
   ``render()`` and the theatre's ``integrate`` only ever want the one, final
   Event for an id.  They do one cheap ``isinstance`` check and ignore the
   PartialEvent handle.

``accumulate`` yields a :class:`PartialEvent` **once per id**, the moment a
new slot is first observed -- not once per delta.  On ``Finish``, every
still-open slot is reified into a ``complete=True`` :class:`Event`.

Interruption needs no special handling in the generator.  If the caller stops
iterating before ``Finish`` arrives, each :class:`PartialEvent`'s
:func:`~PartialEvent.snapshot` is the honest, joined-so-far content.  The
theatre calls ``reify(complete=False)`` on each remaining handle to produce
interrupted Events -- at least as precise as, and simpler than, "whichever
Event object happened to be the last one yielded" (durable_partials.md: "What
the theatre's own bookkeeping simplifies to").
"""

import asyncio
import json
import os
from pathlib import Path
from time import time
from typing import AsyncGenerator, AsyncIterator, Protocol, Sequence

from patchbay_llm.events import (
    ContentBlock,
    Event,
    EventId,
    Media,
    Thought,
    ToolCall,
    new_id,
)
from patchbay_llm.infer_engine.delta import (
    CallDelta,
    Delta,
    Finish,
    TextDelta,
    ThoughtDelta,
)

__all__ = [
    "accumulate",
    "PartialEvent",
    "reify_block",
    "DurabilitySink",
    "FileDeltaLog",
    "recover_events",
]


# ── The free-standing join function ─────────────────────────────────────────


def reify_block(kind: str, deltas: Sequence[Delta]) -> ContentBlock:
    """Join ordered deltas for one slot into a single content block.

    Pure function of ``(kind, ordered deltas) -> ContentBlock``, callable from
    both a live :class:`PartialEvent` and a recovery routine running in a
    different process.  This is the one piece of code the whole proposal
    insists must exist exactly once, in exactly one place -- the same way
    ``accumulate``'s tool-call-argument accumulation already insists on it
    today (durable_partials.md: "Recovery" §2).
    """
    if kind == "message":
        text = "".join(d.text for d in deltas if isinstance(d, TextDelta))
        return Media("text/plain", text.encode())

    if kind == "thought":
        text = "".join(d.text for d in deltas if isinstance(d, ThoughtDelta))
        signature: str | None = None
        for d in deltas:
            if isinstance(d, ThoughtDelta) and d.signature is not None:
                signature = d.signature
        return Thought(text, signature)

    if kind == "tool_call":
        call_id = ""
        tool = ""
        args = ""
        for d in deltas:
            if isinstance(d, CallDelta):
                if d.id is not None:
                    call_id = d.id
                if d.tool is not None:
                    tool = d.tool
                if d.args:
                    args += d.args
        return ToolCall(call_id, tool, args)

    raise ValueError(f"unknown slot kind: {kind!r}")


# ── Durability sink ─────────────────────────────────────────────────────────


class DurabilitySink(Protocol):
    """Durably records deltas for in-flight events.

    Constructor-injected into :func:`accumulate` (DESIGN2 principle 2:
    "dependencies are wired at construction, by you").  ``None``/no-op for
    tests and for theatres that accept losing an in-flight turn (the REPL
    already does, on Ctrl-C).

    A sink is the thing :meth:`PartialEvent.append` calls synchronously
    *before* fanning out to live subscribers (durable_partials.md:
    "Durability").  On :meth:`PartialEvent.reify`, the sink's
    :meth:`finalize` is called so the log for that id can be deleted or
    truncated -- compression happens once, at finalization.
    """

    def append(
        self, eid: EventId, kind: str, author: str, delta: Delta, seq: int
    ) -> None:
        """Durably record one delta for the given event id."""

    def finalize(self, eid: EventId) -> None:
        """The event has been reified; the log for this id may be deleted."""


class _NoOpSink:
    """The default: no durability.  Used when ``sink=None``."""

    def append(
        self, eid: EventId, kind: str, author: str, delta: Delta, seq: int
    ) -> None:
        """No-op: discards all arguments."""
        del eid, kind, author, delta, seq  # nothing to do

    def finalize(self, eid: EventId) -> None:
        """No-op: nothing to finalize."""
        del eid


# ── Delta serialisation (for the file-based sink) ──────────────────────────


def _delta_to_dict(delta: Delta) -> dict[str, object]:
    if isinstance(delta, TextDelta):
        return {"type": "text", "slot": delta.slot, "text": delta.text}
    if isinstance(delta, ThoughtDelta):
        return {
            "type": "thought",
            "slot": delta.slot,
            "text": delta.text,
            "signature": delta.signature,
        }
    if isinstance(delta, CallDelta):
        return {
            "type": "call",
            "slot": delta.slot,
            "id": delta.id,
            "tool": delta.tool,
            "args": delta.args,
        }
    raise TypeError(f"cannot serialise {type(delta).__name__} to a delta log")


def _dict_to_delta(d: dict[str, object]) -> Delta:
    t = d["type"]
    if t == "text":
        return TextDelta(d["slot"], d["text"])  # type: ignore[arg-type]
    if t == "thought":
        return ThoughtDelta(
            d["slot"],  # type: ignore[arg-type]
            d.get("text", ""),  # type: ignore[arg-type]
            d.get("signature"),  # type: ignore[arg-type]
        )
    if t == "call":
        return CallDelta(
            d["slot"],  # type: ignore[arg-type]
            d.get("id"),  # type: ignore[arg-type]
            d.get("tool"),  # type: ignore[arg-type]
            d.get("args", ""),  # type: ignore[arg-type]
        )
    raise ValueError(f"unknown delta type in log: {t!r}")


class FileDeltaLog:
    """Append-only JSONL durability sink, one file per event id.

    The internal buffer of :class:`PartialEvent` *is* the in-memory log; this
    is the durable backing store (durable_partials.md: "Durability").  Each
    append is O(1) regardless of output length.  On ``reify()``, the log for
    that id is deleted -- compression happens once, at finalization.

    Layout::

        <durable-scratch>/<event-id>.deltas.jsonl
        {"seq": 0, "kind": "message", "author": "llm:main", "delta": {...}}
        {"seq": 1, "kind": "message", "author": "llm:main", "delta": {...}}

    ``fsync`` is called on every append, so the guarantee is "survives
    ``kill -9``".  Batching trades durability window for throughput; no opinion
    is offered here (durable_partials.md: "Open questions").
    """

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, eid: EventId) -> Path:
        """Return the JSONL log path for the given event id."""
        return self._dir / f"{eid}.deltas.jsonl"

    def append(
        self, eid: EventId, kind: str, author: str, delta: Delta, seq: int
    ) -> None:
        """Durably append one delta to the JSONL log for the given event id."""
        entry = {
            "seq": seq,
            "kind": kind,
            "author": author,
            "delta": _delta_to_dict(delta),
        }
        line = json.dumps(entry, default=str)
        with self._path(eid).open("a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def finalize(self, eid: EventId) -> None:
        """Delete the log for the given event id (reification complete)."""
        try:
            self._path(eid).unlink()
        except FileNotFoundError:
            pass


def recover_events(directory: Path) -> list[Event]:
    """Reconstruct interrupted Events from dangling delta logs.

    On startup, before a theatre resumes, scan for delta logs with no
    corresponding finalized :class:`Event`.  For each one found
    (durable_partials.md: "Recovery"):

    1. Reconstruct a PartialEvent-shaped buffer purely from the log (no live
       process, no subscribers -- just the durable lines).
    2. Call the **same free-standing join function** :func:`reify_block` that
       a live ``PartialEvent.reify()`` calls.
    3. The result is a ``complete=False`` :class:`Event`, identical in shape
       to what a live cancellation would have produced.  Everything
       downstream already exists: the theatre's own cancellation handling,
       and DESIGN2 §6.7's Retry-vs-Continuation policy.
    4. A recovered ``tool_call``'s :class:`Event` is still ``complete=False``
       -- the same type-level guarantee that partial arguments are never
       executable holds after a crash exactly as after a live cancel.
    5. Delete the log once folded into an :class:`Event`, so a second restart
       doesn't recover it twice.

    None of this requires the theatre to know a crash happened.  It sees an
    interrupted turn, same as always.
    """
    results: list[Event] = []
    directory = Path(directory)
    if not directory.is_dir():
        return results
    for log_path in sorted(directory.glob("*.deltas.jsonl")):
        eid = EventId(log_path.name.removesuffix(".deltas.jsonl"))
        deltas: list[Delta] = []
        kind = ""
        author = ""
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            kind = entry["kind"]
            author = entry.get("author", "")
            deltas.append(_dict_to_delta(entry["delta"]))
        if not deltas:
            continue
        block = reify_block(kind, deltas)
        results.append(
            Event(
                id=eid,
                ts=time(),
                kind=kind,
                content=(block,),
                meta={"author": author},
                complete=False,
            )
        )
        log_path.unlink()
    return results


# ── PartialEvent ───────────────────────────────────────────────────────────


_END = object()  # sentinel: the subscription is over


class PartialEvent:
    """A live, append-only, not-yet-reified record of one in-flight event.

    Exists only between a producer's first delta for one id and the moment
    that id's stream ends -- normally, by live cancellation, or (via
    recovery) by a durable log surviving a crash.  Never integrated into
    state, never journalled as itself, never merged with another
    PartialEvent -- there is exactly one of these per id, and it produces
    exactly one :class:`Event` (durable_partials.md).

    :meth:`subscribe` is fan-out (an Rx-style hot Subject): multiple
    independent subscribers each get every delta.  A durability sink and a
    live wire-forwarder (ACP, a REPL printer) can each subscribe and get
    every delta, independently.
    """

    __slots__ = (
        "id",
        "kind",
        "author",
        "_deltas",
        "_subscribers",
        "_sink",
        "_ended",
    )

    def __init__(
        self,
        eid: EventId,
        kind: str,
        author: str,
        sink: DurabilitySink | None = None,
    ) -> None:
        self.id = eid
        self.kind = kind
        self.author = author
        self._deltas: list[Delta] = []
        self._subscribers: list[asyncio.Queue[object]] = []
        self._sink: DurabilitySink = sink or _NoOpSink()
        self._ended = False

    def append(self, delta: Delta) -> None:
        """Record one delta, persist it, fan out to live subscribers.

        The durability sink is called synchronously *before* fanning out to
        live subscribers, so a crash between the two leaves a durable record
        of the delta but no live consumer in an inconsistent state
        (durable_partials.md: "Durability").
        """
        if self._ended:
            raise RuntimeError(f"PartialEvent {self.id} is already reified")
        seq = len(self._deltas)
        self._deltas.append(delta)
        self._sink.append(self.id, self.kind, self.author, delta, seq)
        for q in self._subscribers:
            q.put_nowait(delta)

    def snapshot(self) -> ContentBlock:
        """The joined-so-far content, on demand, no subscription required.

        A cheap, independent read -- a debugging tool, or some future
        reconnecting client, might want "what's happened so far" without
        having subscribed from the start (durable_partials.md: "Multi-reader").
        """
        return reify_block(self.kind, self._deltas)

    async def subscribe(self) -> AsyncGenerator[Delta, None]:
        """Every increment for this id, from creation, live, as it arrives.

        Multiple independent subscribers are supported -- this is fan-out (an
        Rx-style hot Subject), not a competing-consumers queue.  A durability
        sink and a live wire-forwarder (ACP, a REPL printer) can each
        subscribe and get every delta, independently.

        Every real subscriber today attaches at creation time (before any
        deltas exist), so catch-up is just replaying the buffer.  The stream
        ends when :meth:`reify` is called.  A late-joining subscriber that
        attaches after reify still gets the full buffer via replay, then the
        subscription ends immediately.
        """
        if self._ended:
            for d in self._deltas:
                yield d
            return

        q: asyncio.Queue[object] = asyncio.Queue()
        for d in self._deltas:
            q.put_nowait(d)
        self._subscribers.append(q)
        try:
            while True:
                item = await q.get()
                if item is _END:
                    return
                yield item  # type: ignore[misc]
        finally:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def reify(self, *, complete: bool) -> Event:
        """Join everything appended so far into the one :class:`Event`.

        Called exactly once, by whoever ends this PartialEvent's life:
        :func:`accumulate` itself on Finish, the theatre on a live cancel, or
        a recovery routine replaying a durable log after a crash.  Idempotent
        to call twice is not a requirement worth building for -- it is called
        by exactly one owner, once.
        """
        if self._ended:
            raise RuntimeError(f"PartialEvent {self.id} is already reified")
        self._ended = True
        for q in self._subscribers:
            q.put_nowait(_END)
        self._sink.finalize(self.id)
        block = reify_block(self.kind, self._deltas)
        return Event(
            id=self.id,
            ts=time(),
            kind=self.kind,
            content=(block,),
            meta={"author": self.author},
            complete=complete,
        )


# ── accumulate ─────────────────────────────────────────────────────────────


def _slot_kind(delta: Delta) -> str:
    if isinstance(delta, TextDelta):
        return "message"
    if isinstance(delta, ThoughtDelta):
        return "thought"
    return "tool_call"


def _usage_event(finish: Finish, author: str) -> Event:
    u = finish.usage
    return Event(
        id=new_id(),
        ts=time(),
        kind="usage",
        meta={
            "author": author,
            "model": u.model,
            "input": u.input,
            "cache_read": u.cache_read,
            "cache_write": u.cache_write,
            "output": u.output,
            "thought": u.thought,
            "billed": u.billed,
            "estimated": u.estimated,
            "stop": finish.stop.name,
            "detail": finish.detail,
        },
        complete=True,
    )


async def accumulate(
    stream: AsyncIterator[Delta],
    author: str,
    sink: DurabilitySink | None = None,
) -> AsyncGenerator[Event | PartialEvent, None]:
    """Turn one participant's inference-delta stream into journal Events.

    ``author`` becomes ``meta.author`` on every emitted event (DESIGN2's
    journal vocabulary: message/thought/tool_call are authored).  Passed in
    rather than read from anywhere ambient, so this function is testable with
    nothing but a synthetic ``Delta`` sequence -- no participant, no engine,
    no journal.

    ``sink`` is an optional :class:`DurabilitySink` -- constructor-injected
    per DESIGN2 principle 2, ``None``/no-op for tests and for theatres that
    accept losing an in-flight turn (durable_partials.md: "Durability").

    A :class:`PartialEvent` is yielded **once per id**, the moment a new slot
    is first observed -- not once per delta.  A consumer that only wants
    finals does exactly what it does today, plus one cheap ``isinstance``
    check::

        async for item in accumulate(stream, author):
            if isinstance(item, Event):
                state = integrate(state, item)
            # else: a PartialEvent floated by.  Ignoring it costs one check.

    A consumer that wants live text subscribes the moment it sees the handle::

        if isinstance(item, PartialEvent):
            asyncio.create_task(forward_deltas(item))

    On ``Finish``, every still-open slot is reified into a ``complete=True``
    :class:`Event`.  If the caller stops iterating before ``Finish``, each
    :class:`PartialEvent`'s :meth:`~PartialEvent.snapshot` is the honest,
    joined-so-far content -- the theatre calls
    :meth:`~PartialEvent.reify` with ``complete=False`` on each to produce
    interrupted Events.
    """
    slots: dict[int, PartialEvent] = {}

    async for delta in stream:
        if isinstance(delta, Finish):
            for pe in slots.values():
                yield pe.reify(complete=True)
            yield _usage_event(delta, author)
            return

        pe = slots.get(delta.slot)
        if pe is None:
            kind = _slot_kind(delta)
            pe = PartialEvent(new_id(), kind, author, sink)
            slots[delta.slot] = pe
            pe.append(delta)
            yield pe  # yield the handle once, after the first delta is in
        else:
            pe.append(delta)

    # The stream exhausted with no Finish -- infer_engine's own contract (§5)
    # forbids this (it must raise instead), so reaching here is a bug one
    # layer down, not something to paper over here.
    raise RuntimeError("delta stream ended without a Finish")
