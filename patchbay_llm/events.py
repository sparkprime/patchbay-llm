"""The event.

Event is the one container everything else in the project is built from.
Exactly one Event is ever produced for an id, at reify time, and
``complete`` is decided once as a fact about how that single object came to
exist (Finish reached, vs. cut short by cancel or crash-recovery).  See
DESIGN2 §3 ("One container") and ``durable_partials.md`` ("What
``Event.complete`` means now").

Event is the *ground-truth* type: a first-hand, durable record, produced
exactly once per id.  It is not the in-flight type.  The in-flight phase --
between a producer's first delta for an id and the moment that id's stream
ends -- is represented by the separate :class:`~patchbay_llm.live.LiveUpdate`
vocabulary (see ``patchbay_llm.live``), which exists for one purpose:
fine-grained rendering updates to a UI, decoupled from this module and from
``infer_engine``.  A :class:`LiveUpdate` correlates with its eventual
:class:`Event` by id value only, not by shape.

This module is the substrate: it imports nothing from ``infer_engine``.
``Media`` is defined *here*, and ``infer_engine.prompt`` imports it upward,
which is the correct direction for the one piece of element-vocabulary data
that crosses the layering (DESIGN2: "Two things deliberately cross the
layering... the element vocabulary of the journal").

``Thought``, ``ToolCall`` and ``ToolResult`` are journal-owned types, not
reused from ``infer_engine``.  ``Thought`` and ``ToolResult`` are
shape-identical to their inference counterparts today; ``ToolCall`` diverges
load-bearingly: ``args`` is raw JSON text (possibly a fragment while the
event was interrupted) rather than a parsed mapping, which is what makes
"partial arguments are never executed" true by construction.  The explicit,
boring conversions live in ``infer_engine.convert`` -- which imports from both
this module and ``infer_engine.prompt``, not the other way round.

The bridge between ``infer_engine.delta.Delta`` (the provider stream) and
``Event`` (the durable record) is ``accumulate`` (in
``classic_theatre.repl_theatre``): it assigns ids, converts each ``Delta`` to a
:class:`LiveUpdate` for any live consumer, and joins the per-slot increments
into the one final :class:`Event` per slot on ``Finish``.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, NewType, Union

__all__ = [
    "EventId",
    "new_id",
    "Media",
    "Thought",
    "ToolCall",
    "ToolResult",
    "ContentBlock",
    "Event",
]

EventId = NewType("EventId", str)


def new_id() -> EventId:
    """A fresh, opaque event id.

    UUIDv4 -- deliberately not v7, because a time-sortable id would re-imply
    the order that the journal refuses to bake into identity (DESIGN2 §3).
    """
    return EventId(str(uuid.uuid4()))


@dataclass(frozen=True)
class Media:
    """A content part: text, image, audio or PDF, distinguished only by mime.

    ``data`` is always explicit bytes.  This type is shared with
    ``infer_engine.prompt`` (which imports it from here) because it is pure
    data with no journal-specific behaviour -- the one sanctioned crossing
    of the layering boundary.
    """

    mime: str
    data: bytes


@dataclass(frozen=True)
class Thought:
    """Model reasoning, previously emitted.

    Journal-owned rather than reused from ``infer_engine``: the journal
    tracks its own vocabulary even where it currently mirrors the inference
    one, so a future divergence (a redacted-signature field, a provider-specific
    format) does not ripple through the record.
    """

    text: str
    signature: str | None = None


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation a participant requested.

    ``args`` is raw JSON text, possibly a fragment if the event was
    interrupted.  It is **never** parsed here -- parsing happens in
    ``infer_engine.convert`` and only once the event is complete.  That
    is the type-level guarantee that partial arguments are never executed
    (DESIGN2 §3 "Partial events", durable_partials.md "Recovery" §4).
    """

    id: str
    tool: str
    args: str = ""


@dataclass(frozen=True)
class ToolResult:
    """The outcome of a :class:`ToolCall`.  ``call`` is the ``ToolCall.id`` it answers."""

    call: str
    content: tuple[Media, ...]
    failed: bool = False


ContentBlock = Union[Media, Thought, ToolCall, ToolResult]


@dataclass(frozen=True)
class Event:
    """A piece of information that was raised during a conversation.

    LLMs are trained to respond in the context of a conversation expressed as a
    sequence of events. We don't have to take this representation literally. But
    we do have to express what we want within those concepts.

    Globally defined fields: Events have globally distinct ids. They are
    immutable. They have timestamps. They have content in a standardised format.
    The `complete` field is false when the event was interrupted and only
    partially stored.

    Theatre-defined fields: They have a kind field which is an open namespace.
    They also have metadata for additional open data. The precise way the
    conversation is structured and the meaning of "kind" and "meta" are defined
    by the theatre. As the theatre creates the events, it can impose its own
    expectations on these fields.

    Example kinds: ``message``, ``thought``, ``tool_call``, ``tool_result``,
    ``usage``, ``config_change``, ``turn_start``, ``turn_end``, ``notice``. (But
    it depends on the theatre).
    """

    id: EventId
    ts: float
    kind: str
    content: tuple[ContentBlock, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict[str, Any])
    complete: bool = True
