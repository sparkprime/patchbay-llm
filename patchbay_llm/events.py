"""The event.

Event is the one container everything else in the project is built from.
Partial or complete, merged by id.  See DESIGN2 §3 ("One container",
"Partial events, merged by id").

This module is the substrate: it imports nothing from ``infer_engine``.
``Media`` is defined *here*, and ``infer_engine.prompt`` imports it upward,
which is the correct direction for the one piece of element-vocabulary data
that crosses the layering (DESIGN2: "Two things deliberately cross the
layering... the element vocabulary of the journal").

``Thought``, ``ToolCall`` and ``ToolResult`` are journal-owned types, not
reused from ``infer_engine``.  ``Thought`` and ``ToolResult`` are
shape-identical to their inference counterparts today; ``ToolCall`` diverges
load-bearingly: ``args`` is raw JSON text (possibly a fragment while the
event is partial) rather than a parsed mapping, which is what makes "partial
arguments are never executed" true by construction.  The explicit, boring
conversions live in ``participants.llm.convert`` -- a layer above both this
module and ``infer_engine``, which import into it, not the other way round.
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
    "merge",
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

    ``args`` is raw JSON text, possibly a fragment while the event is partial.
    It is **never** parsed here -- parsing happens in ``participants.llm.convert``
    and only once the event is complete.  That is the type-level guarantee that
    partial arguments are never executed (DESIGN2 §3 "Partial events").
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
    """The one container.  Partial or complete, merged by id.

    ``content`` is cumulative, not incremental: each partial is a complete,
    valid replacement for the one before it, not a delta to be applied.
    ``complete=False`` means the event was interrupted; nothing sets it,
    nothing forgets it.

    ``kind`` is an open namespace (DESIGN2 §3): consumers must ignore kinds
    they do not recognise.  Known kinds: ``message``, ``thought``,
    ``tool_call``, ``tool_result``, ``usage``, ``config_change``,
    ``turn_start``, ``turn_end``, ``notice``.
    """

    id: EventId
    ts: float
    kind: str
    content: tuple[ContentBlock, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict[str, Any])
    complete: bool = True


def merge(previous: Event, update: Event) -> Event:
    """Combine two partials sharing an id.

    Content is cumulative, so this is last-wins by construction -- ``update``
    already carries everything ``previous`` did, plus more.  What earns this
    a real function rather than a bare dict assignment is the invariant it
    checks: a participant must not keep yielding for an id it already marked
    complete, and two events sharing an id must share a kind.
    """
    if previous.id != update.id:
        raise ValueError(f"cannot merge Event {update.id!r} into {previous.id!r}")
    if previous.kind != update.kind:
        raise ValueError(f"Event {previous.id!r} changed kind mid-stream")
    if previous.complete:
        raise ValueError(f"Event {previous.id!r} is already complete")
    return update
