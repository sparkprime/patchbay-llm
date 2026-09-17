"""The live-update vocabulary.

A fourth vocabulary, beside the three DESIGN2 §3 already names (inference,
journal, ACP).  It exists for one purpose: **communicating fine-grained
rendering updates to a UI** -- a REPL printer, an ACP live-forwarder, anything
that wants to draw what is happening as it happens, finer-grained than the one
final :class:`~patchbay_llm.events.Event` per id the journal will eventually
hold.

It is deliberately *not* aligned with :class:`Event`'s shape.  ``Event`` answers
"what is durably, ground-truthfully true"; a :class:`LiveUpdate` answers "what
changed, right now, that something should draw".  Conflating those (by making a
partial an ``Event(complete=False)`` or by giving the live type ``Event``'s
``(id, kind, content, meta)`` envelope) is what made every prior attempt feel
wrong: an ``Event`` is not the right shape for a per-chunk render increment, and
a render increment is not the right shape for a ground-truth record.

The only thing this vocabulary shares with ``Event`` is the *id value* -- an
:class:`~patchbay_llm.events.EventId`, so a live ``ToolCallChunk`` stream can be
correlated with the eventual ``tool_call`` :class:`Event` for the same slot.
That is correlation by value, not coupling by type: nothing here imports
:class:`Event`, :class:`~patchbay_llm.events.ContentBlock`, or any journal-owned
content type, and nothing in ``infer_engine`` or any theatre imports this
module.  ``accumulate`` is the one place that legitimately knows both
``infer_engine.delta.Delta`` and the journal's vocabulary, and it is therefore
the one producer of :class:`LiveUpdate`.

Two consumers are anticipated, both of which pattern-match on the same
variants and know nothing of each other:

- a REPL/CLI printer (``repl_theatre.run_repl``) -- extracts ``text``
  from :class:`MessageChunk` / :class:`ThoughtChunk` and ignores the rest;
- an ACP live-forwarder (not yet built) -- maps the same variants onto ACP's
  ``agent_message_chunk`` / ``agent_thought_chunk`` / ``tool_call_update``
  wire types, the same way ``project`` maps a final :class:`Event` onto ACP's
  per-event updates.

The union is intentionally minimal today: message, thought, tool-call.  When a
real producer streams a tool result or media (ACP can), a ``ToolResultChunk``
or ``MediaChunk`` is one new variant here -- not a field added to a shared
envelope everyone has to widen.
"""

from dataclasses import dataclass
from typing import Union

from patchbay_llm.events import EventId

__all__ = [
    "MessageChunk",
    "ThoughtChunk",
    "ToolCallChunk",
    "ToolResultChunk",
    "LiveUpdate",
]


@dataclass(frozen=True)
class MessageChunk:
    """One increment of generated message text for slot ``id``.

    ``text`` is the raw increment for this chunk, not the cumulative text so
    far.  A consumer that wants cumulative text subscribes to every chunk for
    ``id`` and concatenates; a consumer that only wants to append (a terminal
    printer) takes ``text`` verbatim.
    """

    id: EventId
    text: str


@dataclass(frozen=True)
class ThoughtChunk:
    """One increment of reasoning text for slot ``id``.

    ``signature`` arrives on the last chunk of the slot (or ``None`` until
    then) and is the opaque provider token that must be replayed verbatim in a
    later prompt that continues this thought.
    """

    id: EventId
    text: str
    signature: str | None = None


@dataclass(frozen=True)
class ToolCallChunk:
    """One increment of a tool-call's arguments for slot ``id``.

    Two distinct ids are in play and both must survive into the final
    :class:`~patchbay_llm.events.ToolCall` block:

    - ``id`` is the *slot* id -- the one :class:`EventId` accumulate assigned
      for this in-flight event, shared with every other chunk for this slot
      and with the eventual ``tool_call`` :class:`Event`.  It is what lets a
      consumer correlate a live stream with its final Event.
    - ``call_id`` is the *provider's own* tool-use id (Anthropic's
      ``tool_use.id``, OpenAI's ``tool_call_id``).  It is content-level data,
      not a slot identity: it lands as ``ToolCall.id`` inside the content
      block, where ``render`` and ACP's ``toolCallId`` read it.

    ``tool`` and ``call_id`` arrive on the first chunk of the slot only.
    ``args`` is a JSON *fragment*, not valid JSON alone -- it must not be
    parsed mid-stream, and ``accumulate`` never parses it.  Joining the
    fragments and parsing is a post-hoc step over a *complete* slot, which is
    exactly the safety property that makes a partial ``tool_call`` never
    executable by construction.
    """

    id: EventId
    call_id: str | None = None
    tool: str | None = None
    args: str = ""


@dataclass(frozen=True)
class ToolResultChunk:
    """One increment of a tool's output for the ``tool_result`` event ``id``.

    A tool that takes time (a bash command, an HTTP request) should show its
    output to the UI as it arrives, not freeze until the final
    :class:`~patchbay_llm.events.ToolResult` event exists.  This variant
    carries those increments.

    Like :class:`ToolCallChunk`, two distinct ids are in play:

    - ``id`` is the *slot* id -- the :class:`EventId` the tool executor
      assigned for this ``tool_result`` event, shared with the eventual
      :class:`Event`.
    - ``call_id`` is the *provider's* tool-call id -- the same value that
      appeared on the :class:`ToolCallChunk` that initiated this call.  It
      lets a consumer correlate the result stream with the call stream and
      lands as ``ToolResult.call`` in the final content block.

    ``text`` is the raw increment for this chunk (stdout/stderr fragment),
    not the cumulative output.  ``failed`` is ``None`` until the tool
    finishes; the last chunk carries ``True`` (non-zero exit, error) or
    ``False`` (success), like :class:`ThoughtChunk.signature`.
    """

    id: EventId
    call_id: str = ""
    text: str = ""
    failed: bool | None = None


LiveUpdate = Union[MessageChunk, ThoughtChunk, ToolCallChunk, ToolResultChunk]
