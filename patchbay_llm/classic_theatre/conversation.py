"""The conversation: the minimal theatre's state shape and its folds.

``Conversation`` is the one ``State`` shape that ships in v1 (DESIGN2: "``State``
is theatre-defined... ``Conversation`` is the only shape that ships").
``append_only`` is the one-line integration rule the append-only convention
earns its name from; ``turn_at`` is the two-party alternation fold the REPL
theatre (DESIGN2 §5.2) needs.

These are theatre-level concerns, not substrate: a different theatre
(preemptible, concurrent, search) would define its own ``turn_at`` and its own
``integrate`` over the same ``Event`` substrate.  They live here rather than
in ``repl_theatre.py`` because ``HumanParticipant`` and ``LlmParticipant`` both
need ``turn_end``/``message`` constructors, and importing them from the theatre
would make participants depend on a specific theatre -- the opposite of the
layering DESIGN2 §4 defends (``LlmParticipant`` never imports the theatre).
"""

from time import time

from patchbay_llm.events import Event, Media, new_id

__all__ = [
    "Conversation",
    "append_only",
    "turn_at",
    "message",
    "turn_end",
]

Conversation = tuple[Event, ...]

_CONTENT_KINDS = ("message", "thought", "tool_call", "tool_result")


def append_only(state: Conversation, event: Event) -> Conversation:
    """The one-line default integration rule (DESIGN2 §3)."""
    return state + (event,)


def turn_at(state: Conversation, participants: tuple[str, str]) -> str | None:
    """Two-party strict alternation (DESIGN2 §5.2).

    The next participant is whoever did *not* author the most recent content
    event or ``turn_end``.  An empty state is the first participant's turn.
    ``turn_end`` is the explicit handoff signal; absent one, the last author of
    a content event is the speaker whose turn just ended -- both collapse to
    the same answer under strict alternation.
    """
    if not state:
        return participants[0]
    last_author: str | None = None
    for e in reversed(state):
        if e.kind == "turn_end" or e.kind in _CONTENT_KINDS:
            last_author = e.meta.get("author")
            break
    if last_author is None:
        return participants[0]
    if last_author == participants[0]:
        return participants[1]
    return participants[0]


def message(author: str, text: str) -> Event:
    """A complete ``message`` event authored by ``author``."""
    return Event(
        id=new_id(),
        ts=time(),
        kind="message",
        content=(Media("text/plain", text.encode("utf-8")),),
        meta={"author": author},
        complete=True,
    )


def turn_end(participant: str, reason: str = "relinquished") -> Event:
    """A ``turn_end`` event: ``participant`` relinquished the floor for ``reason``.

    ``meta.reason`` is one of ``relinquished``, ``cancelled``, ``cap``, ``error``
    (DESIGN2 §3 "The journal vocabulary").
    """
    return Event(
        id=new_id(),
        ts=time(),
        kind="turn_end",
        meta={"author": participant, "reason": reason},
        complete=True,
    )
