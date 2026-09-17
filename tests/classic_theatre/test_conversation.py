"""Contract tests for the conversation state shape and its folds (DESIGN2 §3, §5.2)."""

from time import time

from patchbay_llm.classic_theatre.conversation import (
    ASSISTANT,
    HUMAN,
    Conversation,
    append_only,
    message,
    turn_at,
    turn_end,
)
from patchbay_llm.events import Event, Media, new_id


def _ev(
    kind: str,
    author: str,
    *,
    complete: bool = True,
    content: tuple[Media, ...] = (),
) -> Event:
    return Event(
        id=new_id(),
        ts=time(),
        kind=kind,
        content=content,
        meta={"author": author},
        complete=complete,
    )


def test_append_only_extends_the_tuple() -> None:
    """The one-line default integration rule (DESIGN2 §3)."""
    state: Conversation = ()
    e1 = _ev("message", HUMAN)
    e2 = _ev("message", ASSISTANT)
    assert append_only(state, e1) == (e1,)
    assert append_only(append_only(state, e1), e2) == (e1, e2)


def test_turn_at_empty_state_is_the_human() -> None:
    """An empty conversation is the human's turn."""
    assert turn_at(()) == HUMAN


def test_turn_at_alternates_on_last_author() -> None:
    """Absent a turn_end, the last content author's counterpart is next."""
    state = (
        _ev("message", HUMAN),
        _ev("message", ASSISTANT),
    )
    assert turn_at(state) == HUMAN


def test_turn_at_picks_assistant_after_a_single_human_message() -> None:
    """One message from the human means the assistant is next."""
    state = (_ev("message", HUMAN),)
    assert turn_at(state) == ASSISTANT


def test_turn_end_hands_off_to_the_other_participant() -> None:
    """A turn_end by the assistant means the human is next (explicit handoff)."""
    state = (
        _ev("message", HUMAN),
        _ev("message", ASSISTANT),
        turn_end(ASSISTANT, "relinquished"),
    )
    assert turn_at(state) == HUMAN


def test_turn_end_by_human_hands_off_to_the_assistant() -> None:
    """A turn_end by the human hands the floor to the assistant."""
    state = (
        _ev("message", HUMAN),
        turn_end(HUMAN, "relinquished"),
    )
    assert turn_at(state) == ASSISTANT


def test_turn_at_skips_usage_and_config_events() -> None:
    """usage and config_change are authored but not content; turn_at ignores
    them and keys off the last content/turn_end author."""
    state = (
        _ev("message", HUMAN),
        _ev("message", ASSISTANT),
        _ev("usage", ASSISTANT),
        _ev("config_change", HUMAN),
        turn_end(ASSISTANT, "relinquished"),
    )
    assert turn_at(state) == HUMAN


def test_message_event_is_complete_and_authored() -> None:
    """``message`` produces a complete, authored event carrying the text."""
    e = message(HUMAN, "hello")
    assert e.kind == "message"
    assert e.complete is True
    assert e.meta["author"] == HUMAN
    block = e.content[0]
    assert isinstance(block, Media)
    assert block.data == b"hello"


def test_turn_end_event_carries_reason() -> None:
    """``turn_end`` records who relinquished and why."""
    e = turn_end(ASSISTANT, "relinquished")
    assert e.kind == "turn_end"
    assert e.meta["author"] == ASSISTANT
    assert e.meta["reason"] == "relinquished"
    assert e.complete is True
