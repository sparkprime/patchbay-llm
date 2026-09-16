"""Contract tests for the conversation state shape and its folds (DESIGN2 §3, §5.2)."""

from time import time

from patchbay_llm.classic_theatre.conversation import (
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
    e1 = _ev("message", "human:you")
    e2 = _ev("message", "llm:main")
    assert append_only(state, e1) == (e1,)
    assert append_only(append_only(state, e1), e2) == (e1, e2)


def test_turn_at_empty_state_is_the_first_participant() -> None:
    """An empty conversation is the first participant's turn."""
    assert turn_at((), ("human:you", "llm:main")) == "human:you"


def test_turn_at_alternates_on_last_author() -> None:
    """Absent a turn_end, the last content author's counterpartie is next."""
    state = (
        _ev("message", "human:you"),
        _ev("message", "llm:main"),
    )
    assert turn_at(state, ("human:you", "llm:main")) == "human:you"


def test_turn_at_picks_the_other_after_a_single_message() -> None:
    """One message from the human means the LLM is next."""
    state = (_ev("message", "human:you"),)
    assert turn_at(state, ("human:you", "llm:main")) == "llm:main"


def test_turn_end_hands_off_to_the_other_participant() -> None:
    """A turn_end by the LLM means the human is next (explicit handoff)."""
    state = (
        _ev("message", "human:you"),
        _ev("message", "llm:main"),
        turn_end("llm:main", "relinquished"),
    )
    assert turn_at(state, ("human:you", "llm:main")) == "human:you"


def test_turn_end_by_human_hands_off_to_the_llm() -> None:
    """A turn_end by the human hands the floor to the LLM."""
    state = (
        _ev("message", "human:you"),
        turn_end("human:you", "relinquished"),
    )
    assert turn_at(state, ("human:you", "llm:main")) == "llm:main"


def test_turn_at_skips_usage_and_config_events() -> None:
    """usage and config_change are authored but not content; turn_at ignores
    them and keys off the last content/turn_end author."""
    state = (
        _ev("message", "human:you"),
        _ev("message", "llm:main"),
        _ev("usage", "llm:main"),
        _ev("config_change", "human:you"),
        turn_end("llm:main", "relinquished"),
    )
    assert turn_at(state, ("human:you", "llm:main")) == "human:you"


def test_message_event_is_complete_and_authored() -> None:
    """``message`` produces a complete, authored event carrying the text."""
    e = message("human:you", "hello")
    assert e.kind == "message"
    assert e.complete is True
    assert e.meta["author"] == "human:you"
    block = e.content[0]
    assert isinstance(block, Media)
    assert block.data == b"hello"


def test_turn_end_event_carries_reason() -> None:
    """``turn_end`` records who relinquished and why."""
    e = turn_end("llm:main", "relinquished")
    assert e.kind == "turn_end"
    assert e.meta["author"] == "llm:main"
    assert e.meta["reason"] == "relinquished"
    assert e.complete is True
