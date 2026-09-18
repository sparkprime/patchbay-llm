"""Contract tests for ``repl_theatre.render`` (the state -> Prompt fold)."""

from time import time

from patchbay_llm.classic_theatre.conversation import ASSISTANT, HUMAN, message
from patchbay_llm.classic_theatre.seam import to_prompt
from patchbay_llm.events import Event, Media, Thought, ToolCall, new_id
from patchbay_llm.infer_engine.prompt import Call, Prompt


def _thought(author: str, text: str, signature: str | None = None) -> Event:
    return Event(
        id=new_id(),
        ts=time(),
        kind="thought",
        content=(Thought(text, signature),),
        meta={"author": author},
        complete=True,
    )


def _tool_call(author: str, call: ToolCall) -> Event:
    return Event(
        id=new_id(),
        ts=time(),
        kind="tool_call",
        content=(call,),
        meta={"author": author},
        complete=True,
    )


def test_render_collapses_authors_to_roles() -> None:
    """``llm`` if the author is the assistant, ``user`` otherwise (DESIGN2 §3)."""
    state = (
        message(HUMAN, "hello"),
        message(ASSISTANT, "hi there"),
    )
    prompt = to_prompt(state)
    assert prompt.messages[0].role == "user"
    assert prompt.messages[1].role == "llm"


def test_render_skips_usage_turn_start_turn_end() -> None:
    """Journal vocabulary with no inference equivalent is dropped."""
    state = (
        message(HUMAN, "hi"),
        Event(id=new_id(), ts=time(), kind="turn_start", meta={"author": ASSISTANT}),
        message(ASSISTANT, "hello"),
        Event(
            id=new_id(),
            ts=time(),
            kind="turn_end",
            meta={"author": ASSISTANT, "reason": "relinquished"},
        ),
        Event(id=new_id(), ts=time(), kind="usage", meta={"author": ASSISTANT}),
    )
    prompt = to_prompt(state)
    assert len(prompt.messages) == 2
    assert prompt.messages[0].role == "user"
    assert prompt.messages[1].role == "llm"


def test_render_includes_interrupted_messages() -> None:
    """An interrupted (``complete=False``) message is ground truth and is rendered.

    ``complete=False`` means the message was cut short but journalled -- it
    was shown to the user and is immutable.  It must render, not be skipped.
    """
    e1 = message(HUMAN, "hello")
    partial = Event(
        id=new_id(),
        ts=time(),
        kind="message",
        content=(Media("text/plain", b"partial"),),
        meta={"author": ASSISTANT},
        complete=False,
    )
    prompt = to_prompt((e1, partial))
    assert len(prompt.messages) == 2
    assert prompt.messages[0].role == "user"
    assert prompt.messages[1].role == "llm"


def test_render_converts_content_blocks() -> None:
    """A thought and a message in one LLM turn become one ``llm`` Message."""
    state = (
        message(HUMAN, "what is 2+2?"),
        _thought(ASSISTANT, "reasoning..."),
        message(ASSISTANT, "4"),
    )
    prompt = to_prompt(state)
    assert len(prompt.messages) == 2
    llm_msg = prompt.messages[1]
    assert llm_msg.role == "llm"
    assert len(llm_msg.parts) == 2


def test_render_groups_consecutive_same_role() -> None:
    """Two human messages in a row fold into one ``user`` Message with two parts."""
    state = (
        message(HUMAN, "a"),
        message(HUMAN, "b"),
        message(ASSISTANT, "c"),
    )
    prompt = to_prompt(state)
    assert len(prompt.messages) == 2
    assert prompt.messages[0].role == "user"
    assert len(prompt.messages[0].parts) == 2


def test_render_tool_call_becomes_a_call_part() -> None:
    """A tool_call content block round-trips through ``to_part`` into a ``Call``."""
    call = ToolCall(id="c1", tool="get_weather", args='{"city": "Paris"}')
    state = (
        message(HUMAN, "weather?"),
        _tool_call(ASSISTANT, call),
    )
    prompt = to_prompt(state)
    llm_msg = prompt.messages[1]
    assert llm_msg.role == "llm"

    assert isinstance(llm_msg.parts[0], Call)
    assert llm_msg.parts[0].tool == "get_weather"


def test_render_empty_state_yields_empty_prompt() -> None:
    """An empty conversation renders to an empty prompt."""
    prompt = to_prompt(())
    assert isinstance(prompt, Prompt)
    assert not prompt.messages


def test_render_preserves_order() -> None:
    """Messages appear in conversation order, not grouped out of order."""
    state = (
        message(HUMAN, "first"),
        message(ASSISTANT, "second"),
        message(HUMAN, "third"),
    )
    prompt = to_prompt(state)
    roles = [m.role for m in prompt.messages]
    assert roles == ["user", "llm", "user"]
