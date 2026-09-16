"""Contract tests for ``participants.llm.render`` (the state -> Prompt fold)."""

from time import time

from patchbay_llm.classic_theatre.conversation import message
from patchbay_llm.classic_theatre.participants.llm.render import render
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
    """``llm`` if the author is ``me``, ``user`` otherwise (DESIGN2 §3)."""
    state = (
        message("human:you", "hello"),
        message("llm:main", "hi there"),
    )
    prompt = render(state, "llm:main")
    assert prompt.messages[0].role == "user"
    assert prompt.messages[1].role == "llm"


def test_render_skips_usage_turn_start_turn_end() -> None:
    """Journal vocabulary with no inference equivalent is dropped."""
    state = (
        message("human:you", "hi"),
        Event(id=new_id(), ts=time(), kind="turn_start", meta={"author": "llm:main"}),
        message("llm:main", "hello"),
        Event(
            id=new_id(),
            ts=time(),
            kind="turn_end",
            meta={"author": "llm:main", "reason": "relinquished"},
        ),
        Event(id=new_id(), ts=time(), kind="usage", meta={"author": "llm:main"}),
    )
    prompt = render(state, "llm:main")
    assert len(prompt.messages) == 2
    assert prompt.messages[0].role == "user"
    assert prompt.messages[1].role == "llm"


def test_render_budget_is_a_noop() -> None:
    """``budget`` is accepted but unused -- a signature slot, not behaviour."""
    state = (message("human:you", "hello"),)
    assert render(state, "llm:main", budget=100) == render(
        state, "llm:main", budget=None
    )


def test_render_skips_partials() -> None:
    """Only complete content is rendered; a partial is not yet ground truth."""
    e1 = message("human:you", "hello")
    partial = Event(
        id=new_id(),
        ts=time(),
        kind="message",
        content=(Media("text/plain", b"partial"),),
        meta={"author": "llm:main"},
        complete=False,
    )
    prompt = render((e1, partial), "llm:main")
    assert len(prompt.messages) == 1
    assert prompt.messages[0].role == "user"


def test_render_converts_content_blocks() -> None:
    """A thought and a message in one LLM turn become one ``llm`` Message."""
    state = (
        message("human:you", "what is 2+2?"),
        _thought("llm:main", "reasoning..."),
        message("llm:main", "4"),
    )
    prompt = render(state, "llm:main")
    assert len(prompt.messages) == 2
    llm_msg = prompt.messages[1]
    assert llm_msg.role == "llm"
    assert len(llm_msg.parts) == 2


def test_render_groups_consecutive_same_role() -> None:
    """Two human messages in a row fold into one ``user`` Message with two parts."""
    state = (
        message("human:you", "a"),
        message("human:you", "b"),
        message("llm:main", "c"),
    )
    prompt = render(state, "llm:main")
    assert len(prompt.messages) == 2
    assert prompt.messages[0].role == "user"
    assert len(prompt.messages[0].parts) == 2


def test_render_tool_call_becomes_a_call_part() -> None:
    """A tool_call content block round-trips through ``to_part`` into a ``Call``."""
    call = ToolCall(id="c1", tool="get_weather", args='{"city": "Paris"}')
    state = (
        message("human:you", "weather?"),
        _tool_call("llm:main", call),
    )
    prompt = render(state, "llm:main")
    llm_msg = prompt.messages[1]
    assert llm_msg.role == "llm"

    assert isinstance(llm_msg.parts[0], Call)
    assert llm_msg.parts[0].tool == "get_weather"


def test_render_empty_state_yields_empty_prompt() -> None:
    """An empty conversation renders to an empty prompt."""
    prompt = render((), "llm:main")
    assert isinstance(prompt, Prompt)
    assert not prompt.messages


def test_render_preserves_order() -> None:
    """Messages appear in conversation order, not grouped out of order."""
    state = (
        message("human:you", "first"),
        message("llm:main", "second"),
        message("human:you", "third"),
    )
    prompt = render(state, "llm:main")
    roles = [m.role for m in prompt.messages]
    assert roles == ["user", "llm", "user"]
