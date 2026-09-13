"""Validation helpers shipped with the API.

``check_stream`` verifies the stream ordering contract:

1. ``Finish`` is the last item of a stream that ran to completion.
2. There is at most one ``Finish``.

Per-slot ordering (deltas sharing a ``slot`` arrive in order) is an inherent
property of sequential iteration and is not separately checked.  Interleaving
between slots is explicitly permitted, so no block-ordering is asserted.

``check`` validates a :class:`Prompt` against the referential invariants that
the type system cannot express.
"""

from dataclasses import dataclass
from typing import Iterable

from patchbay_llm.infer_engine.delta import Delta, Finish
from patchbay_llm.infer_engine.prompt import Call, Prompt, Result, Thought

__all__ = ["Problem", "check", "check_stream"]


@dataclass(frozen=True)
class Problem:
    """A validation problem found by :func:`check`."""

    message: str


def check(prompt: Prompt) -> tuple[Problem, ...]:
    """Validate a prompt.  An empty result means the prompt is valid.

    The invariants that matter are referential (which the type system cannot
    express), so there is one mechanism instead of splitting ``Part`` into
    ``UserPart`` / ``LlmPart``.
    """
    problems: list[Problem] = []

    # Directives may not contain Call/Result -- those are conversational.
    for part in prompt.directives:
        if isinstance(part, Call | Result):
            problems.append(
                Problem(f"directives must not contain {type(part).__name__}")
            )

    seen_calls: set[str] = set()
    pending_calls: set[str] = set()

    for msg in prompt.messages:
        if msg.role == "user":
            for part in msg.parts:
                if isinstance(part, Thought):
                    problems.append(Problem("Thought may only appear in llm messages"))
                if isinstance(part, Call):
                    problems.append(Problem("Call may only appear in llm messages"))
                if isinstance(part, Result):
                    if part.call not in seen_calls:
                        problems.append(
                            Problem(f"Result references unknown Call id {part.call!r}")
                        )
                    pending_calls.discard(part.call)
        else:  # llm
            # Every Call from the previous llm message must be answered before
            # this one arrives.
            if pending_calls:
                problems.append(
                    Problem(
                        f"unanswered Call(s) before next llm message: {sorted(pending_calls)}"
                    )
                )
                pending_calls.clear()
            for part in msg.parts:
                if isinstance(part, Result):
                    problems.append(Problem("Result may only appear in user messages"))
                if isinstance(part, Call):
                    seen_calls.add(part.id)
                    pending_calls.add(part.id)

    # Trailing unanswered Calls (the model just called tools) are normal -- the
    # round is not finished yet -- so they are not flagged here.
    return tuple(problems)


def check_stream(deltas: Iterable[Delta]) -> None:
    """Assert the stream ordering contract.  Raises ``AssertionError`` on violation."""
    saw_finish = False
    for delta in deltas:
        if saw_finish:
            raise AssertionError("delta arrived after Finish")
        if isinstance(delta, Finish):
            saw_finish = True
