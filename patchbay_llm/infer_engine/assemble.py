"""Prompt assembly.

A :class:`~patchbay_llm.infer_engine.prompt.Prompt` is built once, by an
assembler that sees every Section's contribution at the same time.
Breakpoint placement is inherently n-ary: a binary ``+`` cannot place a
breakpoint, because it does not know whether more contributions are coming or
how many breakpoints remain in the budget.
"""

from dataclasses import dataclass
from enum import Enum, auto

from patchbay_llm.infer_engine.prompt import Breakpoint, Message, Part, Prompt

__all__ = ["CacheHint", "Contribution", "place_breakpoints", "assemble"]


class CacheHint(Enum):
    """A Section's claim about how stable its contribution is.

    ``VOLATILE`` is the default; ``STABLE`` asks for a cache breakpoint at the
    end of the contribution.
    """

    VOLATILE = auto()
    STABLE = auto()


@dataclass(frozen=True)
class Contribution:
    """What a Section returns -- a *proposal*, not a finished prompt."""

    directives: tuple[Part, ...] = ()
    messages: tuple[Message, ...] = ()
    cache_hint: CacheHint = CacheHint.VOLATILE


def place_breakpoints(hints: list[CacheHint], budget: int) -> set[int]:
    """Choose which contribution indices get a breakpoint.

    Up to ``budget`` breakpoints are placed, preferring the most recent
    ``STABLE`` contributions (a later breakpoint covers a larger prefix).
    """
    if budget <= 0:
        return set()
    stable = [i for i, h in enumerate(hints) if h is CacheHint.STABLE]
    chosen = stable[-budget:]
    return set(chosen)


def assemble(cs: list[Contribution], breakpoints: int) -> Prompt:
    """Assemble contributions into a finished :class:`Prompt`.

    All directives precede all messages regardless of which contribution
    supplied them (the only ordering any provider can represent).  The
    leapfrogging rule is enforced: once a contribution has contributed
    messages, no later contribution may contribute directives.
    """
    chosen = place_breakpoints([c.cache_hint for c in cs], breakpoints)
    directives: tuple[Part, ...] = ()
    messages: tuple[Message, ...] = ()
    seen_messages = False
    for i, c in enumerate(cs):
        if c.directives and seen_messages:
            raise ValueError(
                "leapfrogging: a contribution contributed directives after an "
                "earlier one contributed messages"
            )
        directives = directives + c.directives
        messages = messages + c.messages
        if c.messages:
            seen_messages = True
        if i in chosen:
            # Mark a breakpoint at the end of whatever this contribution added.
            if c.messages:
                # Breakpoint is a Part -- it rides inside the last message's parts.
                last = messages[-1]
                messages = messages[:-1] + (
                    Message(last.role, last.parts + (Breakpoint(),)),
                )
            elif c.directives:
                directives = directives + (Breakpoint(),)
    return Prompt(directives, messages)
