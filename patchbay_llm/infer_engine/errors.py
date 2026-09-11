"""Inference error classes.

Four classes, chosen to make exactly one decision answerable: *can the same
request be sent again?*

- ``Unreachable`` / ``Throttled``: yes.
- ``Rejected``: never.  ``TooLarge`` is a subclass because context overflow is a
  prompt-construction bug that should be loud rather than handled.

There are exactly three ways a stream ends (§5): completed (a ``Finish`` item),
failed (an exception out of ``async for``), or abandoned (the consumer stops).
A silent exhaustion with neither ``Finish`` nor exception is forbidden; the
adapter raises ``Unreachable`` if the provider closes the connection without a
terminal chunk, and synthesises an ``estimated`` ``Usage`` if it finishes
properly but omits counts.
"""

from __future__ import annotations


class InferenceError(Exception):
    """Base class for every inference failure."""


class Unreachable(InferenceError):
    """The transport died; the same request is still valid and may be resent."""


class Throttled(InferenceError):
    """A 429/503 response; carries ``retry_after`` when the provider gave one."""

    def __init__(self, message: str = "", retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class Rejected(InferenceError):
    """A 4xx response; the request is wrong and will stay wrong."""


class TooLarge(Rejected):
    """The prompt exceeded the model's context window."""
