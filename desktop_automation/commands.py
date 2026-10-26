"""Define and validate the small set of desktop commands the app supports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Action = Literal["scroll", "click_link", "stop", "zoom"]
Direction = Literal["up", "down", "in", "out", "reset"]
ScrollAmount = Literal["small", "page"]


@dataclass(frozen=True)
class DesktopCommand:
    """Structured command data; never store or execute arbitrary code here."""

    action: Action
    direction: Direction | None = None
    amount: ScrollAmount | None = None
    link_name: str | None = None

    def __post_init__(self) -> None:
        """Reject malformed commands even if they bypass the phrase resolver."""
        if self.action == "scroll":
            if self.direction not in ("up", "down"):
                raise ValueError("Scroll commands require direction 'up' or 'down'.")
            if self.amount not in ("small", "page"):
                raise ValueError("Scroll commands require amount 'small' or 'page'.")
            if self.link_name is not None:
                raise ValueError("Scroll commands cannot include a link name.")
            return

        if self.action == "click_link":
            if not self.link_name or not self.link_name.strip():
                raise ValueError("Click commands require a non-empty link name.")
            if self.direction is not None or self.amount is not None:
                raise ValueError("Click commands cannot include scroll arguments.")
            return

        if self.action == "stop":
            if any(
                value is not None
                for value in (self.direction, self.amount, self.link_name)
            ):
                raise ValueError("Stop commands cannot include additional arguments.")
            return

        if self.action == "zoom":
            if self.direction not in ("in", "out", "reset"):
                raise ValueError(
                    "Zoom commands require direction 'in', 'out', or 'reset'."
                )
            if self.amount is not None or self.link_name is not None:
                raise ValueError(
                    "Zoom commands cannot include scroll or link arguments."
                )
            return

        raise ValueError(f"Unsupported action: {self.action!r}")
