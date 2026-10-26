"""Execute bounded browser actions against the Playwright page owned by this app."""

from __future__ import annotations

from desktop_automation.commands import DesktopCommand
from desktop_automation.safety import SafetyPolicy
from desktop_automation.config_loader import CommandConfig

import re
import json


class BrowserExecutor:
    """Handle scrolling and clicking one unambiguous link by accessible name."""

    command_config = CommandConfig()

    # Scroll distances are fixed and bounded; voice input cannot set arbitrary values.
    SCROLL_PIXELS = {
        "small": command_config.small_scroll_length(),
        "page": command_config.page_scroll_length(),
    }

    ZOOM_LEVELS = command_config.get_zoom_values()
    DEFAULT_ZOOM = command_config.get_default_zoom()

    def __init__(self, page, safety_policy: SafetyPolicy) -> None:
        # Keep the page instance supplied by app.py; do not attach to random tabs.
        self._page = page
        self._safety_policy = safety_policy

        self._zoom_level = self.DEFAULT_ZOOM

    def execute(self, command: DesktopCommand) -> str:
        """Dispatch a validated command to the appropriate action method."""
        if command.action == "scroll":
            return self._scroll(command)
        if command.action == "click_link":
            return self._click_named_link(command.link_name)
        if command.action == "stop":
            # app.py handles stopping because it owns the queue and shutdown event.
            return "Stop command received."
        if command.action == "zoom":
            return self._zoom(command)
        raise ValueError(f"Unsupported command action: {command.action!r}")

    def _scroll(self, command: DesktopCommand) -> str:
        """Scroll up or down by one predefined distance."""
        if (
            command.direction not in ("up", "down")
            or command.amount not in self.SCROLL_PIXELS
        ):
            raise ValueError("Invalid scroll command.")

        distance = self.SCROLL_PIXELS[command.amount]
        if command.direction == "up":
            distance = -distance

        # This JavaScript receives only the bounded number selected above.
        self._page.evaluate(
            "(pixels) => window.scrollBy({ top: pixels, behavior: 'smooth' })",
            distance,
        )
        return f"Scrolled {command.direction} by {abs(distance)} pixels."

    def _click_named_link(self, link_name: str | None) -> str:
        """Find exactly one visible enabled link by accessible name, then confirm."""
        if not link_name:
            return "No link name was provided; nothing was clicked."

        # Exact matching prevents a request for "Docs" from accidentally matching
        # a link called "Docs and Downloads". Accessible names work with ARIA too.
        matches = self._page.get_by_role(
            "link",
            name=re.compile(rf"^{re.escape(link_name)}$", re.IGNORECASE),
        )

        print("Available links:", self._page.get_by_role("link").all_text_contents())
        print("Requested link:", repr(link_name))

        match_count = matches.count()

        if match_count == 0:
            return f'No link with the exact name "{link_name}" was found.'
        if match_count > 1:
            return (
                f'Multiple links named "{link_name}" were found. '
                "Please use a more specific name; nothing was clicked."
            )

        link = matches.nth(0)
        try:
            if not link.is_visible():
                return f'The link "{link_name}" is not visible; nothing was clicked.'
            if not link.is_enabled():
                return f'The link "{link_name}" is disabled; nothing was clicked.'

            # Require confirmation before interacting with a page element.
            if not self._safety_policy.confirm_link_click(link_name):
                return "Click cancelled; no link was clicked."

            # Recheck after confirmation in case the page changed while waiting.
            if not link.is_visible() or not link.is_enabled():
                return "The link changed during confirmation; nothing was clicked."
            if link.count() != 1:
                return "The link is no longer unique; nothing was clicked."

            link.click(timeout=2000)
        except Exception as error:
            return f"Click failed safely: {type(error).__name__}: {error}"

        return f'Clicked the link named "{link_name}".'

    def _zoom(self, command: DesktopCommand) -> str:
        if command.direction == "reset":
            self._zoom_level = self.DEFAULT_ZOOM

        elif command.direction == "in":
            next_levels = [
                level for level in self.ZOOM_LEVELS if level > self._zoom_level
            ]
            if not next_levels:
                return "Maximum zoom reached."

            self._zoom_level = next_levels[0]

        elif command.direction == "out":
            previous_levels = [
                level for level in self.ZOOM_LEVELS if level < self._zoom_level
            ]
            if not previous_levels:
                return "Minimum zoom reached."

            self._zoom_level = previous_levels[-1]

        else:
            return "Invalid zoom direction."

        self._page.evaluate(
            "(level) => { document.documentElement.style.zoom = level; }",
            self._zoom_level,
        )

        return f"Zoom set to {round(self._zoom_level * 100)}%."
