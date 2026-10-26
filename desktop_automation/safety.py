"""Helpers for explicit click confirmation and pending-command cancellation."""

from __future__ import annotations

import queue

from desktop_automation.commands import DesktopCommand


class SafetyPolicy:
    """Centralize user approval and queue-clearing behavior."""

    def confirm_link_click(self, link_name: str) -> bool:
        """Ask before clicking; no approval means no click."""
        print(f'\nPlanned action: click the link named "{link_name}".')
        try:
            answer = input("Type 'y' and press Enter to confirm: ")
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower() == "y"

    def clear_pending_commands(
        self, command_queue: "queue.Queue[DesktopCommand]"
    ) -> int:
        """Remove queued commands and return the number removed."""
        cleared = 0
        while True:
            try:
                command_queue.get_nowait()
            except queue.Empty:
                return cleared
            else:
                command_queue.task_done()
                cleared += 1
