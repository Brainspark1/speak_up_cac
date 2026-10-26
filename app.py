"""Start the browser, listen for voice commands, and execute queued actions."""

from __future__ import annotations

import queue
import signal
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

from desktop_automation.browser_executor import BrowserExecutor
from desktop_automation.commands import DesktopCommand
from desktop_automation.resolver import DesktopCommandResolver
from desktop_automation.safety import SafetyPolicy
from desktop_automation.voice_listener import VoiceCommandListener

PROJECT_DIRECTORY = Path(__file__).resolve().parent
DEMO_PAGE_PATH = PROJECT_DIRECTORY / "test_page.html"
COMMAND_CONFIG_PATH = (
    PROJECT_DIRECTORY / "desktop_automation" / "config" / "commands.json"
)


def main() -> None:
    """Launch the local demo and process voice commands on the main thread."""

    command_queue: queue.Queue[DesktopCommand] = queue.Queue()
    shutdown_event = threading.Event()

    resolver = DesktopCommandResolver(
        config_path=COMMAND_CONFIG_PATH,
        device=-1,
    )

    safety_policy = SafetyPolicy()

    # Keep Playwright's synchronous API on this main thread.
    # The microphone callback should enqueue commands rather than
    # interacting with Playwright directly.
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)

        try:
            page = browser.new_page()
            page.goto(DEMO_PAGE_PATH.as_uri())

            executor = BrowserExecutor(page, safety_policy)
            listener = VoiceCommandListener(
                command_queue,
                resolver,
                shutdown_event,
            )

            # Register Ctrl+C handling on the main thread.
            signal.signal(
                signal.SIGINT,
                lambda signum, frame: shutdown_event.set(),
            )

            listener.start()

            while not shutdown_event.is_set():
                try:
                    command = command_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                try:
                    if command.action == "stop":
                        cleared = safety_policy.clear_pending_commands(command_queue)
                        print(
                            "Stop acknowledged; "
                            f"cleared {cleared} pending command(s)."
                        )
                        shutdown_event.set()
                        continue

                    result = executor.execute(command)
                    print(f"Action result: {result}")

                except Exception as error:
                    print("Action failed safely: " f"{type(error).__name__}: {error}")

                finally:
                    command_queue.task_done()

        finally:
            shutdown_event.set()

            # Stop the microphone listener before closing the browser.
            if "listener" in locals():
                try:
                    listener.stop()
                except Exception as error:
                    print(f"Listener shutdown warning: {error}")

            browser.close()
            print("Speak Up has shut down.")


if __name__ == "__main__":
    main()
