"""Capture speech, resolve it to a command, and enqueue it for the main thread."""

from __future__ import annotations

import queue
import threading

import speech_recognition as sr

from desktop_automation.commands import DesktopCommand
from desktop_automation.resolver import DesktopCommandResolver


class VoiceCommandListener:
    """Microphone component; this class never calls Playwright."""

    def __init__(
        self, command_queue, resolver, shutdown_event: threading.Event
    ) -> None:
        self._command_queue: queue.Queue[DesktopCommand] = command_queue
        self._resolver: DesktopCommandResolver = resolver
        self._shutdown_event = shutdown_event
        self._recognizer = sr.Recognizer()
        self._microphone = sr.Microphone()
        self._stop_background_listener = None

    def start(self) -> None:
        """Calibrate briefly and start background microphone listening."""
        print("Calibrating microphone; please stay quiet briefly...")
        with self._microphone as source:
            self._recognizer.adjust_for_ambient_noise(source, duration=0.8)

        # SpeechRecognition invokes _audio_callback on a background thread.
        self._stop_background_listener = self._recognizer.listen_in_background(
            self._microphone,
            self._audio_callback,
            phrase_time_limit=4,
        )
        print("Listening for supported commands.")

    def _audio_callback(self, _recognizer, audio: sr.AudioData) -> None:
        """Transcribe and queue one command; do not execute it here."""
        if self._shutdown_event.is_set():
            return

        try:
            # For a private/offline app, replace this with local Whisper transcription.
            transcript = self._recognizer.recognize_google(audio)
            print(f"\nHeard transcript: {transcript}", flush=True)
        except sr.UnknownValueError:
            return
        except sr.RequestError as error:
            print(f"Speech transcription service error: {error}")
            return

        command = self._resolver.resolve(transcript)
        if command is None:
            print("Unrecognized command; no action taken.")
            return

        if command.action == "stop":
            # Clear waiting work so Stop takes priority over older queued actions.
            while True:
                try:
                    self._command_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._command_queue.task_done()

        self._command_queue.put(command)
        print(f'Heard "{transcript}"; queued "{command.action}".')

    def stop(self) -> None:
        """Stop background microphone listening during shutdown."""
        if self._stop_background_listener is not None:
            self._stop_background_listener(wait_for_stop=False)
            self._stop_background_listener = None
