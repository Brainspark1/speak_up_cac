"""Whack-a-Word backend: FastAPI + WebSocket.

Run from the project root:
    .venv/bin/python -m backend.app                 # voice + typed (uses the server's microphone)
    .venv/bin/python -m backend.app --typed-only    # NER only, no mic, no Whisper

Then open http://localhost:8000
"""

import argparse
import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .arbiter import WhackArbiter
from .config import DEFAULT_CONFIG_PATH, load_config, write_mapping
from .entities import clean_entities
from .ranked_mapper import RankedMapper
from .resolver import Resolver
from .tracker import MoleTracker

logger = logging.getLogger("whack.app")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def now_ms():
    return time.time() * 1000


class Game:
    """Server state shared by the WebSocket, /match, and the listener thread."""

    def __init__(self, config, backend=None, typed_only=False):
        self.config = config
        self.voice_cfg = config["voice"]
        self.backend = backend or self.voice_cfg.get("device_backend", "cpu")
        self.typed_only = typed_only
        self.mapping_path = write_mapping(config)
        self.mapper = RankedMapper(self.mapping_path)
        self.resolver = Resolver(config, self.mapper)
        timing = config["timing"]
        self.tracker = MoleTracker(start_tolerance_ms=timing.get("UTTERANCE_START_TOLERANCE_MS", 500))
        self.arbiter = WhackArbiter(
            windup_ms=timing["WINDUP_MS"],
            regret_window_ms=timing["REGRET_WINDOW_MS"],
            refund_fraction=config["penalties"]["regret_refund_fraction"],
        )
        self.controller = None
        self.stop_listening = None
        self.clients = set()
        self.loop = None
        self.queue = None
        self.health = {
            "phase": "starting",
            "model_loaded": False,
            "ner_model": self.voice_cfg["ner_model"],
            "id2label": None,
            "backend": self.backend,
            "whisper_model": self.voice_cfg.get("whisper_model_size"),
            "voice_enabled": not typed_only,
            "mic_calibrated": False,
            "mic_error": None,
            "mic_warning": None,
            "energy_threshold": None,
            "error": None,
        }

    # ---------- status ----------

    def set_status(self, phase, **info):
        self.health["phase"] = phase
        self.health.update({k: v for k, v in info.items() if k in self.health})
        msg = {"type": "status", "health": dict(self.health), **info}
        if self.loop:
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self.broadcast(msg)))

    # ---------- model loading (background thread) ----------

    def load_models(self):
        from .voice import MoleVoiceController

        ner = self.voice_cfg["ner_model"]
        try:
            if self.typed_only:
                self.set_status("loading_models")
                self.controller = MoleVoiceController.text_only(ner, device_backend=self._ner_device(),
                                                                ner_aggregation=self.voice_cfg.get("ner_aggregation", "first"))
            else:
                self.controller = MoleVoiceController(
                    str(self.mapping_path), ner_model=ner, emit=self.emit_from_thread,
                    status=self._status_from_thread, device_backend=self.backend,
                    whisper_model_size=self.voice_cfg.get("whisper_model_size", "tiny.en"),
                    initial_prompt=self.voice_cfg.get("initial_prompt"),
                    pause_threshold=self.voice_cfg.get("pause_threshold"),
                    transcript_fixes=self.voice_cfg.get("transcript_fixes"),
                    transcript_fix_skip_after=self.voice_cfg.get("transcript_fix_skip_after", []),
                    ner_aggregation=self.voice_cfg.get("ner_aggregation", "first"),
                )
        except Exception as e:
            logger.exception("Voice controller failed to load; falling back to typed-only")
            self.health["error"] = f"{type(e).__name__}: {e}"
            try:
                self.controller = MoleVoiceController.text_only(ner, device_backend=self._ner_device(),
                                                                ner_aggregation=self.voice_cfg.get("ner_aggregation", "first"))
                self.health["mic_error"] = "voice unavailable: " + self.health["error"]
            except Exception as e2:
                logger.exception("NER model failed to load")
                self.set_status("error", error=f"{type(e2).__name__}: {e2}")
                return

        self.health["model_loaded"] = True
        self.health["id2label"] = {str(k): v for k, v in self.controller.id2label.items()}
        if self.controller.mic_error:
            if not self.typed_only and not self.health["mic_error"]:
                self.health["mic_error"] = self.controller.mic_error
            self.set_status("typed_only")
            return
        self.health["mic_calibrated"] = True
        warm_error = self._warm_up()
        if warm_error:
            # e.g. cuda selected but libcublas.so.12 missing: only fails on the first transcription
            self.health["error"] = f"Whisper failed on '{self.backend}' backend: {warm_error}. Try --backend cpu."
            self.health["mic_error"] = "voice disabled: " + self.health["error"]
            self.set_status("typed_only")
            return
        self.stop_listening = self.controller.start_listening(
            phrase_time_limit=self.voice_cfg.get("phrase_time_limit"))
        self.set_status("listening")

    def _warm_up(self):
        # The first Whisper call is ~4x slower than later ones; pay that cost before play starts.
        import numpy as np

        t0 = time.time()
        try:
            self.controller.transcribe_audio(np.zeros(16000, dtype=np.float32))
            self.controller.extract_entities("whack the chef")
            logger.info("Warm-up done in %.0f ms", (time.time() - t0) * 1000)
            return None
        except Exception as e:
            logger.exception("Warm-up failed")
            return f"{type(e).__name__}: {e}"

    def _ner_device(self):
        # the package moves NER to cuda:0 for "cuda" and to cpu for anything else
        return self.backend if self.backend in ("cuda", "mps") else "cpu"

    def _status_from_thread(self, phase, **info):
        if phase == "calibrated":
            self.health["mic_calibrated"] = True
            self.health["energy_threshold"] = info.get("energy_threshold")
            self.health["mic_warning"] = info.get("mic_warning")
            return
        self.set_status(phase, **info)

    def emit_from_thread(self, event):
        self.loop.call_soon_threadsafe(self.queue.put_nowait, event)

    # ---------- utterance handling ----------

    async def consume(self):
        while True:
            event = await self.queue.get()
            try:
                await self.handle_utterance(event)
            except Exception:
                logger.exception("Failed to handle utterance %r", event)

    async def handle_utterance(self, event):
        if event.get("error"):
            await self.broadcast({"type": "status", "health": dict(self.health), "error": event["error"]})
            return
        entities = clean_entities(event.get("entities"))
        spoken_at = event.get("spoken_at") or now_ms()
        utter = {
            "type": "utterance",
            "source": event.get("source", "voice"),
            "transcript": event.get("transcript", ""),
            "heard": event.get("heard"),
            "fixed_words": event.get("fixed_words") or [],
            "entities": entities,
            "latency_ms": event.get("latency_ms"),
            "whisper_ms": event.get("whisper_ms"),
            "ner_ms": event.get("ner_ms"),
            "audio_ms": event.get("audio_ms"),
            "timestamp": event.get("timestamp", now_ms()),
            "spoken_at": spoken_at,
            "dropped": event.get("dropped"),
            "locked": self.tracker.input_locked,
        }
        if event.get("dropped") or not utter["transcript"]:
            await self.broadcast({**utter, "outcome": "no_match", "dropped": event.get("dropped") or "empty"})
            return
        candidates = self.tracker.candidates(spoken_at)
        await self.process(utter, entities, candidates, self.tracker.mission_tag, spoken_at, play=True)

    async def process(self, utter, entities, candidates, mission_tag, spoken_at, play):
        result = self.resolver.resolve(utter["transcript"], entities, candidates, mission_tag)
        result["spoken_at"] = spoken_at
        t = now_ms()
        result["resolve_ms"] = round(t - utter["timestamp"], 2) if utter.get("timestamp") else None
        if not play:
            return result
        locked = self.tracker.input_locked
        await self.broadcast({**utter, **{k: result.get(k) for k in
                                          ("target_words", "action_words", "correction_words", "query",
                                           "mode", "top", "outcome")}, "locked": locked})
        if locked:
            return result

        base = {"spoken_at": spoken_at, "transcript": utter["transcript"], "source": utter["source"],
                "latency_ms": utter.get("latency_ms")}
        if result["correction"]:
            effect = self.arbiter.correction(spoken_at, t)
            await self.broadcast({"type": "correction", **base, **effect,
                                  "words": result["correction_words"]})

        outcome = result["outcome"]
        if outcome == "whack":
            pending, superseded = self.arbiter.begin_windup(result, t)
            if superseded:
                await self.broadcast({"type": "correction", **base, "effect": "superseded",
                                      "whack": superseded["whack"]})
            await self.broadcast({"type": "whack", "phase": "windup", "windup_id": pending["id"],
                                  "strike_at": pending["strike_at"], **base, **result})
            self.loop.call_later(self.arbiter.windup_ms / 1000, self._fire_strike, pending["id"])
        elif outcome in ("ambiguous", "no_match"):
            await self.broadcast({"type": outcome, **base, **result})
        return result

    def _fire_strike(self, windup_id):
        p = self.arbiter.strike_due(windup_id, now_ms() + 1)
        if p:
            asyncio.ensure_future(self.broadcast({
                "type": "whack", "phase": "strike", "windup_id": windup_id, **p["whack"]}))

    async def match_text(self, text, visible_moles=None, mission_tag=None, play=False):
        if not self.controller:
            raise RuntimeError("model still loading")
        text = text.strip().lower()
        t0 = now_ms()
        entities = clean_entities(await asyncio.to_thread(self.controller.extract_entities, text))
        utter = {"type": "utterance", "source": "typed", "transcript": text, "entities": entities,
                 "latency_ms": round(now_ms() - t0, 1), "timestamp": now_ms(), "spoken_at": t0,
                 "dropped": None}
        if visible_moles is not None:
            candidates = [{**m, "key": m.get("key") or m["id"]} for m in visible_moles]
        else:
            candidates = self.tracker.candidates(t0)
            mission_tag = mission_tag if mission_tag is not None else self.tracker.mission_tag
        result = await self.process(utter, entities, candidates, mission_tag, t0, play=play)
        return {"entities": entities, "latency_ms": utter["latency_ms"], **result}

    # ---------- websocket ----------

    async def broadcast(self, msg):
        data = json.dumps(msg, default=float)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


def create_app(config_path=DEFAULT_CONFIG_PATH, backend=None, typed_only=None):
    config = load_config(config_path)
    if typed_only is None:
        typed_only = os.environ.get("WHACK_TYPED_ONLY") == "1"
    backend = backend or os.environ.get("WHACK_BACKEND")
    game = Game(config, backend=backend, typed_only=typed_only)

    @asynccontextmanager
    async def lifespan(app):
        game.loop = asyncio.get_running_loop()
        game.queue = asyncio.Queue()
        consumer = asyncio.create_task(game.consume())
        if os.environ.get("WHACK_SKIP_MODELS") != "1":
            threading.Thread(target=game.load_models, name="model-loader", daemon=True).start()
        yield
        consumer.cancel()
        if game.stop_listening:
            game.stop_listening(wait_for_stop=False)

    app = FastAPI(title="Whack-a-Word", lifespan=lifespan)
    app.state.game = game

    class MatchRequest(BaseModel):
        text: str
        visible_moles: list[dict] | None = None
        mission_tag: str | None = None
        play: bool = False

    @app.get("/health")
    async def health():
        return dict(game.health)

    @app.get("/config")
    async def get_config():
        return game.config

    @app.post("/match")
    async def match(req: MatchRequest):
        try:
            return await game.match_text(req.text, req.visible_moles, req.mission_tag, req.play)
        except RuntimeError as e:
            return {"error": str(e), "health": dict(game.health)}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        game.clients.add(ws)
        await ws.send_text(json.dumps({"type": "status", "health": dict(game.health)}))
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                kind = msg.get("type")
                if kind == "round_state":
                    game.tracker.update(msg.get("visible_moles", []), now_ms(),
                                        mission_tag=msg.get("mission_tag"),
                                        input_locked=msg.get("input_locked", False))
                elif kind == "reset":
                    game.tracker.reset()
                    game.arbiter.pending = None
                    game.arbiter.last_strike = None
        except WebSocketDisconnect:
            pass
        finally:
            game.clients.discard(ws)

    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


def main():
    parser = argparse.ArgumentParser(description="Whack-a-Word backend")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--backend", choices=["cpu", "cuda", "mps"], default=None,
                        help="overrides voice.device_backend in game_config.json")
    parser.add_argument("--typed-only", action="store_true", help="no mic/Whisper; typed input only")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import uvicorn
    uvicorn.run(create_app(args.config, args.backend, args.typed_only), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
