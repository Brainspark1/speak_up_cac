"""MoleVoiceController: voice-speak-up's VoiceController adapted for Whack-a-Word.

The installed package is not modified. Overrides:
  - _init_ner_pipeline: loads GAMEBERT (from config), not the hardcoded NESBERT, with a
    configurable aggregation strategy (default "first": one label per whole word)
  - _init_transcription_engine / transcribe_audio: adds a "cpu" backend (faster_whisper int8)
  - _calibrate_mic: reports status so the UI can show "stay quiet"
  - audio_callback: emits an event for EVERY utterance, including ones with no entities
"""

import importlib
import logging

import time

from .entities import fix_mishearings

_su = importlib.import_module("speak-up")
VoiceController = _su.VoiceController

logger = logging.getLogger("whack.voice")


def preload_cuda12_libs():
    """faster_whisper's CTranslate2 needs CUDA 12 cuBLAS/cuDNN, but torch may ship CUDA 13 ones.
    If the nvidia-cublas-cu12 / nvidia-cudnn-cu12 wheels are installed, load them globally so
    CTranslate2 finds them without LD_LIBRARY_PATH."""
    import ctypes
    import glob
    import os
    import site

    loaded = []
    for sp in site.getsitepackages():
        for sub in ("cublas", "cudnn"):
            libs = sorted(glob.glob(os.path.join(sp, "nvidia", sub, "lib", "lib*.so.*")))
            # cuBLASLt before cuBLAS, cudnn core before its sub-libraries
            libs.sort(key=lambda p: (0 if "cublasLt" in p or p.endswith("libcudnn.so.9") else 1, p))
            for lib in libs:
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(os.path.basename(lib))
                except OSError:
                    pass
    if loaded:
        logger.info("Preloaded CUDA 12 libs: %s", ", ".join(loaded))
    return loaded


class MoleVoiceController(VoiceController):
    def __init__(self, mapping_json_path, ner_model, emit, status=None,
                 device_backend="cpu", whisper_model_size="tiny.en", initial_prompt=None,
                 pause_threshold=None, transcript_fixes=None, transcript_fix_skip_after=(),
                 ner_aggregation="first"):
        """
        emit(event): called from the listener thread with every utterance event.
        status(phase, **info): progress callback ("loading_models", "calibrating", ...).
        """
        self.ner_model_id = ner_model
        self.ner_aggregation = ner_aggregation
        self.emit = emit
        self.status = status or (lambda phase, **info: None)
        self.mic_error = None
        self.mic_warning = None
        self.mic_calibrated = False
        self.pause_threshold = pause_threshold
        self.transcript_fixes = {k.lower(): v for k, v in (transcript_fixes or {}).items()}
        self.transcript_fix_skip_after = transcript_fix_skip_after
        self.status("loading_models", backend=device_backend)
        try:
            super().__init__(mapping_json_path, device_backend=device_backend,
                             whisper_model_size=whisper_model_size, initial_prompt=initial_prompt)
        except Exception as e:
            # Whisper and NER load before the mic opens. If they are loaded, keep this
            # object usable for typed input and report the mic failure.
            if getattr(self, "nlp_pipeline", None) is not None and not self.mic_calibrated:
                self.mic_error = f"{type(e).__name__}: {e}"
                logger.error("Microphone unavailable: %s", self.mic_error)
            else:
                raise
        if self.pause_threshold is not None and getattr(self, "recognizer", None) is not None:
            # sr.Recognizer waits this long in silence before ending a phrase (default 0.8 s)
            self.recognizer.pause_threshold = self.pause_threshold

    # ---------- text-only construction (calibrate.py, typed mode without mic/whisper) ----------

    @classmethod
    def text_only(cls, ner_model, device_backend="cpu", ner_aggregation="first"):
        """Loads only the NER pipeline, so extract_entities works with no mic and no Whisper."""
        obj = cls.__new__(cls)
        obj.ner_model_id = ner_model
        obj.ner_aggregation = ner_aggregation
        obj.device_backend = device_backend
        obj.mic_error = "text-only mode"
        obj.mic_warning = None
        obj.mic_calibrated = False
        import threading
        obj.lock = threading.Lock()
        obj._init_ner_pipeline(ner_model)
        return obj

    # ---------- overrides ----------

    def _init_ner_pipeline(self, model_path):
        # The package passes the hardcoded "Saggarwal/NESBERT". Ignore it and load GAMEBERT.
        logger.info("Ignoring hardcoded NER model %r; loading %r", model_path, self.ner_model_id)
        super()._init_ner_pipeline(self.ner_model_id)
        if self.ner_aggregation != "simple":
            # The package hardcodes "simple", which labels each subword on its own, so one stray
            # "##" piece can split a word into two entities. "first" labels the whole word by its
            # first subword. Same model, tokenizer and device.
            from transformers import pipeline

            self.nlp_pipeline = pipeline(
                "token-classification", model=self.model, tokenizer=self.tokenizer,
                aggregation_strategy=self.ner_aggregation, device=self.nlp_pipeline.device,
            )
            logger.info("NER aggregation strategy: %s", self.ner_aggregation)
        labels = dict(self.model.config.id2label)
        logger.info("GAMEBERT id2label: %s", labels)
        expected = {"O", "B-ACTION", "I-ACTION", "B-TARGET", "I-TARGET", "B-CORRECTION", "I-CORRECTION"}
        if set(labels.values()) != expected:
            logger.warning("Unexpected GAMEBERT labels %s (expected %s)", sorted(labels.values()), sorted(expected))
        self.id2label = labels

    def _init_transcription_engine(self, model_size):
        if self.device_backend == "cpu":
            from faster_whisper import WhisperModel

            self.whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            logger.info("Initialized CPU Whisper backend (Size: %s, int8)", model_size)
            return
        if self.device_backend == "cuda":
            preload_cuda12_libs()
        super()._init_transcription_engine(model_size)

    def transcribe_audio(self, audio_np):
        if self.device_backend == "cpu":
            # same segment-joining logic as the package's cuda branch
            segments, _ = self.whisper_model.transcribe(
                audio_np, language="en", initial_prompt=self.initial_prompt
            )
            return "".join([segment.text for segment in segments]).strip()
        return super().transcribe_audio(audio_np)

    def _calibrate_mic(self, duration=2):
        self.status("calibrating", seconds=duration)
        super()._calibrate_mic(duration)
        self.mic_calibrated = True
        # A muted or dead mic reads RMS 0, so calibration settles near 0 and nothing ever triggers.
        if self.recognizer.energy_threshold < 15:
            self.mic_warning = (f"Microphone looks silent (energy {self.recognizer.energy_threshold:.0f}). "
                                "Is it muted or is the wrong input device selected?")
            logger.warning(self.mic_warning)
        self.status("calibrated", energy_threshold=round(self.recognizer.energy_threshold, 1),
                    mic_warning=self.mic_warning)

    def extract_entities(self, text):
        # The listener thread and typed /match requests share one pipeline.
        with self.lock:
            return super().extract_entities(text)

    def audio_callback(self, recognizer, audio):
        received_at = time.time()
        try:
            audio_np = self.audio_to_numpy(audio)
            audio_ms = len(audio_np) / 16000 * 1000
            raw_text = self.transcribe_audio(audio_np) or ""
            transcribed_at = time.time()

            words = raw_text.split()
            dropped = None
            if not words:
                dropped = "empty"

            heard = raw_text.lower()
            raw_text, fixed = fix_mishearings(heard, self.transcript_fixes, self.transcript_fix_skip_after)
            entities = [] if dropped else self.extract_entities(raw_text)
            done = time.time()

            event = {
                "transcript": raw_text,
                "heard": heard,
                "fixed_words": fixed,
                "entities": entities,
                "latency_ms": round((done - received_at) * 1000, 1),
                "whisper_ms": round((transcribed_at - received_at) * 1000, 1),
                "ner_ms": round((done - transcribed_at) * 1000, 1),
                "audio_ms": round(audio_ms, 1),
                "timestamp": done * 1000,
                # when the phrase started, so whacks count for moles that were up then
                "spoken_at": (received_at * 1000) - audio_ms,
                "dropped": dropped,
                "source": "voice",
            }
            logger.info("Transcript: %r | NER: %s | Latency: %.0f ms",
                        raw_text, [(e["entity_group"], e["word"]) for e in entities], event["latency_ms"])
            self.emit(event)
        except Exception as e:
            logger.exception("Audio Callback Error")
            self.emit({"error": f"{type(e).__name__}: {e}", "timestamp": time.time() * 1000, "source": "voice"})

    def process_game_commands(self, entities):
        # Required by the base class. Unused: audio_callback emits full events instead.
        pass
