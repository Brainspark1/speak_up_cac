import io
import json
import logging
import threading
import time
import wave

import numpy as np
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from ..library.SemanticMapper import SemanticMapper

logger = logging.getLogger("nes_voice")


class VoiceController:
    def __init__(
        self,
        mapping_json_path,
        device_backend="mps",
        whisper_model_size="tiny.en",
        initial_prompt=None,
    ):
        """
        :param mapping_json_path: path to the game-specific json file containing character/item/target memory addresses
        :param device_backend: hardware backend currently found in user's computer (write either "mps" for MacOS, "cuda" for Nvidia, or "cpu")
        :param model_size: size variant for OpenAI Whisper core (defaulted to tiny English model for minimal latency issues)
        :param initial_prompt: context prompt to guide Whisper transcriptions
        """

        self.lock = (
            threading.Lock()
        )  # prevents multiple threads from occurring simultaneously, allowing for actions to be completed fully for duration before next action is started
        self.device_backend = device_backend
        self.initial_prompt = (
            initial_prompt
            or 'Make sure to listen for NES gameplay commands like "jump on that enemy."'
        )

        # loading game-specific memory mappings from json file
        self.game_mappings = {}
        self.load_game_mappings(mapping_json_path)

        # semantic mapper used to resolve transcribed speech into targets/actions
        # defined in the game-specific json config - entire point is to pass the json file to the semantic mapper, not using in this file
        self.semantic_mapper = SemanticMapper(mapping_json_path)

        # initializing transcription and ner pipelines
        self.initialize_transcription_models(whisper_model_size)
        self.initialize_model_pipeline("Saggarwal/GAMEBERT")  # passing in bert model

        # setting up classes to capture audio
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone(sample_rate=16000)  # fixed recording rate at 16 kHz
        self.calibrate_microphone()

    # method to load and parse the json file that maps objects to memory addresses
    def load_game_mappings(self, json_path):
        try:
            with open(json_path, "r") as f:
                content = f.read()

            self.game_mappings = json.loads(content)
            logger.info(f"Successfully loaded game mappings from {json_path}")

        except Exception as e:
            logger.error(f"Failed to load game mappings: {e}")
            self.game_mappings = {}

    # method to initialize the tokenizer and model pipeline with the passed in model path (Saggarwal/token_bert before sarthak builds the next bert)
    def initialize_model_pipeline(self, model_path):
        logger.info(f"Loading GAMEBERT Token Classification Model from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)

        # checking hardware components found
        if self.device_backend == "cuda":
            device_str = "cuda:0"
        elif self.device_backend == "mps":
            try:
                self.model = self.model.to("mps")
                device_str = "mps"
            except Exception:
                logger.warning("Failed to move GAMEBERT to MPS. Defaulting to CPU.")
                device_str = "cpu"
        else:
            device_str = "cpu"

        # loading the pipeline
        self.nlp_pipeline = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=device_str,
        )

    # method to set up whisper engine based on hardware availability
    def initialize_transcription_models(self, model_size):
        if self.device_backend == "mps":
            try:
                import mlx_whisper

                self.mlx_whisper = mlx_whisper
                self.whisper_model_path = f"mlx-community/whisper-{model_size}-mlx"

                logger.info(
                    f"Initialized MPS Whisper backend: {self.whisper_model_path}"
                )

            # error if mlx_whisper not imported
            except ImportError:
                raise ImportError(
                    "mlx_whisper is required for MPS. Run: pip install mlx-whisper"
                )

        elif self.device_backend == "cuda":
            try:
                from faster_whisper import WhisperModel

                self.whisper_model = WhisperModel(
                    model_size, device="cuda", compute_type="float16"
                )

                logger.info(f"Initialized CUDA Whisper backend (Size: {model_size})")

            # error if faster_whisper not imported
            except ImportError:
                raise ImportError(
                    "faster_whisper is required for CUDA. Run: pip install faster-whisper"
                )
        else:
            raise ValueError(
                "Unsupported backend model, choose either 'mps' or 'cuda'."
            )

    # method to calibrate microphone based on background noise
    def calibrate_microphone(self, duration=2):
        with self.mic as source:
            logger.info("Adjusting for background noise for {duration} seconds...")

            self.recognizer.adjust_for_ambient_noise(source, duration=duration)

    # method to store temporary wav file in 1D numpy array to minimize latency - taken from StackOverflow, fine-tuned with AI agent
    def audio_to_numpy(self, audio):
        wav_data = audio.get_wav_data()

        with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
            frames = wav_file.readframes(
                wav_file.getnframes()
            )  # reading audio frames from the recording wav file

            # reading every pairs of bytes of binary audio stream as int16 first as int16 can read 2 bytes at a time, then converting it to float32 to return a normalized range of values in the array from -1.0 to 1.0
            audio_16 = np.frombuffer(frames, dtype=np.int16)
            audio_np = audio_16.astype(np.float32)

            audio_np = (
                audio_np / 32768.0
            )  # dividing by maximum value for signed 16-bit integer (negative included) to normalize range of values in numpy array

        return audio_np

    # method to transcribe audio using loaded whisper model
    def transcribe_audio(self, audio_np):
        if self.device_backend == "mps":
            result = self.mlx_whisper.transcribe(
                audio_np,
                path_or_hf_repo=self.whisper_model_path,
                language="en",
                initial_prompt=self.initial_prompt,
            )

            # returning text part of transcription, cleaning up spaces before and after to prevent noise
            return result["text"].strip()

        elif self.device_backend == "cuda":
            segments, _ = (
                self.whisper_model.transcribe(  # only parsing the segments, discarding everything else using _ operator in explosion
                    audio_np, language="en", initial_prompt=self.initial_prompt
                )
            )

            # joining collected segments together into transcribed sentence
            return "".join([segment.text for segment in segments]).strip()

    # method to pass text through GAMEBERT pipeline to extract tags/entity names
    def extract_entities(self, text):

        # return nothing if no text to pass through the bert
        if not text:
            return []

        return self.nlp_pipeline(text)

    # method to combine all functions together to transcribe audio, extract entities and process game commands based on method user needs to implement in subclass
    def audio_callback(
        self, recognizer, audio
    ):  # recognizer included just to prevent error throwing in speech recognition package
        try:
            start_time = time.time()  # to check for latency, current time

            audio_np = self.audio_to_numpy(audio)  # converting audio to 1D numpy array
            raw_text = self.transcribe_audio(
                audio_np
            )  # transcribing the audio calling on loaded whisper model

            if not raw_text or len(raw_text.split()) > 15:
                return

            raw_text = raw_text.lower()
            entities = self.extract_entities(raw_text)

            logger.info(
                f"Transcript: {raw_text}, Tags: {entities}, Latency Time: {time.time() - start_time:.4f}s"
            )

            # if entities have been recognized, reference them with json mapping into memory adddresses for bert to use
            if entities:
                self.process_game_commands(entities)

        except Exception as e:
            logger.error(f"Audio Callback Error: {e}")

    def start_listening(self, phrase_time_limit=1.5):
        return self.recognizer.listen_in_background(
            self.mic, self.audio_callback, phrase_time_limit=phrase_time_limit
        )

    # method must be overriden by developers for emulator loop to translate extracted entities into use case/game controls
    def process_game_commands(self, entities):

        # raise error if not implemented
        raise NotImplementedError(
            "Subclasses or game wrappers must implement `process_game_commands(self, entities)`"
        )
