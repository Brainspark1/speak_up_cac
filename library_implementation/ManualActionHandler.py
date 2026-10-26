import json
import logging

from ..library.VoiceController import VoiceController

logger = logging.getLogger("ManualActionHandler")

# if null, set to this large value for holding
HOLD_FRAMES = 999999


class ManualActionHandler(VoiceController):
    def __init__(
        self,
        mapping_json_path,
        auto_tracking_class,
        device_backend="mps",
        whisper_model_size="tiny.en",
        initial_prompt=None,
    ):
        super().__init__(
            mapping_json_path=mapping_json_path,
            device_backend=device_backend,
            whisper_model_size=whisper_model_size,
            initial_prompt=initial_prompt,
        )

        self.data = self.read_json_file(mapping_json_path)
        self.auto_tracking_class = auto_tracking_class

        self.duration_array = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.button_to_index_mapping = {
            "b": 0,
            "mode": 2,
            "start": 3,
            "up": 4,
            "down": 5,
            "left": 6,
            "right": 7,
            "a": 8,
        }

        self.action_data = self.data["actions"]
        self.target_data = self.data["targets"]
        self.names = []

        for action_name in self.action_data:
            self.names.append(action_name)

        for target_name in self.target_data:
            self.names.append(target_name)

        self.names_string = ", ".join(
            self.names
        )  # returns string of all actions and targets in json file, separated by commas

    def process_game_commands(self, entities):
        if not entities:
            return

        correction_words = [
            entity.get("word", "").strip().lower()
            for entity in entities
            if entity.get("entity_group") == "CORRECTION"
        ]

        target_words = [
            entity.get("word", "").strip().lower()
            for entity in entities
            if entity.get("entity_group") == "TARGET"
        ]

        action_words = [
            entity.get("word", "").strip().lower()
            for entity in entities
            if entity.get("entity_group") == "ACTION"
        ]

        # if any correction words detecting, assuming they are to cancel ongiong actions
        if correction_words:
            logger.info("Correction command received, stopping all actions")

            self.duration_array = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.auto_tracking_class.deactivate_tracking()

            return
        else:
            # handing off to AutoTracking class to set the target, runs each frame of main loop
            if target_words:
                sentence = " ".join(target_words)

                if "##" in sentence:
                    sentence = sentence.replace("#", "").replace(" ", "")
                logger.info(f"Target command received: {sentence}")

                name, score = self.auto_tracking_class.activate_set_target(sentence)

                if name:
                    logger.info(
                        f"Tracking started on {name} with confidence {score:.2f}"
                    )
                else:
                    logger.info(f"No target recognized in {sentence}")

                return

            # resolving into simple button hold via SemanticMapper
            if action_words:
                action_sentence = " ".join(action_words)

                if "##" in action_sentence:
                    action_sentence = action_sentence.replace("#", "").replace(" ", "")

                logger.info(f"Action command received: {action_sentence}")

                action_name, score = self.set_action_from_similarity(action_sentence)

                logger.info(f"Resolved action: {action_name} (score={score:.3f})")

                if not action_name or score < 0.15:
                    logger.info(f"No action recognized in {action_sentence}")

                    return

                if action_name.lower() == "stop" or action_name.lower() == "cancel":
                    logger.info("Stop command received")
                    self.duration_array = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    self.auto_tracking_class.deactivate_tracking()

                    return

                action_info = self.data["actions"].get(action_name, {})
                buttons = action_info.get("button_array", [])
                durations = action_info.get("duration_array", [])

                if not buttons:
                    logger.info(f"{action_name} has no button array")

                    return

                # loop to return duration array indexed by buttons - generated via autocomplete
                for i, button in enumerate(buttons):
                    index = self.button_to_index_mapping.get(button.lower())

                    if index is None:
                        continue

                    # current duration is the current duration of the button slot number being looked at if we haven't reached the end of the duration array
                    if i < len(durations):
                        duration = durations[i]
                    else:
                        duration = None  # assuming that if no duration found at this current index, button should be held until manually stopped

                    # handling set duration values
                    if duration is not None:
                        self.duration_array[index] = duration
                    else:
                        self.duration_array[index] = HOLD_FRAMES

                logger.info(f"Duration array after update: {self.duration_array}")

    def set_action_from_similarity(self, transcript_sentence, min_confidence=0.2):
        name, score = self.semantic_mapper.find_max_action_similarity(
            transcript_sentence
        )

        if score < min_confidence:
            return None, score

        return name, score

    def read_json_file(self, json_path):
        with open(json_path, "r") as file:
            data = json.load(file)

        return data

    def get_action_array(self):
        return self.action_array

    def get_duration_array(self):
        return self.duration_array
