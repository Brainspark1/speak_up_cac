"""Resolve GAMEBERT-tagged transcripts into browser commands."""

import logging
import re

from desktop_automation.commands import DesktopCommand
from desktop_automation.gamebert_resolver import GAMEBERTResolver
from desktop_automation.semantic_mapper import SemanticMapper

logger = logging.getLogger(__name__)


class DesktopCommandResolver:
    def __init__(self, config_path, device=-1):
        self.semantic_mapper = SemanticMapper(config_path)
        self.gamebert = GAMEBERTResolver(device=device)

        self.commands = self.semantic_mapper.commands
        self.last_entities = []

    @staticmethod
    def normalize(text):
        return SemanticMapper.normalize(text)

    def tag_entities(self, transcript):
        """Return GAMEBERT entities for a transcript."""
        if not transcript or not transcript.strip():
            return []

        entities = self.gamebert.extract_entities(transcript)
        self.last_entities = entities or []

        logger.info("GAMEBERT entities: %s", self.last_entities)
        return self.last_entities

    @staticmethod
    def _entity_words(entities, group):
        """Collect words belonging to a specific entity group."""
        words = []

        for entity in entities:
            if entity.get("entity_group", "").upper() != group:
                continue

            word = entity.get("word", "").strip()

            if word:
                words.append(word)

        return " ".join(words).strip()

    @staticmethod
    def _clean_target(text):
        """Remove common tokenizer artifacts and filler words."""
        text = text.replace("##", "").strip()
        text = re.sub(r"\s+", " ", text)

        # Remove a trailing generic word, not part of the link name.
        text = re.sub(r"\s+link$", "", text, flags=re.IGNORECASE)

        return text.strip()

    def resolve(self, transcript):
        entities = self.tag_entities(transcript)

        print(f"GAMEBERT entities: {entities}", flush=True)

        if not entities:
            print("No entities found.", flush=True)
            return None

        correction_text = self._entity_words(entities, "CORRECTION")
        if correction_text:
            return DesktopCommand(action="stop")

        action_text = self._entity_words(entities, "ACTION")
        target_text = self._entity_words(entities, "TARGET")

        # GAMEBERT may classify the entire utterance as ACTION.
        # If no TARGET was extracted, try resolving the full transcript.

        # First, try matching the ACTION entity directly.
        if action_text:
            print(f"Text passed to action mapper: {action_text!r}", flush=True)
            action_match = self.semantic_mapper.find_best_action(action_text)

            if action_match is not None:
                action_key, score = action_match
                config = self.commands[action_key]

                if config.get("action") != "click_link":
                    print(
                        f"Matched action: {action_key} (score={score:.3f})",
                        flush=True,
                    )
                    return self._build_command(config)

        # then going to target text
        if not target_text:
            full_text = self._clean_target(transcript).lower()

            for command_key, config in self.commands.items():
                if config.get("action") != "click_link":
                    continue

                action_match = self.semantic_mapper.find_best_action("click")
                if action_match is None:
                    continue

                # Try each configured link against the utterance.
                for link_key in self.semantic_mapper.link_names:
                    link_name = self.semantic_mapper.get_link_display_name(link_key)
                    aliases = [link_key.replace("_", " "), link_name]

                    link_config = self.semantic_mapper.links[link_key]
                    if isinstance(link_config, dict):
                        aliases.extend(
                            link_config.get(
                                "phrases",
                                link_config.get("synonyms", []),
                            )
                        )
                    elif isinstance(link_config, list):
                        aliases.extend(link_config)

                    for alias in aliases:
                        alias_normalized = self.normalize(alias)
                        if not alias_normalized:
                            continue

                        # Match "click Step 2", "click step two", etc.
                        normalized_full = self.normalize(full_text)
                        normalized_alias = self.normalize(alias)

                        if normalized_full.endswith(normalized_alias):
                            prefix = normalized_full[: -len(normalized_alias)].strip()

                            prefix_match = self.semantic_mapper.find_best_action(prefix)

                            if (
                                prefix_match is not None
                                and self.commands[prefix_match[0]].get("action")
                                == "click_link"
                            ):
                                return DesktopCommand(
                                    action="click_link",
                                    link_name=link_name,
                                )

            print(
                f"No target entity found for transcript: {transcript!r}",
                flush=True,
            )
            return None

        print(f"Action text: {action_text!r}", flush=True)
        print(f"Target text: {target_text!r}", flush=True)

        action_match = self.semantic_mapper.find_best_action(action_text)
        if action_match is None:
            return None

        command_key, _ = action_match
        config = self.commands[command_key]

        if config.get("action") == "click_link":
            if not target_text:
                return None

            link_match = self.semantic_mapper.find_best_link(target_text)
            if link_match is None:
                return None

            link_name = self.semantic_mapper.get_link_display_name(link_match[0])
            return DesktopCommand(
                action="click_link",
                link_name=link_name,
            )

        return self._build_command(config)

    @staticmethod
    def _build_command(config):
        action = config.get("action")

        if action == "stop":
            return DesktopCommand(action="stop")

        if action == "scroll":
            return DesktopCommand(
                action="scroll",
                direction=config["direction"],
                amount=config["amount"],
            )

        if action == "zoom":
            return DesktopCommand(
                action="zoom",
                direction=config["direction"],
            )

        logger.warning("Unsupported configured action: %r", action)
        return None
