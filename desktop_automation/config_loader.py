"""Load and validate Speak Up command configuration."""

import json
from pathlib import Path


class CommandConfig:
    def __init__(self, config_path=None):
        project_root = Path(__file__).resolve().parent.parent

        if config_path is None:
            config_path = Path(__file__).resolve().parent / "config" / "commands.json"

        self.config_path = Path(config_path)
        self.data = self._load()

    def _load(self):
        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"Command configuration not found: {self.config_path}"
            )

        with self.config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data.get("commands"), dict):
            raise ValueError("Configuration must contain a 'commands' object.")

        for name, command in data["commands"].items():
            if not isinstance(command, dict):
                raise ValueError(f"Command '{name}' must be a JSON object.")

            if not isinstance(command.get("action"), str):
                raise ValueError(f"Command '{name}' must define an action.")

            phrases = command.get("phrases", [])

            if not isinstance(phrases, list) or not all(
                isinstance(phrase, str) and phrase.strip() for phrase in phrases
            ):
                raise ValueError(f"Command '{name}' has invalid phrases.")

        links = data.get("links", {})
        if not isinstance(links, dict):
            raise ValueError("Configuration 'links' must be an object.")

        for name, link in links.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Link names must be non-empty strings.")

            # Support the original format: "link_key": ["synonym", ...]
            if isinstance(link, list):
                if not all(
                    isinstance(synonym, str) and synonym.strip() for synonym in link
                ):
                    raise ValueError(
                        f"Link '{name}' must have non-empty synonym strings."
                    )
                continue

            # Support the richer format with name, description, and phrases.
            if not isinstance(link, dict):
                raise ValueError(f"Link '{name}' must be a list or a JSON object.")

            for field in ("name", "description"):
                value = link.get(field)
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"Link '{name}' has an invalid '{field}' field.")

            phrases = link.get("phrases", link.get("synonyms", []))
            if not isinstance(phrases, list) or not all(
                isinstance(phrase, str) and phrase.strip() for phrase in phrases
            ):
                raise ValueError(
                    f"Link '{name}' must have a list of non-empty phrases."
                )

        return data

    @property
    def commands(self):
        return self.data["commands"]

    def get(self, name):
        return self.commands.get(name)

    @property
    def links(self):
        return self.data.get("links", {})

    def small_scroll_length(self):
        return self.data["constants"]["scroll_distances"]["small"]

    def page_scroll_length(self):
        return self.data["constants"]["scroll_distances"]["page"]

    def get_zoom_values(self):
        return self.data["constants"]["zoom"]["zoom_levels"]

    def get_default_zoom(self):
        return self.data["constants"]["zoom"]["default_zoom"]
