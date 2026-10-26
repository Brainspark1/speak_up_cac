"""Loads game_config.json and derives the voice-speak-up mapping JSON from it.

game_config.json is the single file you edit. The library only understands
{"targets": {...}, "actions": {...}}, so we generate that file on startup.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "game_config.json"
GENERATED_MAPPING_PATH = Path(__file__).resolve().parent / ".generated_mapping.json"

RESERVED_IDS = {"enemy"}  # SemanticMapper always skips "enemy"


def load_config(path=DEFAULT_CONFIG_PATH):
    with open(path, "r") as f:
        config = json.load(f)
    validate_config(config)
    return config


def validate_config(config):
    moles = config.get("moles", {})
    if not moles:
        raise ValueError("game_config.json has no moles")
    for mole_id, mole in moles.items():
        if mole_id in RESERVED_IDS:
            raise ValueError(f"Mole id '{mole_id}' is reserved by SemanticMapper and would never match")
        if not mole.get("description", "").strip():
            raise ValueError(f"Mole '{mole_id}' has an empty description")
    for action_id, action in config.get("actions", {}).items():
        if not action.get("description", "").strip():
            raise ValueError(f"Action '{action_id}' has an empty description")
    for pair in config.get("twins", []):
        for mole_id in pair:
            if mole_id not in moles:
                raise ValueError(f"Twin '{mole_id}' is not a defined mole")


def build_mapping(config):
    """The voice-speak-up mapping dict: every mole as a target, plus actions.

    Extra keys (family, tags, ...) are allowed by the library and ignored.
    """
    return {
        "targets": {mole_id: dict(mole) for mole_id, mole in config["moles"].items()},
        "actions": {action_id: dict(action) for action_id, action in config["actions"].items()},
    }


def write_mapping(config, path=GENERATED_MAPPING_PATH):
    path = Path(path)
    with open(path, "w") as f:
        json.dump(build_mapping(config), f, indent=2)
    return path
