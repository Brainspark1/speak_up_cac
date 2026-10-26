"""GAMEBERT entity extraction for desktop voice commands."""

import logging

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)

logger = logging.getLogger(__name__)


class GAMEBERTResolver:
    """Load GAMEBERT and extract entities from recognized speech."""

    def __init__(self, model_path="Saggarwal/GAMEBERT", device=-1):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)

        # device=-1 uses CPU; use device=0 for a supported CUDA GPU.
        self.nlp_pipeline = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=device,
        )

    def extract_entities(self, text):
        """Return GAMEBERT's predicted entities and labels."""
        if not text or not text.strip():
            return []

        predictions = self.nlp_pipeline(text.strip())

        logger.debug(
            "GAMEBERT predictions for %r: %s",
            text,
            predictions,
        )

        return predictions
