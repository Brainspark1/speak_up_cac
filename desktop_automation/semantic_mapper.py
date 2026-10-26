"""Match spoken commands and link names against JSON configuration."""

import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticMapper:
    def __init__(self, json_path):
        self.json_path = Path(json_path)

        with self.json_path.open("r", encoding="utf-8") as file:
            self.json_data = json.load(file)

        self.commands = self.json_data.get("commands", {})
        self.links = self.json_data.get("links", {})

        similarity_config = self.json_data.get("similarity", {})
        self.action_threshold = similarity_config.get("action_threshold", 0.12)
        self.link_threshold = similarity_config.get("link_threshold", 0.12)

        self.action_vectorizer = None
        self.action_matrix = None
        self.action_names = []

        self.link_vectorizer = None
        self.link_matrix = None
        self.link_names = []

        self._fit_action_index()
        self._fit_link_index()

    @staticmethod
    def normalize(text):
        """Normalize whitespace and capitalization."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s'-]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _make_document(self, key, entry, include_name=None):
        """Combine a candidate's name, description, and aliases."""
        parts = [key.replace("_", " ")]

        if include_name:
            parts.append(include_name)

        description = entry.get("description", "")
        if description:
            parts.append(description)

        parts.extend(entry.get("phrases", []))

        return self.normalize(" ".join(parts))

    def _fit_action_index(self):
        self.action_names = list(self.commands.keys())

        if not self.action_names:
            return

        documents = [
            self._make_document(name, self.commands[name]) for name in self.action_names
        ]

        self.action_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.action_matrix = self.action_vectorizer.fit_transform(documents)

    def _fit_link_index(self):
        self.link_names = list(self.links.keys())

        if not self.link_names:
            return

        documents = [
            self._make_document(
                key,
                self.links[key],
                include_name=self.links[key].get("name", key),
            )
            for key in self.link_names
        ]

        self.link_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.link_matrix = self.link_vectorizer.fit_transform(documents)

    def _find_best_match(
        self,
        transcript,
        names,
        vectorizer,
        document_matrix,
        threshold,
    ):
        """Return the best candidate and score, or None if too weak."""
        transcript = self.normalize(transcript)

        if not transcript or not names or vectorizer is None or document_matrix is None:
            return None

        query_vector = vectorizer.transform([transcript])

        if query_vector.nnz == 0:
            return None

        scores = cosine_similarity(
            query_vector,
            document_matrix,
        )[0]

        best_index = scores.argmax()
        best_score = float(scores[best_index])

        if best_score < threshold:
            return None

        return names[best_index], best_score

    def find_best_action(self, transcript):
        """Match tagged action text against configured actions."""
        normalized = self.normalize(transcript)

        for key, command in self.commands.items():
            aliases = [key.replace("_", " ")]
            aliases.extend(command.get("phrases", []))

            if any(normalized == self.normalize(alias) for alias in aliases):
                return key, 1.0

        return self._find_best_match(
            transcript,
            self.action_names,
            self.action_vectorizer,
            self.action_matrix,
            self.action_threshold,
        )

    def find_best_link(self, transcript):
        """Match tagged target text against configured links."""
        normalized = self.normalize(transcript)

        for key, link in self.links.items():
            aliases = [
                key.replace("_", " "),
                link.get("name", key),
            ]
            aliases.extend(link.get("phrases", []))

            if any(normalized == self.normalize(alias) for alias in aliases):
                return key, 1.0

        return self._find_best_match(
            transcript,
            self.link_names,
            self.link_vectorizer,
            self.link_matrix,
            self.link_threshold,
        )

    def get_link_display_name(self, link_key):
        """Return the canonical visible link name from JSON."""
        link = self.links.get(link_key)

        if not link:
            return None

        return link.get("name", link_key)
