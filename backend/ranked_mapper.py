"""RankedMapper: SemanticMapper that returns full rankings instead of only the top hit.

Uses the same scoring as voice-speak-up v1.0.1: a fresh TfidfVectorizer fit on
[text] + descriptions, then cosine similarity between the text and each
description. The only difference is that every score is returned, so the
resolver can compare top1 with top2.
"""

import importlib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SemanticMapper = importlib.import_module("speak-up").SemanticMapper


class RankedMapper(SemanticMapper):
    def __init__(self, json_path=None, json_data=None):
        # json_data lets tests and the server build a mapper without a file.
        if json_data is not None:
            self.json_data = json_data
        else:
            super().__init__(json_path)

    @staticmethod
    def _rank(text, names, descriptions):
        if not names:
            return []
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([text] + descriptions)
        except ValueError:
            # "empty vocabulary": nothing tokenizable anywhere
            return [(name, 0.0) for name in names]
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        ranked = sorted(zip(names, (float(s) for s in scores)), key=lambda p: p[1], reverse=True)
        return ranked

    def rank_targets(self, text, target_ids=None):
        """[(target_id, score), ...] sorted descending, restricted to target_ids.

        "enemy" is skipped, the same as in find_max_target_similarity.
        """
        targets = self.json_data["targets"]
        names = [n for n in (target_ids if target_ids is not None else targets.keys())
                 if n in targets and n != "enemy"]
        return self._rank(text, names, [targets[n]["description"] for n in names])

    def rank_actions(self, text):
        actions = self.json_data["actions"]
        names = list(actions.keys())
        return self._rank(text, names, [actions[n]["description"] for n in names])
