import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticMapper:
    def __init__(self, json_path):
        self.json_data = self.read_json_file(json_path)

    def read_json_file(self, json_path):
        with open(json_path, "r") as file:
            data = json.load(file)

        return data

    def find_max_target_similarity(self, transcript_sentence):
        # loading data into python dictionary
        targets_data = self.json_data["targets"]

        target_names = [name for name in targets_data.keys() if name != "enemy"]
        descriptions = [targets_data[name]["description"] for name in target_names]

        # combining to be vectorized
        all_documents = [transcript_sentence] + descriptions

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_documents)

        target_vector = tfidf_matrix[0:1]
        description_vectors = tfidf_matrix[1:]
        similarity_scores = cosine_similarity(target_vector, description_vectors)[
            0
        ]  # returns a list wrapped inside another, [0] removes the outer list and only extracts the inner one

        max_idx = np.argmax(similarity_scores)
        max_score = similarity_scores[max_idx]
        max_target_name = target_names[max_idx]

        return max_target_name, max_score

    def find_max_action_similarity(self, transcript_sentence):
        # loading data into python dictionary
        actions_data = self.json_data["actions"]

        action_names = [action for action in actions_data.keys()]
        action_descriptions = [
            actions_data[action]["description"] for action in action_names
        ]

        # combining to be vectorized
        all_documents = [transcript_sentence] + action_descriptions

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_documents)

        target_vector = tfidf_matrix[0:1]
        description_vectors = tfidf_matrix[1:]
        similarity_scores = cosine_similarity(target_vector, description_vectors)[0]

        max_idx = np.argmax(similarity_scores)
        max_score = similarity_scores[max_idx]
        max_action_name = action_names[max_idx]

        return max_action_name, max_score
