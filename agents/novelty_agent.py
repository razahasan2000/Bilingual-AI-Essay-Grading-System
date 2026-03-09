import numpy as np
import pandas as pd
import logging
from typing import List, Dict
import collections
import re

log = logging.getLogger(__name__)

class NoveltyAgent:
    """
    Agent responsible for calculating Lexical and Semantic novelty features.
    Lexical: Type-Token Ratio (TTR) and Rare Vocabulary.
    Semantic: Distance from prompt-level centroid.
    """
    def __init__(self):
        self.rare_word_threshold = 0.05  # Words appearing in < 5% of essays are 'rare'
        self.prompt_centroids = {} # essay_set -> centroid_embedding

    def get_lexical_novelty(self, text: str, corpus_tokens: List[List[str]] = None) -> Dict[str, float]:
        """
        Calculate Lexical Novelty metrics.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return {"ttr": 0.0, "lexical_richness": 0.0}

        # Type-Token Ratio (TTR)
        ttr = len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0.0
        
        return {
            "ttr": ttr,
            "token_count": float(len(tokens))
        }

    def compute_prompt_centroids(self, df: pd.DataFrame, embeddings: np.ndarray):
        """
        Compute the average embedding (centroid) for each prompt/essay_set.
        """
        self.prompt_centroids = {}
        unique_sets = df["essay_set"].unique()
        for essay_set in unique_sets:
            indices = df[df["essay_set"] == essay_set].index
            set_embeddings = embeddings[indices]
            self.prompt_centroids[essay_set] = np.mean(set_embeddings, axis=0)
        log.info(f"Computed centroids for {len(self.prompt_centroids)} essay sets.")

    def get_semantic_novelty(self, essay_set: int, embedding: np.ndarray) -> float:
        """
        Distance from the prompt centroid. High distance = high novelty.
        """
        centroid = self.prompt_centroids.get(essay_set)
        if centroid is None:
            return 0.0
        
        # Euclidean distance
        distance = np.linalg.norm(embedding - centroid)
        return float(distance)

    def _tokenize(self, text: str) -> List[str]:
        # Simple whitespace tokenization after removing non-alphanumeric
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        return text.split()
