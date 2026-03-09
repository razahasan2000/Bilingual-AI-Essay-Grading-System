import sys
import os
import numpy as np
import pickle
from pathlib import Path

# Add root to path to import agents and config
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config
from agents.scoring_agent import build_scoring_model
from agents.feature_agent import build_feature_matrix

class ScoringWrapper:
    def __init__(self, checkpoint_path=None):
        if checkpoint_path is None:
            # Use the verified 0.91 QWK model
            checkpoint_path = os.path.join(root_dir, "checkpoints", "best_model_exp_20260307_224545_e487fada.pkl")
            
        self.model_type = config.SCORING_MODEL
        print(f"Loading Scoring Model: {self.model_type} from {checkpoint_path}")
        
        # SVR pipeline handles its own dimensions, but build_scoring_model needs the type
        self.model = build_scoring_model(self.model_type, input_dim=1550) 
        
        try:
            self.model.load(checkpoint_path)
            print(f"Model loaded successfully from {checkpoint_path}")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load checkpoint from {checkpoint_path}")
            print(f"Error details: {e}")
            raise e

    def score_essay(self, text, lang="english"):
        # Placeholder model answer (Reference)
        # In a real system, this would come from the prompt database.
        if lang == "arabic":
            model_answer = "هذا نص نموذجي للإجابة المثالية التي تغطي كافة جوانب الموضوع بشكل دقيق ولغة رصينة."
        else:
            model_answer = "This is a model answer that represents high-quality writing with clear structure and sophisticated vocabulary."

        # Build feature matrix (returns features and cosine sims)
        features, _ = build_feature_matrix(
            student_texts=[text],
            model_texts=[model_answer],
            lang=lang
        )

        # Predict score (0–1 scale)
        # Some models take prompt_ids, we use None or 0
        score_norm = self.model.predict(features, prompt_ids=[0])[0]
        
        # Map 0-1 to 0-10 scale
        final_score = score_norm * 10.0
        return final_score

def score_essay(text, lang="english"):
    wrapper = ScoringWrapper()
    return wrapper.score_essay(text, lang)
