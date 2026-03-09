"""
feature_agent.py — Transformer embedding feature extraction.

Encodes student and model answers using sentence-transformers.
Computes cosine similarity as a key semantic feature.
"""
import sys
import logging
import re
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import pandas as pd
import collections

log = logging.getLogger(__name__)

_encoder = None
_encoder_name = None


def _get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_encoder(model_name: Optional[str] = None, lang: str = "english"):
    """Load sentence-transformer encoder (cached)."""
    global _encoder, _encoder_name

    if model_name is None:
        if lang.lower() == "arabic":
            model_name = config.ARABIC_ENCODER
        else:
            model_name = config.ENGLISH_ENCODER

    if _encoder is not None and _encoder_name == model_name:
        return _encoder

    log.info(f"Loading encoder: {model_name}")
    try:
        from sentence_transformers import SentenceTransformer
        device = _get_device()
        _encoder = SentenceTransformer(model_name, device=device)
        _encoder_name = model_name
        log.info(f"Encoder loaded on {device}")
    except Exception as e:
        log.error(f"Failed to load sentence-transformer: {e}")
        raise
    return _encoder


def encode(texts: List[str], model_name: Optional[str] = None,
           lang: str = "english", batch_size: int = 64) -> np.ndarray:
    """
    Encode a list of texts into embedding vectors.

    Returns:
        numpy array of shape (N, embedding_dim)
    """
    encoder = load_encoder(model_name, lang)
    embeddings = encoder.encode(texts, batch_size=batch_size,
                                show_progress_bar=len(texts) > 100,
                                convert_to_numpy=True)
    return embeddings


def cosine_similarity_pairwise(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Compute element-wise cosine similarity between two embedding matrices."""
    norm_a = np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-10
    norm_b = np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-10
    return np.sum((emb_a / norm_a) * (emb_b / norm_b), axis=1)


from agents.novelty_agent import NoveltyAgent

def _extract_structural_entropy(text: str) -> float:
    """Measures variance/entropy in sentence structure."""
    sents = re.split(r'[.!?؟]', text)
    lengths = [len(s.split()) for s in sents if s.strip()]
    if len(lengths) < 2: return 0.0
    return float(np.std(lengths) / (np.mean(lengths) + 1e-10))

def _extract_rich_features(text: str, lang: str = "english") -> np.ndarray:
    """Extract deep linguistic complexity features."""
    words = text.split()
    if not words:
        return np.zeros((1, 9))
    
    char_count = len(text)
    word_count = len(words)
    avg_word_len = char_count / (word_count + 1e-10)
    
    # Word length variance
    word_lens = [len(w) for w in words]
    var_word_len = np.var(word_lens) if len(word_lens) > 1 else 0.0
    
    # Sentence context
    sents = re.split(r'[.!?؟]', text)
    sent_count = len([s for s in sents if s.strip()])
    avg_sent_len = word_count / (sent_count + 1e-10)
    
    # Lexical Diversity
    unique_words = set(words)
    lexical_density = len(unique_words) / (word_count + 1e-10)
    
    # Punctuation density
    punc_matches = re.findall(r'[,;:()\[\]ـ«»]', text)
    punc_density = len(punc_matches) / (word_count + 1e-10)

    # Arabic Specific: Stem Ratio (simplified)
    stem_ratio = 0.0
    if lang == "arabic":
        try:
            import pyarabic.araby as araby
            stems = set([araby.strip_tashkeel(w)[:4] for w in words]) # Crude stemmer
            stem_ratio = len(stems) / (word_count + 1e-10)
        except: pass

    struct_entropy = _extract_structural_entropy(text)

    return np.array([
        avg_word_len, var_word_len, float(word_count), float(sent_count),
        avg_sent_len, lexical_density, punc_density, stem_ratio, struct_entropy
    ]).reshape(1, -1)


class FeatureAgent:
    """Manages one or more transformer encoders."""
    def __init__(self, model_name1: str, model_name2: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
            device = _get_device()
            log.info(f"Loading primary encoder: {model_name1} on {device}")
            self.model1 = SentenceTransformer(model_name1, device=device)
            self.model2 = None
            if model_name2:
                log.info(f"Loading secondary encoder: {model_name2} on {device}")
                self.model2 = SentenceTransformer(model_name2, device=device)
        except Exception as e:
            log.error(f"Failed to load sentence-transformers: {e}")
            raise

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        embs1 = self.model1.encode(texts, batch_size=batch_size, 
                                   show_progress_bar=len(texts) > 200, 
                                   convert_to_numpy=True)
        if self.model2:
            embs2 = self.model2.encode(texts, batch_size=batch_size, 
                                       show_progress_bar=len(texts) > 200, 
                                       convert_to_numpy=True)
            return np.concatenate([embs1, embs2], axis=1)
        return embs1


def build_feature_matrix(
    student_texts: List[str],
    model_texts: List[str],
    essay_sets: Optional[List[int]] = None,
    lang: str = "english",
    encoder_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix for scoring.

    Returns:
        features: np.ndarray of shape (N, 2*embed_dim + 8)
            Concatenation of [student_emb, model_emb, cos_sim, len_feats, novelty, morph_feats]
        cos_sims: np.ndarray of shape (N,) — raw cosine similarities
    """
    # 1. Initialize Encoders
    m1 = encoder_name or (config.ARABIC_ENCODER if lang == "arabic" else config.ENGLISH_ENCODER)
    m2 = config.ARABIC_ENCODER_2 if lang == "arabic" else None
    
    agent = FeatureAgent(m1, m2)
    
    log.info(f"Encoding {len(student_texts)} student answers …")
    student_embs = agent.encode(student_texts)

    log.info(f"Encoding {len(model_texts)} model answers …")
    model_embs = agent.encode(model_texts)

    cos_sims = cosine_similarity_pairwise(student_embs, model_embs).reshape(-1, 1)

    # Base features
    student_lengths = np.array([len(t.split()) for t in student_texts]).reshape(-1, 1).astype(float)
    model_lengths = np.array([len(t.split()) for t in model_texts]).reshape(-1, 1).astype(float)
    # Normalize lengths
    student_lengths /= (student_lengths.max() + 1e-10)
    model_lengths /= (model_lengths.max() + 1e-10)

    # --- Novelty Extraction ---
    novelty_agent = NoveltyAgent()
    
    # Lexical features
    lexical_features = [novelty_agent.get_lexical_novelty(t) for t in student_texts]
    ttrs = np.array([f["ttr"] for f in lexical_features]).reshape(-1, 1)
    
    # Semantic features (Prompt-relative)
    if essay_sets is not None:
        dummy_df = pd.DataFrame({"essay_set": essay_sets})
        novelty_agent.compute_prompt_centroids(dummy_df, student_embs)
        sem_novelties = np.array([
            novelty_agent.get_semantic_novelty(es, emb) 
            for es, emb in zip(essay_sets, student_embs)
        ]).reshape(-1, 1)
    else:
        sem_novelties = np.zeros((len(student_texts), 1))

    # Normalize novelty features
    ttrs /= (ttrs.max() + 1e-10)
    sem_novelties /= (sem_novelties.max() + 1e-10)

    # Rich linguistic features
    rich_feats = np.vstack([_extract_rich_features(t, lang) for t in student_texts])
    # Normalize
    rich_feats /= (rich_feats.max(axis=0) + 1e-10)

    features = np.concatenate(
        [student_embs, model_embs, cos_sims, student_lengths, model_lengths, ttrs, sem_novelties, rich_feats],
        axis=1
    )
    return features, cos_sims.flatten()


if __name__ == "__main__":
    texts_a = ["Technology has changed society greatly.", "Education is very important."]
    texts_b = ["Technology impacts modern society significantly.", "Learning shapes individuals."]
    feats, sims = build_feature_matrix(texts_a, texts_b, lang="english")
    print(f"Feature matrix shape: {feats.shape}")
    print(f"Cosine similarities: {sims}")
