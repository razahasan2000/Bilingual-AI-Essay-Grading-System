"""
augmentation_agent.py — Data augmentation for Arabic/English essays.
Helps the model generalize with small datasets.
"""
import random
import re
import logging
from typing import List, Tuple

log = logging.getLogger(__name__)

def augment_text(text: str, lang: str = "arabic", factor: float = 0.2) -> str:
    """
    Applies random augmentation to a single text.
    Strategies: Sentence Shuffling, Word Deletion, Word Swap.
    """
    if not text or len(text) < 10:
        return text

    # Split into sentences
    sentences = re.split(r'([.!?؟])', text)
    # Reconstruct sentences with their punctuation
    s_list = []
    for i in range(0, len(sentences)-1, 2):
        s_list.append(sentences[i] + sentences[i+1])
    if len(sentences) % 2 == 1:
        s_list.append(sentences[-1])

    # 1. Sentence Shuffling (if enough sentences)
    if len(s_list) > 3 and random.random() < factor:
        random.shuffle(s_list)
        return "".join(s_list).strip()

    # 2. Word Level operations
    words = text.split()
    if len(words) < 5:
        return text

    op = random.random()
    if op < 0.3:
        # Random Deletion
        idx = random.randint(0, len(words) - 1)
        words.pop(idx)
    elif op < 0.6:
        # Random Swap
        i1, i2 = random.sample(range(len(words)), 2)
        words[i1], words[i2] = words[i2], words[i1]
    
    # 3. Arabic Specific: Random Tashkeel Removal (if present) or insertion
    if lang == "arabic" and random.random() < factor:
        import pyarabic.araby as araby
        text = araby.strip_tashkeel(text)
        return text

    return " ".join(words)

def augment_batch(texts: List[str], scores: List[float], lang: str = "arabic", multiplier: int = 2) -> Tuple[List[str], List[float]]:
    """
    Increases dataset size by generating augmented versions of existing essays.
    """
    aug_texts = list(texts)
    aug_scores = list(scores)
    
    for _ in range(multiplier - 1):
        for t, s in zip(texts, scores):
            aug_texts.append(augment_text(t, lang=lang))
            aug_scores.append(s)
            
    log.info(f"Augmented dataset from {len(texts)} to {len(aug_texts)} samples.")
    return aug_texts, aug_scores
