"""
preprocessing_agent.py — Language-aware text preprocessing.

English: tokenization, lemmatization, stopword removal.
Arabic: diacritics removal, normalization, stemming.
Also performs automatic language detection.
"""
import re
import sys
import logging
from functools import lru_cache
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# NLTK setup
# ──────────────────────────────────────────────

def _ensure_nltk():
    import nltk
    for pkg in ["punkt", "wordnet", "stopwords", "averaged_perceptron_tagger", "omw-1.4", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


# ──────────────────────────────────────────────
# Language detection
# ──────────────────────────────────────────────

@lru_cache(maxsize=1024)
def detect_language(text: str) -> str:
    """
    Detect language of text.
    Returns 'arabic', 'english', or 'unknown'.
    """
    # Fast heuristic: check Arabic Unicode range
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    if arabic_chars / max(len(text), 1) > 0.2:
        return "arabic"

    try:
        from langdetect import detect
        lang = detect(text)
        if lang in ("ar", "fa", "ur"):
            return "arabic"
        return "english"
    except Exception:
        return "english"


# ──────────────────────────────────────────────
# English preprocessing
# ──────────────────────────────────────────────

def preprocess_english(text: str) -> str:
    """Tokenize, lemmatize, remove stopwords (English)."""
    _ensure_nltk()
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords

    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


# ──────────────────────────────────────────────
# Arabic preprocessing
# ──────────────────────────────────────────────

def _remove_diacritics(text: str) -> str:
    """Remove Arabic diacritical marks (tashkeel)."""
    diacritics = re.compile(r"[\u064B-\u065F\u0670]")
    return diacritics.sub("", text)


def _normalize_arabic(text: str) -> str:
    """Normalize common Arabic character variants."""
    # Alef variants
    text = re.sub(r"[أإآاٱ]", "ا", text)
    # Teh Marbuta
    text = re.sub(r"ة", "ه", text)
    # Yeh variants (Ya, Alef Maksura, Hamza on Ya)
    text = re.sub(r"[يىئ]", "ي", text)
    # Waw variants (Waw, Waw with Hamza)
    text = re.sub(r"[ؤو]", "و", text)
    # Remove tatweel (elongation)
    text = re.sub(r"\u0640+", "", text)
    return text


def _arabic_stopwords() -> set:
    return {
        "في", "من", "إلى", "على", "عن", "مع", "هذا", "هذه", "أن", "لا", "ما",
        "هو", "هي", "هم", "نحن", "كان", "كانت", "قد", "لم", "لن", "أو", "و",
        "لكن", "إذا", "حتى", "منذ", "بين", "خلال", "بعد", "قبل", "عند",
    }


def _arabic_light_stem(word: str) -> str:
    """Very simple Arabic light stemmer (prefix/suffix removal)."""
    prefixes = ["ال", "وال", "بال", "فال", "كال", "لل", "وب", "فب", "وف", "في"]
    suffixes = ["ون", "ين", "ات", "ان", "ها", "هم", "هن", "كم", "نا", "تم", "ية"]
    for p in sorted(prefixes, key=len, reverse=True):
        if word.startswith(p) and len(word) - len(p) >= 2:
            word = word[len(p):]
            break
    for s in sorted(suffixes, key=len, reverse=True):
        if word.endswith(s) and len(word) - len(s) >= 2:
            word = word[: -len(s)]
            break
    return word


def preprocess_arabic(text: str) -> str:
    """Remove diacritics, normalize, remove stopwords, light-stem (Arabic)."""
    text = _remove_diacritics(text)
    text = _normalize_arabic(text)
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    stop_words = _arabic_stopwords()
    tokens = text.split()
    tokens = [_arabic_light_stem(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def preprocess(text: str, lang: str = "auto") -> Tuple[str, str]:
    """
    Preprocess text with language-aware pipeline.

    Args:
        text: Raw input text.
        lang: 'english', 'arabic', or 'auto' (detect automatically).

    Returns:
        (processed_text, detected_language)
    """
    if not isinstance(text, str) or not text.strip():
        return "", lang if lang != "auto" else "english"

    if lang == "auto":
        lang = detect_language(text)

    if lang == "arabic":
        return preprocess_arabic(text), "arabic"
    else:
        return preprocess_english(text), "english"


def preprocess_pair(student: str, model: str, lang: str = "auto"):
    """
    Preprocess a student/model answer pair.

    Returns:
        (processed_student, processed_model, detected_language)
    """
    proc_student, detected = preprocess(student, lang)
    proc_model, _ = preprocess(model, detected)   # use same detected lang
    return proc_student, proc_model, detected


if __name__ == "__main__":
    eng = "The impact of Technology on Society is great!  It has changed how we communicate."
    ara = "التكنولوجيا تغيرت الحياة في المجتمع الحديث كثيراً جداً."

    p_eng, l_eng = preprocess(eng)
    print(f"[{l_eng}] {p_eng}")

    p_ara, l_ara = preprocess(ara)
    print(f"[{l_ara}] {p_ara}")
