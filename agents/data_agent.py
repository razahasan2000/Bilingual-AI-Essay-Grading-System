"""
data_agent.py — Dataset acquisition and preparation.

Supports:
  English:
    • ASAP (Kaggle kaggle competitions download -c asap-aes)
    • Falls back to high-quality synthetic data

  Arabic:
    1. AR-AES    — HuggingFace: Rnghazawi-NLP/AR-AES
                   2,046 essays, Question_id 1–12, dual-rater rubric scores,
                   separate typical-answers (model answer) file.
    2. GLUPS-ASAG — GitHub ODS file: FSTT-LIST/GLUPS-ASAG-Dataset
                    Arabic short-answer grading, Islamic Education.
    3. MBZUAI arabic-aes-bea25 — GitHub: mbzuai-nlp/arabic-aes-bea25
                    topics.csv + prompts.csv + essays with GPT-4o generations.

Datasets are cached as CSV in data/arabic/ and data/english/.
"""
import os
import sys
import csv
import json
import random
import logging
import zipfile
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="[DataAgent] %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════
# ENGLISH — ASAP
# ══════════════════════════════════════════════════════════

def _try_kaggle_download() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        log.warning("~/.kaggle/kaggle.json not found — skipping Kaggle download.")
        return False
    try:
        dest = Path(config.ENGLISH_DATA_DIR) / "asap_raw"
        dest.mkdir(parents=True, exist_ok=True)
        log.info("Attempting Kaggle ASAP download …")
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", "asap-aes", "-p", str(dest)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            for z in dest.glob("*.zip"):
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(dest)
            log.info("Kaggle ASAP download successful.")
            return True
        log.warning(f"Kaggle error: {result.stderr.strip()}")
        return False
    except Exception as e:
        log.warning(f"Kaggle download failed: {e}")
        return False


def _load_asap_from_disk() -> "pd.DataFrame | None":
    raw_dir = Path(config.ENGLISH_DATA_DIR) / "asap_raw"
    tsv_files = list(raw_dir.glob("*.tsv")) if raw_dir.exists() else []
    if not tsv_files:
        return None
    dfs = []
    for f in tsv_files:
        try:
            df = pd.read_csv(f, sep="\t", encoding="utf-8", on_bad_lines="skip")
            dfs.append(df)
        except Exception as e:
            log.warning(f"Could not read {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else None


_ASAP_MODEL_ANSWERS = {
    1: "Technology has both positive and negative effects on society, impacting communication, education, and economy.",
    2: "The principal's message emphasizes dedication, positive attitude, and community involvement in achieving success.",
    3: "The author uses figurative language and vivid descriptions to convey mood and setting.",
    4: "The excerpt shows how perseverance and hard work lead to personal growth and achievement.",
    5: "The story illustrates themes of courage, friendship, and overcoming adversity.",
    6: "The author argues that censorship is harmful to democracy and free expression.",
    7: "The narrative explores the relationship between memory and identity through personal experience.",
    8: "The informational text explains the importance of volunteering and community service.",
}
_ASAP_MAX_SCORES = {1: 12, 2: 6, 3: 3, 4: 3, 5: 4, 6: 4, 7: 30, 8: 60}


def _parse_asap_df(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "essay_id": "essay_id", "essay_set": "essay_set",
        "essay": "student_answer", "domain1_score": "score",
        "rater1_domain1": "score",
    }
    rename = {c: col_map[c] for c in df.columns if c in col_map}
    df = df.rename(columns=rename)
    needed = ["essay_id", "essay_set", "student_answer", "score"]
    for col in needed:
        if col not in df.columns:
            df[col] = None
    df = df[needed].dropna(subset=["student_answer", "score"])
    df["model_answer"] = df["essay_set"].map(
        lambda s: _ASAP_MODEL_ANSWERS.get(int(s), "Model answer placeholder.") if pd.notna(s) else ""
    )
    def _normalize(row):
        m = _ASAP_MAX_SCORES.get(int(row["essay_set"]), 10) if pd.notna(row["essay_set"]) else 10
        return float(row["score"]) / m
    df["score_normalized"] = df.apply(_normalize, axis=1)
    df["language"] = "english"
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════
# ARABIC DATASET 1 — AR-AES (HuggingFace)
# ══════════════════════════════════════════════════════════

def _load_araes_hf() -> "pd.DataFrame | None":
    """
    Load AR-AES from HuggingFace using the datasets library.
    Dataset: Rnghazawi-NLP/AR-AES
    """
    try:
        from datasets import load_dataset
        log.info("Loading AR-AES from HuggingFace ...")
        # Specifying the exact file to avoid column mismatch with rubrics/typical answers
        ds = load_dataset("Rnghazawi-NLP/AR-AES", data_files="ARAES Dataset - Essays and Marks.csv")
        
        # Determine the best split
        split = "train" if "train" in ds.keys() else list(ds.keys())[0]
        df = ds[split].to_pandas()
        df.columns = [c.strip().lower() for c in df.columns]

        # Map AR-AES columns
        text_col = next((c for c in df.columns if any(k in c for k in ["text", "essay", "answer"])), None)
        score_col = next((c for c in df.columns if any(k in c for k in ["final", "score", "grade"])), None)
        q_col = next((c for c in df.columns if "question" in c or "topic" in c), None)

        if text_col is None or score_col is None:
            log.warning(f"AR-AES: mapped columns fail {list(df.columns)}")
            return None

        # Build clean DF
        clean_df = pd.DataFrame({
            "student_answer": df[text_col].astype(str),
            "score": pd.to_numeric(df[score_col], errors="coerce"),
            "essay_set": df[q_col] if q_col else 1
        })
        clean_df = clean_df.dropna(subset=["student_answer", "score"])
        
        # ── Join Real Typical Answers ──
        try:
            ta_ds = load_dataset("Rnghazawi-NLP/AR-AES", data_files="ARAES Dataset - Typical Answers.csv")
            ta_split = "train" if "train" in ta_ds.keys() else list(ta_ds.keys())[0]
            ta_df = ta_ds[ta_split].to_pandas()
            ta_df.columns = [c.strip().lower() for c in ta_df.columns]
            
            # Identify columns
            ta_q_col = next((c for c in ta_df.columns if "question" in c or "id" in c), None)
            ta_text_col = next((c for c in ta_df.columns if any(k in c for k in ["answer", "typical", "model"])), None)
            
            if ta_q_col and ta_text_col:
                ta_map = dict(zip(ta_df[ta_q_col], ta_df[ta_text_col].astype(str)))
                clean_df["model_answer"] = clean_df["essay_set"].map(lambda q: ta_map.get(q, "نموذج الإجابة للسؤال."))
                log.info(f"AR-AES: Joined {len(ta_map)} real model answers.")
        except Exception as te:
            log.warning(f"AR-AES: Could not load typical answers: {te}")

        clean_df["language"] = "arabic"
        clean_df["source"] = "AR-AES"
        
        max_s = clean_df["score"].max()
        clean_df["score_normalized"] = clean_df["score"] / (max_s if max_s > 0 else 1)
        clean_df["essay_id"] = range(len(clean_df))
        
        log.info(f"AR-AES loaded: {len(clean_df)} rows")
        return clean_df.reset_index(drop=True)
    except Exception as e:
        log.warning(f"AR-AES HuggingFace load failed: {e}")
        return None

    except Exception as e:
        log.warning(f"AR-AES HuggingFace load failed: {e}")
        return None


# ══════════════════════════════════════════════════════════
# ARABIC DATASET 2 — GLUPS-ASAG (GitHub ODS)
# ══════════════════════════════════════════════════════════

def _load_glups() -> "pd.DataFrame | None":
    """
    Download and parse GLUPS-ASAG ODS file from GitHub.
    Islamic Education Arabic short-answer grading dataset.
    """
    import requests
    url = (
        "https://raw.githubusercontent.com/FSTT-LIST/GLUPS-ASAG-Dataset/main/"
        "Dataset_Islamic_Education_1/ASAG-GLUPS-dataset1.2023%2004.public.1.ods"
    )
    dest = Path(config.ARABIC_DATA_DIR) / "glups_raw.ods"
    if not dest.exists():
        log.info("Downloading GLUPS-ASAG ODS …")
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            log.info(f"GLUPS saved to {dest}")
        except Exception as e:
            log.warning(f"GLUPS download failed: {e}")
            return None

    try:
        df = pd.read_excel(dest, engine="odf")
        # Remove empty unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Heuristic column mapping with fuzzy matching
        student_col = next((c for c in df.columns if any(k in c for k in ["student", "answer", "إجابة", "original_essay", "essay"])), None)
        model_col   = next((c for c in df.columns if any(k in c for k in ["model", "reference", "نموذج", "الإجابة النموذجية", "model_answer"])), None)
        score_col   = next((c for c in df.columns if any(k in c for k in ["score", "grade", "mark", "درجة", "العلامة", "final_score", "marks"])), None)

        if student_col is None or score_col is None:
            # Try to guess by index
            if len(df.columns) >= 3:
                student_col = df.columns[1] if "id" in df.columns[0].lower() else df.columns[0]
                score_col   = df.columns[-1]

        if student_col is None or score_col is None:
            log.warning(f"GLUPS: cannot identify columns in {list(df.columns)}")
            return None

        # Build clean DF
        clean_df = pd.DataFrame({
            "student_answer": df[student_col].astype(str),
            "score": pd.to_numeric(df[score_col], errors="coerce"),
            "model_answer": df[model_col].astype(str) if model_col else "إجابة مرجعية."
        })
        clean_df = clean_df.dropna(subset=["student_answer", "score"])
        
        max_s = clean_df["score"].max()
        clean_df["score_normalized"] = clean_df["score"] / (max_s if max_s > 0 else 1)
        clean_df["essay_id"] = range(len(clean_df))
        clean_df["essay_set"] = 1
        clean_df["language"] = "arabic"
        clean_df["source"] = "GLUPS-ASAG"

        log.info(f"GLUPS-ASAG loaded: {len(clean_df)} rows")
        return clean_df.reset_index(drop=True)

    except Exception as e:
        log.warning(f"GLUPS-ASAG parse failed: {e}")
        return None


# ══════════════════════════════════════════════════════════
# ARABIC DATASET 3 — MBZUAI arabic-aes-bea25 (GitHub)
# ══════════════════════════════════════════════════════════

def _load_mbzuai_arabic_aes() -> "pd.DataFrame | None":
    """
    Load MBZUAI arabic-aes-bea25 dataset from GitHub.
    Structure: topics.csv + prompts.csv + essays (scored Arabic text)
    Essay ID format: AR_{language_model}_{topic_id}{prompt_id}_{essay_id}
    """
    import requests
    base_url = "https://raw.githubusercontent.com/mbzuai-nlp/arabic-aes-bea25/main/"
    dest_dir = Path(config.ARABIC_DATA_DIR) / "mbzuai_raw"
    dest_dir.mkdir(exist_ok=True)

    files_to_try = [
        "Data/arwi_original_essays.csv",
        "Data/prompts.csv",
        "Data/topics.csv",
        "arwi_train.csv",
        "arwi_test.csv"
    ]

    def _fetch(fname: str) -> "pd.DataFrame | None":
        local_name = fname.replace("/", "_")
        local = dest_dir / local_name
        if local.exists():
            return pd.read_csv(local, encoding="utf-8")
        try:
            r = requests.get(base_url + fname, timeout=30)
            if r.status_code == 200:
                local.write_bytes(r.content)
                return pd.read_csv(local, encoding="utf-8")
        except Exception:
            pass
        return None

    essay_df = _fetch("Data/arwi_original_essays.csv")
    if essay_df is None:
        essay_df = _fetch("Data/Splits/arwi_train.csv")

    prompts_df = _fetch("Data/prompts.csv")

    if essay_df is None:
        log.warning("MBZUAI arabic-aes-bea25: could not find essay CSV.")
        return None

    essay_df.columns = [c.strip().lower() for c in essay_df.columns]

    # Map column names for ARWI format
    # columns usually include: id, essay, score_1, score_2, final_score, prompt_id
    text_col  = next((c for c in essay_df.columns
                      if any(k in c for k in ["essay", "text", "answer", "content"])), None)
    score_col = next((c for c in essay_df.columns
                      if any(k in c for k in ["final_score", "score", "grade"])), None)
    prompt_col = next((c for c in essay_df.columns
                       if any(k in c for k in ["prompt", "topic", "question"])), None)

    if text_col is None or score_col is None:
        log.warning(f"MBZUAI: unexpected columns {list(essay_df.columns)}")
        return None

    essay_df = essay_df.rename(columns={text_col: "student_answer", score_col: "score"})

    # Attach prompt text
    if prompts_df is not None and prompt_col:
        prompts_df.columns = [c.strip().lower() for c in prompts_df.columns]
        # Prompts usually have prompt_id, prompt_text
        pid_col = prompts_df.columns[0]
        ptext_col = next((c for c in prompts_df.columns if "text" in c or "prompt" in c), prompts_df.columns[1])
        prompt_map = dict(zip(prompts_df[pid_col], prompts_df[ptext_col]))
        essay_df["model_answer"] = essay_df[prompt_col].map(
            lambda q: prompt_map.get(q, "النص المرجعي للسؤال."))
    else:
        essay_df["model_answer"] = "النص المرجعي للسؤال."

    essay_df = essay_df[essay_df["student_answer"].notna() & essay_df["score"].notna()].copy()
    essay_df["score"] = pd.to_numeric(essay_df["score"], errors="coerce")
    essay_df = essay_df.dropna(subset=["score"])
    max_score = essay_df["score"].max()
    essay_df["score_normalized"] = essay_df["score"] / (max_score if max_score > 0 else 1)
    essay_df["essay_id"] = range(len(essay_df))
    essay_df["essay_set"] = 1
    essay_df["language"] = "arabic"
    essay_df["source"] = "MBZUAI-arabic-aes"

    log.info(f"MBZUAI arabic-aes-bea25 loaded: {len(essay_df)} rows")
    return essay_df[["essay_id", "essay_set", "student_answer", "model_answer",
                     "score", "score_normalized", "language", "source"]].reset_index(drop=True)


# ══════════════════════════════════════════════════════════
# SYNTHETIC FALLBACKS
# ══════════════════════════════════════════════════════════

ENGLISH_PROMPTS = [
    {
        "prompt": "Discuss the impact of technology on modern society.",
        "model_answer": "Technology has transformed modern society by revolutionizing communication, education, healthcare, and economic activity. While it offers unprecedented access to information and global connectivity, it also raises concerns about privacy, digital divide, and social isolation. Overall, its benefits outweigh the negatives when used responsibly.",
        "max_score": 10,
    },
    {
        "prompt": "Describe the importance of environmental conservation.",
        "model_answer": "Environmental conservation is critical to sustaining biodiversity, clean air and water, and the natural resources future generations depend on. Human activities like deforestation and pollution accelerate climate change. Proactive conservation policies and individual responsibility are essential to protect our planet.",
        "max_score": 10,
    },
    {
        "prompt": "What are the advantages and disadvantages of social media?",
        "model_answer": "Social media enables instant communication, community building, and information sharing on a global scale. However, it also contributes to misinformation, cyberbullying, and mental health issues, particularly among young users. A balanced, mindful approach to social media use is key to maximizing benefits while minimizing harms.",
        "max_score": 10,
    },
    {
        "prompt": "Explain the role of education in personal development.",
        "model_answer": "Education fosters critical thinking, knowledge acquisition, and personal growth. It equips individuals with the skills needed to navigate complex societies and pursue meaningful careers. Beyond academic achievement, education cultivates empathy, civic responsibility, and lifelong learning habits.",
        "max_score": 10,
    },
    {
        "prompt": "Discuss the effects of climate change on global weather patterns.",
        "model_answer": "Climate change is intensifying extreme weather events including more frequent hurricanes, prolonged droughts, and unprecedented flooding. Rising global temperatures disrupt ecosystems and agricultural cycles. International cooperation and rapid transition to renewable energy are urgent priorities to mitigate these effects.",
        "max_score": 10,
    },
]

_STUDENT_EN = [
    [  # High (7–10)
        "Technology has profoundly changed how we live, work, and communicate. It has made education more accessible and improved healthcare diagnostics. However, we must address issues like digital addiction and cybersecurity threats to ensure technology serves humanity positively.",
        "Technology's impact on society is multifaceted. It has accelerated scientific discovery and connected people across continents. The challenges of privacy and misinformation require thoughtful regulation and digital literacy education.",
        "In the modern era, technology plays an indispensable role. From smartphones to artificial intelligence, it enhances productivity and quality of life. The key is to harness its power responsibly while mitigating risks like job displacement.",
    ],
    [  # Medium (4–7)
        "Technology is very important in today's world. It helps us communicate and find information quickly. But there are also problems like addiction and privacy issues that need to be solved.",
        "Technology affects many parts of life. Communication is faster, and education has changed a lot. There are bad things about technology we should be careful about.",
        "Technology has changed the world in many ways. It is useful for work and school. Sometimes technology can cause problems but mostly it is helpful.",
    ],
    [  # Low (1–4)
        "Technology is good. We use phones and computers every day. It makes life easy.",
        "Technology helps people. We can talk to friends and find things on the internet. Technology is nice.",
        "I use technology every day. It is good and bad. We need technology for school.",
    ],
]

ARABIC_PROMPTS = [
    {
        "prompt": "ناقش تأثير التكنولوجيا على المجتمع الحديث.",
        "model_answer": "أحدثت التكنولوجيا ثورة في المجتمع الحديث من خلال تحويل الاتصالات والتعليم والرعاية الصحية والنشاط الاقتصادي. وعلى الرغم من أنها توفر وصولاً غير مسبوق إلى المعلومات والترابط العالمي، فإنها تثير أيضاً مخاوف تتعلق بالخصوصية والفجوة الرقمية والعزلة الاجتماعية.",
        "max_score": 10,
    },
    {
        "prompt": "اشرح أهمية المحافظة على البيئة.",
        "model_answer": "تعد المحافظة على البيئة أمراً بالغ الأهمية للحفاظ على التنوع البيولوجي والهواء والمياه النظيفة والموارد الطبيعية التي تعتمد عليها الأجيال القادمة. وتؤدي الأنشطة البشرية كإزالة الغابات والتلوث إلى تسريع التغيرات المناخية.",
        "max_score": 10,
    },
    {
        "prompt": "ما هو دور التعليم في تنمية الشخصية؟",
        "model_answer": "يعزز التعليم التفكير النقدي واكتساب المعرفة والنمو الشخصي. إنه يزود الأفراد بالمهارات اللازمة للتنقل في المجتمعات المعقدة ومتابعة مسيرة مهنية ذات معنى. وبعيداً عن التحصيل الأكاديمي، يغرس التعليم التعاطف والمسؤولية المدنية.",
        "max_score": 10,
    },
]

_STUDENT_AR = [
    [
        "التكنولوجيا غيرت حياتنا بشكل كبير. أصبح التواصل أسهل والحصول على المعلومات أسرع. ومع ذلك، يجب أن نتعامل مع التكنولوجيا بمسؤولية لتجنب مخاطرها مثل الإدمان وانتهاك الخصوصية.",
        "تؤثر التكنولوجيا على المجتمع الحديث بطرق متعددة. فهي تسهم في تطوير التعليم والاقتصاد، وتوفر فرص عمل جديدة. غير أنها تستدعي منا الوعي الرقمي للتعامل مع التحديات المصاحبة لها.",
    ],
    [
        "التكنولوجيا مهمة جداً في حياتنا. نستخدمها للتواصل والبحث عن المعلومات. لها فوائد كثيرة لكن بعض الأشخاص يستخدمونها بشكل خاطئ.",
        "التكنولوجيا تساعد الناس في العمل والدراسة. هي مفيدة ولكن لها أضرار أيضاً. يجب أن نستخدمها بحكمة.",
    ],
    [
        "التكنولوجيا جيدة. نستخدم الهاتف كل يوم. هي مهمة للجميع.",
        "التكنولوجيا تغير الحياة. نستخدم الإنترنت. هي مفيدة.",
    ],
]


def _generate_synthetic_english(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    records = []
    bins = [(7, 10, 0), (4, 7, 1), (1, 4, 2)]
    for i in range(n):
        prompt_info = rng.choice(ENGLISH_PROMPTS)
        s_min, s_max, ti = rng.choice(bins)
        score = rng.uniform(s_min, s_max)
        student = rng.choice(_STUDENT_EN[ti]) + rng.choice(["", " This is very important.", ""])
        records.append({
            "essay_id": i, "essay_set": 1,
            "student_answer": student,
            "model_answer": prompt_info["model_answer"],
            "score": round(score, 1),
            "score_normalized": score / prompt_info["max_score"],
            "language": "english", "source": "synthetic",
        })
    return pd.DataFrame(records)


def _generate_synthetic_arabic(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    records = []
    bins = [(7, 10, 0), (4, 7, 1), (1, 4, 2)]
    for i in range(n):
        prompt_info = rng.choice(ARABIC_PROMPTS)
        s_min, s_max, ti = rng.choice(bins)
        score = rng.uniform(s_min, s_max)
        student = rng.choice(_STUDENT_AR[min(ti, 2)])
        records.append({
            "essay_id": i, "essay_set": 1,
            "student_answer": student,
            "model_answer": prompt_info["model_answer"],
            "score": round(score, 1),
            "score_normalized": score / prompt_info["max_score"],
            "language": "arabic", "source": "synthetic",
        })
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════

def acquire_english_data() -> pd.DataFrame:
    out = Path(config.ENGLISH_DATA_DIR) / "english_essays.csv"
    if out.exists():
        log.info(f"Loading cached English data from {out}")
        return pd.read_csv(out)

    df = _load_asap_from_disk()
    if df is None:
        _try_kaggle_download()
        df = _load_asap_from_disk()

    if df is not None:
        df = _parse_asap_df(df)
        if config.ASAP_ESSAY_SETS:
            df = df[df["essay_set"].astype(int).isin(config.ASAP_ESSAY_SETS)]
        log.info(f"ASAP loaded: {len(df)} rows")
    else:
        log.warning("ASAP unavailable — generating synthetic English dataset.")
        df = _generate_synthetic_english()
        log.info(f"Synthetic English data: {len(df)} rows")

    df.to_csv(out, index=False)
    return df


def acquire_arabic_data() -> pd.DataFrame:
    out = Path(config.ARABIC_DATA_DIR) / "arabic_essays.csv"
    if out.exists():
        log.info(f"Loading cached Arabic data from {out}")
        return pd.read_csv(out)

    dfs = []

    # 1. AR-AES (HuggingFace) — primary, 2,046 essays
    araes = _load_araes_hf()
    if araes is not None:
        dfs.append(araes)

    # 2. GLUPS-ASAG (GitHub ODS) — Islamic Education
    glups = _load_glups()
    if glups is not None:
        dfs.append(glups)

    # 3. MBZUAI arabic-aes-bea25 (GitHub)
    mbzuai = _load_mbzuai_arabic_aes()
    if mbzuai is not None:
        dfs.append(mbzuai)

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df["essay_set"] = df["essay_set"].astype(str)
        
        # ── Bilingual Integration (Phase 16) ──
        if getattr(config, "USE_BILINGUAL_DATA", False):
            log.info("Bilingual integration enabled: Merging English ASAP data ...")
            en_df = acquire_english_data()
            en_df["essay_set"] = en_df["essay_set"].astype(str)
            # Ensure consistent columns
            cols = ["student_answer", "model_answer", "score", "score_normalized", "essay_set", "language", "source"]
            df_merged = pd.concat([df[cols], en_df[cols]], ignore_index=True)
            df = df_merged
            log.info(f"Bilingual Combined: {len(df)} rows ({len(en_df)} English, {len(df)-len(en_df)} Arabic)")

        df["essay_id"] = range(len(df))
        log.info(f"Combined Arabic data: {len(df)} rows from {len(dfs)} source(s)")
    else:
        log.warning("All Arabic sources unavailable — using synthetic dataset.")
        df = _generate_synthetic_arabic()
        log.info(f"Synthetic Arabic data: {len(df)} rows")

    df.to_csv(out, index=False)
    return df


def acquire_data(lang: str = "english") -> pd.DataFrame:
    """Top-level entry point."""
    if lang.lower() == "english":
        return acquire_english_data()
    elif lang.lower() == "arabic":
        return acquire_arabic_data()
    else:
        raise ValueError(f"Unsupported language: {lang}")


if __name__ == "__main__":
    en = acquire_english_data()
    print(f"English: {len(en)} rows")
    print(en[["student_answer", "score"]].head(2))

    ar = acquire_arabic_data()
    print(f"Arabic: {len(ar)} rows")
    source_counts = ar.get("source", pd.Series(["unknown"] * len(ar))).value_counts()
    print("Sources:", source_counts.to_dict())
