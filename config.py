"""
config.py — Central configuration for the AES system.
Automatically patched by improvement_agent during the autonomous loop.
"""
import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
EVALUATION_DIR  = os.path.join(BASE_DIR, "evaluation")

ENGLISH_DATA_DIR = os.path.join(DATA_DIR, "english")
ARABIC_DATA_DIR  = os.path.join(DATA_DIR, "arabic")

# ──────────────────────────────────────────────
# Model Selection (patched by improvement_agent)
# ──────────────────────────────────────────────
ENGLISH_ENCODER = "sentence-transformers/all-mpnet-base-v2"    # GPU-grade quality
ARABIC_ENCODER  = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
ARABIC_ENCODER_2 = None

# ──────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────
RANDOM_SEED      = 42
K_FOLDS          = 5
BATCH_SIZE       = 32
LEARNING_RATE    = 1e-3
EPOCHS           = 30
EARLY_STOPPING   = 10
HIDDEN_DIM       = 256
DROPOUT          = 0.2
FINE_TUNE_ENCODER = True
UNFREEZE_LAYERS  = 8           # Deeper fine-tuning for complex Arabic features
LR_HEAD          = 1e-4        # More cautious LR for ensemble stability
LR_ENCODER       = 2e-6        # Lowered for stable deep fine-tuning
SCORING_MODEL    = "svr" # Best performance (0.91 QWK) achieved with SVR in Exp 867e6b85
USE_BILINGUAL_DATA = True         # Integrate English ASAP with Arabic AR-AES
USE_PROMPT_EMBS  = True        # Enable Prompt-Aware Scoring
USE_POS_FEATURES = True        # Enable POS Distribution features
NUM_CLASSES      = 21          # Essential for 0.5 increment AR-AES scores
USE_SOFT_QWK     = False       # Disabled due to mathematically invalid continuous broadcast in Ordinal MLP
USE_CONTRASTIVE  = True        # Enable SupCon for better embedding separation
LR_HEAD          = 5e-4
LR_ENCODER       = 2e-6        # Consistent with updated param

# ──────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────
TARGET_QWK       = 0.85
MAX_ITERATIONS   = 50
MAX_NO_IMPROVE   = 10

# ──────────────────────────────────────────────
# ASAP essay sets to use (1-8); None = all
# ──────────────────────────────────────────────
ASAP_ESSAY_SETS  = [1, 2]      # start with 2 sets for speed

# ──────────────────────────────────────────────
# OCR
# ──────────────────────────────────────────────
TESSERACT_CMD    = r"tesseract"   # update if installed elsewhere
USE_TROCR        = True           # GPU available — use TrOCR for handwriting
