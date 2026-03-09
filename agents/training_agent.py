"""
training_agent.py — K-Fold cross-validation training loop.

Integrates feature extraction and scoring model training.
Saves best checkpoint to checkpoints/.
Logs metrics to experiments/experiment_tracker.
"""
import sys
import logging
import numpy as np
import os
from pathlib import Path
from typing import Optional
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from agents.feature_agent import build_feature_matrix
from agents.scoring_agent import build_scoring_model
from agents.evaluation_agent import compute_metrics
from agents.augmentation_agent import augment_batch

log = logging.getLogger(__name__)


def train(
    student_texts: list,
    model_texts: list,
    scores_normalized: list,
    scores_raw: list,
    essay_sets: Optional[list] = None,
    lang: str = "english",
    exp_id: str = "exp_001",
    scoring_model_type: Optional[str] = None,
    encoder_name: Optional[str] = None,
    n_folds: int = None,
    epochs: int = None,
    lr: float = None,
    batch_size: int = None,
    patience: int = None,
    languages: Optional[list] = None,
) -> dict:
    """
    Full training pipeline with K-Fold CV.

    Args:
        student_texts: List of student answer strings.
        model_texts: List of model/gold answer strings.
        scores_normalized: Normalized scores in [0, 1].
        scores_raw: Raw scores for metric computation.
        lang: 'english' or 'arabic'.
        exp_id: Unique experiment identifier.
        ... (rest from config defaults if None)

    Returns:
        dict with keys: qwk, rmse, pearson, accuracy, fold_metrics, checkpoint_path
    """
    # Apply defaults from config
    scoring_model_type = scoring_model_type or config.SCORING_MODEL
    encoder_name = encoder_name or (
        config.ARABIC_ENCODER if lang == "arabic" else config.ENGLISH_ENCODER)
    n_folds = n_folds or config.K_FOLDS
    epochs = epochs or config.EPOCHS
    lr = lr or config.LEARNING_RATE
    batch_size = batch_size or config.BATCH_SIZE
    patience = patience or config.EARLY_STOPPING

    # ── Build full feature matrix once (expensive) ──
    log.info(f"[{exp_id}] Building feature matrix for {len(student_texts)} samples …")
    X, cos_sims = build_feature_matrix(
        student_texts, model_texts, essay_sets=essay_sets, lang=lang, encoder_name=encoder_name
    )
    y = np.array(scores_normalized, dtype=float)
    y_raw = np.array(scores_raw, dtype=float)

    input_dim = X.shape[1]
    log.info(f"[{exp_id}] Feature dim: {input_dim}  |  Model: {scoring_model_type}")

    # ── K-Fold CV ──
    # ── Custom Bilingual K-Fold (Phase 16) ──
    if languages is not None and lang == "arabic":
        langs_npy = np.array(languages)
        ara_idx = np.where(langs_npy == "arabic")[0]
        eng_idx = np.where(langs_npy == "english")[0]
        
        if len(eng_idx) > 0:
            log.info(f"[{exp_id}] Custom Bilingual K-Fold: {len(ara_idx)} Arabic, {len(eng_idx)} English")
            kf_ara = KFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_SEED)
            splits = []
            for t_idx_ara, v_idx_ara in kf_ara.split(ara_idx):
                t_idx = np.concatenate([ara_idx[t_idx_ara], eng_idx])
                v_idx = ara_idx[v_idx_ara]
                splits.append((t_idx, v_idx))
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_SEED)
            splits = list(kf.split(X))
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_SEED)
        splits = list(kf.split(X))

    fold_metrics = []
    all_val_idxs = []
    all_preds = np.zeros(len(y))
    all_true = y_raw.copy()
    best_qwk = -1.0
    best_model_state = None

    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        all_val_idxs.append(val_idx)
        log.info(f"[{exp_id}] Fold {fold_idx}/{n_folds}")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        y_raw_val = y_raw[val_idx]

        model = build_scoring_model(scoring_model_type, input_dim=input_dim)

        # ── Apply Augmentation (Phase 14) ──
        st_tr = [student_texts[i] for i in train_idx]
        mt_tr = [model_texts[i] for i in train_idx]
        pr_tr = [essay_sets[i] for i in train_idx] if essay_sets else None
        
        if lang == "arabic":
            st_tr_aug, y_tr_aug = augment_batch(st_tr, y_tr.tolist(), lang=lang, multiplier=4)
            # Duplicate mt and pr for augmented samples
            mt_tr_aug = mt_tr * 4
            pr_tr_aug = pr_tr * 4 if pr_tr else None
            # Need to duplicate X_tr as well (as a placeholder)
            X_tr_aug = np.tile(X_tr, (4, 1))
        else:
            st_tr_aug, mt_tr_aug, y_tr_aug, pr_tr_aug, X_tr_aug = st_tr, mt_tr, y_tr, pr_tr, X_tr

        # Fit
        if scoring_model_type in ["mlp", "ensemble", "stacking", "attention"]:
            model.fit(
                X_tr_aug, np.array(y_tr_aug),
                epochs=epochs, lr=lr, batch_size=batch_size,
                X_val=X_val, y_val=y_val, patience=patience,
                student_texts=st_tr_aug,
                model_texts=mt_tr_aug,
                val_student_texts=[student_texts[i] for i in val_idx],
                val_model_texts=[model_texts[i] for i in val_idx],
                prompt_ids=pr_tr_aug,
                val_prompt_ids=[essay_sets[i] for i in val_idx] if essay_sets else None
            )
        else:
            model.fit(X_tr, y_tr)

        # Predict
        preds_norm = model.predict(X_val, prompt_ids=[essay_sets[i] for i in val_idx] if essay_sets else None)

        # Rescale to raw score range based on the validation target language
        target_max = y_raw_val.max()
        target_min = y_raw_val.min()
        preds_raw = preds_norm * (target_max - target_min) + target_min
        preds_raw = np.clip(preds_raw, target_min, target_max)
        
        with open("logs/debug.txt", "a") as dbg_file:
            dbg_file.write(f"Fold target bounds: min={target_min:.2f} max={target_max:.2f}\n")
            dbg_file.write(f"Fold preds: min={preds_raw.min():.2f} max={preds_raw.max():.2f} std={preds_raw.std():.8f}\n")
            dbg_file.write(f"Fold targets: min={y_raw_val.min():.2f} max={y_raw_val.max():.2f} std={y_raw_val.std():.8f}\n")
        
        all_preds[val_idx] = preds_raw

        fmetrics = compute_metrics(y_raw_val, preds_raw)
        fold_metrics.append(fmetrics)
        log.info(
            f"  QWK={fmetrics['qwk']:.4f}  RMSE={fmetrics['rmse']:.4f}  "
            f"Acc={fmetrics['accuracy']:.4f}"
        )

        if fmetrics["qwk"] > best_qwk:
            best_qwk = fmetrics["qwk"]
            # Save best fold model
            ckpt_dir = Path(config.CHECKPOINTS_DIR)
            ckpt_dir.mkdir(exist_ok=True)
            ckpt_path = str(ckpt_dir / f"best_model_{exp_id}.pkl")
            try:
                model.save(ckpt_path)
                best_model_state = ckpt_path
            except Exception as e:
                log.warning(f"Could not save checkpoint: {e}")

    # ── Aggregate metrics ──
    # ONLY aggregate over indices that were actually evaluated
    evaluated_idx = np.concatenate(all_val_idxs) if len(all_val_idxs) > 0 else np.arange(len(all_true))
    overall = compute_metrics(all_true[evaluated_idx], all_preds[evaluated_idx])
    overall["fold_metrics"] = fold_metrics
    overall["checkpoint_path"] = best_model_state or ""


    log.info(
        f"[{exp_id}] FINAL ─ QWK={overall['qwk']:.4f}  "
        f"RMSE={overall['rmse']:.4f}  Acc={overall['accuracy']:.4f}"
    )
    return overall
