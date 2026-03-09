"""
improvement_agent.py — Autonomous improvement loop.

Reads previous experiment metrics, selects a strategy, patches config,
and returns updated training kwargs for the next run.
"""
import sys
import json
import logging
import copy
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

log = logging.getLogger(__name__)

# ── Improvement strategies (applied in sequence then cycling) ──────────────
STRATEGIES = [
    {
        "name": "increase_hidden_dim",
        "description": "Increase MLP hidden dimension to 512 for more capacity.",
        "patch": {"HIDDEN_DIM": 512, "DROPOUT": 0.3},
    },
    {
        "name": "lower_lr_more_epochs",
        "description": "Halve learning rate and add more epochs for finer convergence.",
        "patch": {"LEARNING_RATE": 5e-4, "EPOCHS": 50},
    },
    {
        "name": "switch_to_svr",
        "description": "Try SVR as scoring model for potentially better generalisation.",
        "patch": {"SCORING_MODEL": "svr"},
    },
    {
        "name": "switch_to_ridge_baseline",
        "description": "Try Ridge regression — strong simple baseline.",
        "patch": {"SCORING_MODEL": "ridge"},
    },
    {
        "name": "switch_back_mlp_wide",
        "description": "Return to MLP with wider hidden layer (1024) and lower dropout.",
        "patch": {"SCORING_MODEL": "mlp", "HIDDEN_DIM": 1024, "DROPOUT": 0.15, "LEARNING_RATE": 3e-4},
    },
    {
        "name": "upgrade_encoder_english",
        "description": "Upgrade English encoder to all-mpnet-base-v2 for richer embeddings.",
        "patch": {"ENGLISH_ENCODER": "sentence-transformers/all-mpnet-base-v2"},
    },
    {
        "name": "increase_kfolds",
        "description": "Use 10-fold CV for more reliable estimates.",
        "patch": {"K_FOLDS": 10},
    },
    {
        "name": "reduce_dropout_increase_batch",
        "description": "Reduce dropout to 0.1 and increase batch size to 64.",
        "patch": {"DROPOUT": 0.1, "BATCH_SIZE": 64, "EPOCHS": 40},
    },
    {
        "name": "cosine_lr_restart",
        "description": "Very low LR with extended epochs for fine-tuning.",
        "patch": {"LEARNING_RATE": 1e-4, "EPOCHS": 60, "DROPOUT": 0.2, "HIDDEN_DIM": 512},
    },
    {
        "name": "mlp_deep_narrow",
        "description": "Try a deeper, narrower MLP.",
        "patch": {"SCORING_MODEL": "mlp", "HIDDEN_DIM": 256, "LEARNING_RATE": 5e-4, "EPOCHS": 40},
    },
]


def select_strategy(iteration: int, history: list) -> dict:
    """
    Select the next improvement strategy based on iteration number and history.

    Args:
        iteration: Current iteration index (0-based).
        history: List of metric dicts from previous experiments.

    Returns:
        Strategy dict with 'name', 'description', 'patch'.
    """
    # If last result improved significantly, continue with same strategy
    if len(history) >= 2:
        last_qwk = history[-1].get("qwk", 0)
        prev_qwk = history[-2].get("qwk", 0)
        if last_qwk - prev_qwk > 0.02:
            log.info("Strategy: continuing previous because QWK improved by >0.02")
            strategy = STRATEGIES[(iteration - 1) % len(STRATEGIES)]
            return strategy

    # Otherwise cycle through strategies
    strategy = STRATEGIES[iteration % len(STRATEGIES)]
    log.info(f"Selected strategy [{iteration}]: {strategy['name']} — {strategy['description']}")
    return strategy


def apply_strategy(strategy: dict) -> dict:
    """
    Apply strategy patches to config and return a kwargs dict
    for the training_agent.train() call.
    """
    patch = strategy.get("patch", {})
    train_kwargs = {}

    for key, value in patch.items():
        if hasattr(cfg, key):
            old = getattr(cfg, key)
            setattr(cfg, key, value)
            log.info(f"  config.{key}: {old} → {value}")
        # Map config keys to train() kwargs
        train_kwargs_map = {
            "SCORING_MODEL": "scoring_model_type",
            "LEARNING_RATE": "lr",
            "EPOCHS": "epochs",
            "BATCH_SIZE": "batch_size",
            "EARLY_STOPPING": "patience",
            "K_FOLDS": "n_folds",
            "ENGLISH_ENCODER": "encoder_name",
            "ARABIC_ENCODER": "encoder_name",
        }
        if key in train_kwargs_map:
            train_kwargs[train_kwargs_map[key]] = value

    return train_kwargs


def diagnose(metrics: dict) -> str:
    """
    Diagnose model weaknesses from metrics.

    Returns a human-readable diagnosis string.
    """
    qwk = metrics.get("qwk", 0)
    rmse = metrics.get("rmse", 999)
    acc = metrics.get("accuracy", 0)
    diagnosis_parts = []

    if qwk < 0.4:
        diagnosis_parts.append("QWK very low — model not capturing ordinal relationships.")
    elif qwk < 0.6:
        diagnosis_parts.append("QWK moderate — try richer embeddings or more capacity.")
    elif qwk < 0.75:
        diagnosis_parts.append("QWK good — fine-tune LR or try ensemble.")
    else:
        diagnosis_parts.append("QWK near target — minor tuning needed.")

    if rmse > 2.0:
        diagnosis_parts.append("High RMSE — predictions far from true scores.")
    if acc < 0.3:
        diagnosis_parts.append("Low exact accuracy — scoring distribution mismatch.")

    return " | ".join(diagnosis_parts) if diagnosis_parts else "Performance acceptable."


def propose_improvement(iteration: int, metrics: dict, history: list) -> dict:
    """
    Main entry: propose + apply next improvement.

    Returns:
        dict with 'strategy_name', 'description', 'diagnosis', 'train_kwargs'
    """
    diagnosis = diagnose(metrics)
    log.info(f"Diagnosis: {diagnosis}")

    strategy = select_strategy(iteration, history)
    train_kwargs = apply_strategy(strategy)

    return {
        "strategy_name": strategy["name"],
        "description": strategy["description"],
        "diagnosis": diagnosis,
        "train_kwargs": train_kwargs,
    }
