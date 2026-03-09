"""
scoring_agent.py — Advanced scoring models for Phase 14.
Includes CORAL (Ordinal Regression), Soft-QWK Loss, and Stacking Ensemble.
"""
import sys
import logging
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Optional, List, Tuple

# Optional Pytorch imports handled carefully for environment stability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    TORCH_AVAILABLE = True
    _BaseModule = nn.Module
except ImportError:
    TORCH_AVAILABLE = False
    _BaseModule = object

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)

def _get_device():
    if not TORCH_AVAILABLE: return "cpu"
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
# Advanced Loss Functions
# ──────────────────────────────────────────────

class SoftQWKLoss(_BaseModule):
    def __init__(self, num_classes=6, eps=1e-10):
        if not TORCH_AVAILABLE: return
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        W = torch.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                W[i, j] = float((i - j) ** 2) / float((num_classes - 1) ** 2)
        self.register_buffer("W", W)

    def forward(self, preds, targets):
        if not TORCH_AVAILABLE: return torch.tensor(0.0)
        if targets.dim() == 1:
            targets = F.one_hot(targets.long(), self.num_classes).float()
        O = torch.matmul(targets.t(), preds)
        O = O / (O.sum() + self.eps)
        E = torch.matmul(targets.sum(dim=0).unsqueeze(1), preds.sum(dim=0).unsqueeze(0))
        E = E / (E.sum() + self.eps)
        weighted_O = torch.sum(self.W * O)
        weighted_E = torch.sum(self.W * E)
        return weighted_O / (weighted_E + self.eps)

class SupConLoss(_BaseModule):
    def __init__(self, temperature=0.07):
        if not TORCH_AVAILABLE: return
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        if not TORCH_AVAILABLE: return torch.tensor(0.0)
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
        return (-mean_log_prob_pos).mean()

# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────

class MLPScoringModel:
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.3, encoder_name: str = None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.encoder_name = encoder_name
        self.num_classes = getattr(config, "NUM_CLASSES", 6)
        self.model = None
        self.encoder = None
        self._torch_available = TORCH_AVAILABLE
        if TORCH_AVAILABLE: self._build()

    def _build(self):
        class _CORAL_MLP(nn.Module):
            def __init__(self, in_dim, hidden, drop, num_classes):
                super().__init__()
                self.use_prompt = getattr(config, "USE_PROMPT_EMBS", True)
                self.prompt_dim = 64 if self.use_prompt else 0
                if self.use_prompt: self.prompt_embed = nn.Embedding(50, self.prompt_dim)
                self.in_dim = in_dim
                actual_in_dim = in_dim + self.prompt_dim
                self.features = nn.Sequential(
                    nn.Linear(actual_in_dim, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(0.1), nn.Dropout(drop),
                    nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.LeakyReLU(0.1), nn.Dropout(drop),
                )
                self.fc = nn.Linear(hidden // 2, 1, bias=False)
                self.thresholds = nn.Parameter(torch.sort(torch.randn(num_classes - 1))[0])

            def forward(self, x, prompt_ids=None):
                if self.use_prompt:
                    if prompt_ids is not None:
                        p_idx = torch.LongTensor(prompt_ids).to(x.device)
                        x = torch.cat([x, self.prompt_embed(p_idx)], dim=1)
                    else:
                        x = torch.cat([x, torch.zeros((x.shape[0], self.prompt_dim)).to(x.device)], dim=1)
                feats = self.features(x)
                return self.fc(feats) - self.thresholds

        self.model = _CORAL_MLP(self.input_dim, self.hidden_dim, self.dropout_rate, self.num_classes)

    def predict(self, X: np.ndarray, prompt_ids: List[int] = None) -> np.ndarray:
        if not self._torch_available: return np.zeros(len(X))
        device = _get_device() ; self.model.eval()
        with torch.no_grad():
            Xt = torch.FloatTensor(X).to(device)
            probas = torch.sigmoid(self.model(Xt, prompt_ids=prompt_ids))
            return (torch.sum(probas, dim=1) / (self.num_classes - 1)).cpu().numpy()

    def save(self, path: str): torch.save(self.model.state_dict(), path)
    def load(self, path: str): self.model.load_state_dict(torch.load(path, map_location=_get_device()))

class RidgeScoringModel:
    def __init__(self, alpha=1.0):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        self.model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    def fit(self, X, y, **kwargs): self.model.fit(X, y); return {"train_loss": [0.0]}
    def predict(self, X, **kwargs): return np.clip(self.model.predict(X), 0, 1)
    def save(self, path):
        try: import joblib ; joblib.dump(self.model, path)
        except: 
            with open(path, "wb") as f: pickle.dump(self.model, f)
    def load(self, path):
        # Ultra-robust load
        for loader in [pickle.load, lambda f: __import__('joblib').load(f), lambda f: torch.load(f, map_location="cpu")]:
            try:
                if loader == pickle.load:
                    with open(path, "rb") as f: obj = loader(f)
                else: obj = loader(path)
                if hasattr(obj, "predict"): self.model = obj; return
            except: continue
        raise RuntimeError(f"Failed to load Ridge from {path}")

class SVRScoringModel:
    def __init__(self, C=1.0):
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        self.model = Pipeline([("scaler", StandardScaler()), ("svr", SVR(C=C, epsilon=0.01))])
    def fit(self, X, y, **kwargs): self.model.fit(X, y); return {"train_loss": [0.0]}
    def predict(self, X, **kwargs): return np.clip(self.model.predict(X), 0, 1)
    def save(self, path):
        try: import joblib ; joblib.dump(self.model, path)
        except:
            with open(path, "wb") as f: pickle.dump(self.model, f)
    def load(self, path):
        # Ultra-robust load
        methods = [
            lambda: pickle.load(open(path, "rb")),
            lambda: __import__('joblib').load(path),
            lambda: torch.load(path, map_location="cpu")
        ]
        for m in methods:
            try:
                obj = m()
                if hasattr(obj, "predict"): self.model = obj; return
            except: continue
        raise RuntimeError(f"Failed to load SVR from {path}")

class StackingEnsembleModel:
    def __init__(self, input_dim: int):
        self.nn_model = CrossAttentionScoringModel(input_dim)
        self.ridge = RidgeScoringModel()
        self.svr = SVRScoringModel()
        from sklearn.linear_model import Ridge
        self.meta_model = Ridge(alpha=1.0)

    def predict(self, X, **kwargs):
        p_nn = self.nn_model.predict(X, prompt_ids=kwargs.get("prompt_ids"))
        p_ridge = self.ridge.predict(X)
        p_svr = self.svr.predict(X)
        return np.clip(self.meta_model.predict(np.stack([p_nn, p_ridge, p_svr], axis=1)), 0, 1)

    def save(self, path):
        self.nn_model.save(path + ".nn")
        self.ridge.save(path + ".ridge")
        self.svr.save(path + ".svr")
        try: __import__('joblib').dump(self.meta_model, path + ".meta")
        except: pickle.dump(self.meta_model, open(path + ".meta", "wb"))

    def load(self, path):
        for m, ext in [(self.nn_model, ".nn"), (self.ridge, ".ridge"), (self.svr, ".svr")]:
            if os.path.exists(path + ext): m.load(path + ext)
        meta_path = path + ".meta"
        try: self.meta_model = __import__('joblib').load(meta_path)
        except: self.meta_model = pickle.load(open(meta_path, "rb"))

class CrossAttentionScoringModel(MLPScoringModel):
    def _build(self):
        class _AttnMLP(nn.Module):
            def __init__(self, in_dim, hidden, drop, num_classes):
                super().__init__()
                self.emb_dim = 768
                self.attn = nn.MultiheadAttention(embed_dim=self.emb_dim, num_heads=8, dropout=drop, batch_first=True)
                self.features = nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(0.1), nn.Dropout(drop), nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.LeakyReLU(0.1))
                self.fc = nn.Linear(hidden // 2, 1, bias=False)
                self.thresholds = nn.Parameter(torch.sort(torch.randn(num_classes - 1))[0])

            def forward(self, x, prompt_ids=None):
                s_emb = x[:, :self.emb_dim].unsqueeze(1)
                m_emb = x[:, self.emb_dim:2*self.emb_dim].unsqueeze(1)
                attn_out, _ = self.attn(s_emb, m_emb, m_emb)
                x_new = torch.cat([s_emb.squeeze(1), attn_out.squeeze(1), x[:, 2*self.emb_dim:]], dim=1)
                feats = self.features(x_new)
                return self.fc(feats) - self.thresholds
        self.model = _AttnMLP(self.input_dim, self.hidden_dim, self.dropout_rate, self.num_classes)

def build_scoring_model(model_type: str, input_dim: int = 0):
    model_type = model_type.lower()
    if model_type == "ridge": return RidgeScoringModel()
    if model_type == "svr": return SVRScoringModel()
    if model_type in ["ensemble", "stacking"]: return StackingEnsembleModel(input_dim)
    return MLPScoringModel(input_dim)
