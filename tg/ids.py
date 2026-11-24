"""Downstream IDS evaluation: augmentation of the training partition, RF/DNN classifiers, robustness (Section 3.9)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, f1_score
from .utils import device, set_seed


def augmentation_counts(y_train, classes, factor):
    """Windows to add per class: factor x training windows for every attack class, 0 for Benign."""
    return {k: (0 if classes[k] == "Benign" else int(round(factor * (y_train == k).sum()))) for k in range(len(classes))}


def augmented_flows(data, extra_windows: dict, T):
    """Training flows + synthetic windows (flattened to flows). extra_windows: class -> (n, T, d) array."""
    d = data["X_train"].shape[-1]
    Xs = [data["X_train"].reshape(-1, d)]; ys = [np.repeat(data["y_train"], T)]
    for k, W in extra_windows.items():
        if W is not None and len(W):
            Xs.append(W.reshape(-1, d)); ys.append(np.full(len(W) * T, k))
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys)


def ros_windows(data, counts, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for k, n in counts.items():
        idx = np.where(data["y_train"] == k)[0]
        out[k] = data["X_train"][rng.choice(idx, n, replace=True)] if n and len(idx) else None
    return out


def smote_flows(data, counts, T, seed):
    from imblearn.over_sampling import SMOTE
    d = data["X_train"].shape[-1]
    X = data["X_train"].reshape(-1, d); y = np.repeat(data["y_train"], T)
    strat = {k: int((y == k).sum() + n * T) for k, n in counts.items() if n > 0}
    Xr, yr = SMOTE(sampling_strategy=strat, k_neighbors=5, random_state=seed).fit_resample(X, y)
    return Xr.astype(np.float32), yr


class _DNN(nn.Module):
    def __init__(self, d, K):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, K))

    def forward(self, x):
        return self.net(x)


class IDS:
    """Random forest (100 trees) or DNN (256-128-64, Adam 1e-3, batch 1024) on individual flows."""

    def __init__(self, kind, K, seed=0, trees=100, epochs=10):
        self.kind, self.K, self.seed, self.trees, self.epochs = kind, K, seed, trees, epochs

    def fit(self, X, y):
        if self.kind == "rf":
            self.m = RandomForestClassifier(self.trees, n_jobs=-1, random_state=self.seed).fit(X, y)
            return self
        set_seed(self.seed); dev = device()
        self.m = _DNN(X.shape[1], self.K).to(dev)
        opt = torch.optim.Adam(self.m.parameters(), 1e-3)
        Xt = torch.as_tensor(X, device=dev); yt = torch.as_tensor(y, device=dev, dtype=torch.long)
        for _ in range(self.epochs):
            for i in torch.randperm(len(Xt), device=dev).split(1024):
                loss = nn.functional.cross_entropy(self.m(Xt[i]), yt[i])
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def predict(self, X):
        if self.kind == "rf":
            return self.m.predict(X)
        dev = device(); self.m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 65536):
                out.append(self.m(torch.as_tensor(X[i:i + 65536], device=dev)).argmax(1).cpu().numpy())
        return np.concatenate(out)


def ids_report(y, pred, classes):
    K = len(classes)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(K)), zero_division=0)
    attack = [k for k in range(K) if classes[k] != "Benign"]
    return {"per_class": {classes[k]: {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]),
                                       "support": int(s[k])} for k in range(K)},
            "macro_f1": float(f1_score(y, pred, labels=list(range(K)), average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, labels=list(range(K)), average="weighted", zero_division=0)),
            "attack_macro_recall": float(np.mean(r[attack])) if attack else float("nan")}


def perturb_timing(prep, X_windows, frac, seed):
    """Multiply timing features (original space) by (1+u), u~U[-frac, frac], independently per flow."""
    rng = np.random.default_rng(seed)
    d = X_windows.shape[-1]
    O = prep.inverse(X_windows).reshape(-1, d)
    if prep.timing_idx:
        O[:, prep.timing_idx] *= 1 + rng.uniform(-frac, frac, size=(len(O), len(prep.timing_idx)))
    return prep.scale(O).reshape(X_windows.shape)


def evaluate_ids(prep, data, X_aug, y_aug, cfg, seed, eval_split="test"):
    """Train RF and DNN on (augmented) training flows; report clean and timing-perturbed performance."""
    classes, T = prep.classes, prep.T
    d = X_aug.shape[1]
    Xe, ye = data[f"X_{eval_split}"], data[f"y_{eval_split}"]
    Xe_f, ye_f = Xe.reshape(-1, d), np.repeat(ye, T)
    Xp_f = perturb_timing(prep, Xe, cfg["eval"]["perturb"], seed + 1000).reshape(-1, d)
    out = {}
    for kind in ("rf", "dnn"):
        clf = IDS(kind, len(classes), seed, cfg["eval"]["rf_trees"], cfg["eval"]["dnn_epochs"]).fit(X_aug, y_aug)
        clean = ids_report(ye_f, clf.predict(Xe_f), classes)
        pert = ids_report(ye_f, clf.predict(Xp_f), classes)
        clean["perturbed_attack_macro_recall"] = pert["attack_macro_recall"]
        clean["robustness_ratio"] = pert["attack_macro_recall"] / max(clean["attack_macro_recall"], 1e-9)
        clean["perturbed_macro_f1"] = pert["macro_f1"]
        out[kind] = clean
    return out
