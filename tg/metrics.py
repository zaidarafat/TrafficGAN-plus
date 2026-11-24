"""Fidelity, temporal fidelity, sample-quality, diversity and TSTR metrics (Section 3.9)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from .utils import ks_2samp_stat, device


def fidelity(prep, real_X, syn_X, n_boot=1000, sub=1000, seed=0):
    """Per-feature KS (continuous, bootstrapped) or TVD (discrete) in the original feature space."""
    rng = np.random.default_rng(seed)
    d = real_X.shape[-1]
    R = prep.inverse(real_X).reshape(-1, d)
    S = prep.inverse(syn_X).reshape(-1, d)
    per = {}
    for j, name in enumerate(prep.feature_cols):
        if prep.discrete[j]:
            vals = np.union1d(np.unique(R[:, j]), np.unique(S[:, j]))
            pr = np.array([(R[:, j] == v).mean() for v in vals])
            ps = np.array([(S[:, j] == v).mean() for v in vals])
            per[name] = ("TVD", float(0.5 * np.abs(pr - ps).sum()))
        else:
            ks = [ks_2samp_stat(R[rng.integers(0, len(R), sub), j], S[rng.integers(0, len(S), sub), j])
                  for _ in range(n_boot)]
            per[name] = ("KS", float(np.mean(ks)))
    return {"per_feature": per, "mean": float(np.mean([v for _, v in per.values()])),
            "mean_ks_continuous": float(np.mean([v for m, v in per.values() if m == "KS"] or [np.nan]))}


def temporal(prep, real_X, syn_X, max_n=200_000, seed=0):
    """Increment-KS and lag-1 autocorrelation error on continuous features (model space)."""
    rng = np.random.default_rng(seed)
    cont = np.where(~prep.discrete)[0]
    dr = (real_X[:, 1:] - real_X[:, :-1]).reshape(-1, real_X.shape[-1])
    ds = (syn_X[:, 1:] - syn_X[:, :-1]).reshape(-1, syn_X.shape[-1])
    dr = dr[rng.choice(len(dr), min(max_n, len(dr)), replace=False)]
    ds = ds[rng.choice(len(ds), min(max_n, len(ds)), replace=False)]
    inc_ks = float(np.mean([ks_2samp_stat(dr[:, j], ds[:, j]) for j in cont]))

    def acf1(X, j):
        a, b = X[:, :-1, j].ravel(), X[:, 1:, j].ravel()
        if a.std() < 1e-9 or b.std() < 1e-9:
            return np.nan
        return np.corrcoef(a, b)[0, 1]
    errs = [abs(acf1(real_X, j) - acf1(syn_X, j)) for j in cont]
    errs = [e for e in errs if np.isfinite(e)]
    return {"increment_ks": inc_ks, "acf_error": float(np.mean(errs)) if errs else float("nan")}


class _SeqClf(nn.Module):
    def __init__(self, d, h=32, K=1):
        super().__init__()
        self.lstm = nn.LSTM(d, h, num_layers=2, batch_first=True)
        self.out = nn.Linear(h, K)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.out(h[:, -1]).squeeze(-1)


def discriminative_score(real_X, syn_X, epochs=5, seed=0, max_n=20000):
    """|accuracy - 0.5| of a 2-layer LSTM trained to separate real from synthetic windows (80/20 split)."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed); dev = device()
    n = min(len(real_X), len(syn_X), max_n)
    X = np.concatenate([real_X[rng.choice(len(real_X), n, False)], syn_X[rng.choice(len(syn_X), n, False)]])
    y = np.concatenate([np.ones(n), np.zeros(n)]).astype(np.float32)
    p = rng.permutation(2 * n); X, y = X[p], y[p]
    cut = int(0.8 * 2 * n)
    m = _SeqClf(X.shape[-1]).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3)
    Xt, yt = torch.as_tensor(X[:cut], device=dev), torch.as_tensor(y[:cut], device=dev)
    for _ in range(epochs):
        for i in torch.randperm(cut).split(256):
            loss = nn.functional.binary_cross_entropy_with_logits(m(Xt[i]), yt[i])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pr = (m(torch.as_tensor(X[cut:], device=dev)) > 0).float().cpu().numpy()
    return float(abs((pr == y[cut:]).mean() - 0.5))


def prdc(real_X, syn_X, k=5, n=5000, seed=0):
    """Precision, recall, density, coverage (Naeem et al., 2020) on standardised flattened windows."""
    rng = np.random.default_rng(seed); dev = device()
    R = real_X[rng.choice(len(real_X), min(n, len(real_X)), False)].reshape(min(n, len(real_X)), -1)
    S = syn_X[rng.choice(len(syn_X), min(n, len(syn_X)), False)].reshape(min(n, len(syn_X)), -1)
    mu, sd = R.mean(0), R.std(0) + 1e-6
    R, S = torch.as_tensor((R - mu) / sd, device=dev), torch.as_tensor((S - mu) / sd, device=dev)
    drr, dss, drs = torch.cdist(R, R), torch.cdist(S, S), torch.cdist(R, S)
    rr = drr.kthvalue(k + 1, dim=1).values                     # k-NN radius of each real point (self excluded)
    rs = dss.kthvalue(k + 1, dim=1).values
    precision = (drs < rr[:, None]).any(0).float().mean().item()
    recall = (drs < rs[None, :]).any(1).float().mean().item()
    density = (1.0 / k) * (drs < rr[:, None]).sum(0).float().mean().item()
    coverage = (drs.min(1).values < rr).float().mean().item()
    return {"precision": precision, "recall": recall, "density": density, "coverage": coverage}


class EntropyJudge:
    """Independent random forest trained on real training windows; judges classes of synthetic windows."""

    def __init__(self, X_train, y_train, seed=0, max_per_class=5000):
        rng = np.random.default_rng(seed)
        idx = np.concatenate([rng.permutation(np.where(y_train == k)[0])[:max_per_class] for k in np.unique(y_train)])
        self.K = int(y_train.max() + 1)
        self.rf = RandomForestClassifier(200, n_jobs=-1, random_state=seed).fit(
            X_train[idx].reshape(len(idx), -1), y_train[idx])

    def __call__(self, syn_X, requested):
        pred = self.rf.predict(syn_X.reshape(len(syn_X), -1))
        p = np.bincount(pred, minlength=self.K) / len(pred)
        nz = p[p > 0]
        H = float(-(nz * np.log2(nz)).sum())
        return {"class_entropy_bits": H, "class_entropy_normalized": H / np.log2(self.K),
                "max_entropy_bits": float(np.log2(self.K)), "label_agreement": float((pred == requested).mean())}


def tstr(syn_X, syn_y, test_X, test_y, seed=0, trees=100):
    """Train a flow-level RF on synthetic flows only, test on real test flows (macro-F1)."""
    T = syn_X.shape[1]; d = syn_X.shape[-1]
    rf = RandomForestClassifier(trees, n_jobs=-1, random_state=seed)
    rf.fit(syn_X.reshape(-1, d), np.repeat(syn_y, T))
    pred = rf.predict(test_X.reshape(-1, d))
    return float(f1_score(np.repeat(test_y, T), pred, average="macro"))
