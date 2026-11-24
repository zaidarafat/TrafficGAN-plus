import json, os, random
import numpy as np
import torch
import yaml


def load_cfg(path="configs/default.yaml", overrides=None):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    for k, v in (overrides or {}).items():          # "train.n_iters=200"
        sec, key = k.split(".", 1)
        cfg[sec][key] = v
    return cfg


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ks_2samp_stat(a, b):
    """Two-sample KS statistic sup_v |F_a(v) - F_b(v)| (exact, vectorised over evaluation points)."""
    a, b = np.sort(a), np.sort(b)
    v = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, v, "right") / len(a) - np.searchsorted(b, v, "right") / len(b))))


def mean_ks_fast(real, fake):
    return float(np.mean([ks_2samp_stat(real[:, j], fake[:, j]) for j in range(real.shape[1])]))


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_json(path):
    with open(path) as fh:
        return json.load(fh)
