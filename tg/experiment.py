"""One (dataset, method, seed) experiment: train generator -> fidelity/diversity metrics -> augmentation -> IDS."""
from __future__ import annotations
import os, pickle, re, time
import numpy as np
import torch
from .train_gan import VARIANTS, train_trafficgan, train_mlp_gan, generate
from .metrics import fidelity, temporal, discriminative_score, prdc, EntropyJudge, tstr
from .ids import augmentation_counts, augmented_flows, ros_windows, smote_flows, evaluate_ids
from .models import Generator
from .utils import device, save_json, load_json

TG_METHODS = ["trafficgan", "no_tgp", "no_ase", "wgan_gp", "alt_sn", "alt_lp", "alt_acgan", "alt_pacgan"]
OTHER_GEN = ["mlp_gan", "mlp_wgan", "ctgan", "copulagan", "tabddpm", "timegan", "dgan"]
NON_GEN = ["none", "ros", "smote"]
ALL_METHODS = NON_GEN + OTHER_GEN + TG_METHODS


def build_sampler(method, data, prep, cfg, seed, run_dir, log=print):
    """Train `method` and return (sampler: labels -> scaled windows, train_info)."""
    T, d = data["X_train"].shape[1:]
    K = len(prep.classes)
    be = cfg["experiments"]["baseline_epochs"]
    dev = device()
    if method in VARIANTS:
        G, info = train_trafficgan(data, cfg, method, seed, run_dir, log=log)
        return (lambda labels: generate(G, labels, dev, seed=seed)), info
    if method in ("mlp_gan", "mlp_wgan"):
        return train_mlp_gan(data, cfg, wasserstein=(method == "mlp_wgan"), seed=seed, out_dir=run_dir, log=log)
    t0 = time.time()
    if method in ("ctgan", "copulagan"):
        from .baselines.external import SDVBaseline
        m = SDVBaseline(method, prep, epochs=be[method], seed=seed).fit(data["X_train"], data["y_train"], log=log)
    elif method == "dgan":
        from .baselines.external import DGANBaseline
        m = DGANBaseline(T, d, epochs=be["dgan"], seed=seed).fit(data["X_train"], data["y_train"], log=log)
    elif method == "timegan":
        from .baselines.timegan import TimeGAN
        models = {}
        for k in range(K):
            Xk = data["X_train"][data["y_train"] == k]
            models[k] = TimeGAN(d, T, seed=seed).fit(Xk, iters=be["timegan_iters"], log=log)

        class _M:
            def sample_windows(self, labels):
                out = np.zeros((len(labels), T, d), np.float32)
                for k in np.unique(labels):
                    pos = np.where(labels == k)[0]; out[pos] = models[int(k)].sample(len(pos))
                return out
        m = _M()
    elif method == "tabddpm":
        from .baselines.ddpm import TabDDPM
        m = TabDDPM(d, K, T, steps=be.get("ddpm_steps", 1000), seed=seed).fit(data["X_train"].reshape(-1, d), np.repeat(data["y_train"], T),
                                            iters=be["ddpm_iters"], log=log)
    else:
        raise ValueError(method)
    info = {"seconds_total": time.time() - t0}
    return m.sample_windows, info


def get_judge(ds, data, cfg):
    path = os.path.join(cfg["paths"]["results_root"], ds, f"entropy_judge_T{data['X_train'].shape[1]}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    j = EntropyJudge(data["X_train"], data["y_train"], seed=0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(j, open(path, "wb"))
    return j


def run_one(ds, method, seed, data, prep, cfg, out_dir, eval_split="test", full_metrics=True, log=print,
            ids_kinds=("rf", "dnn")):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(10_000 + seed)
    T = prep.T
    K = len(prep.classes)
    res = {"dataset": ds, "method": method, "seed": seed, "eval_split": eval_split, "classes": prep.classes}
    counts = augmentation_counts(data["y_train"], prep.classes, cfg["eval"]["aug_factor"])
    if method in NON_GEN:
        if method == "none":
            Xa, ya = augmented_flows(data, {}, T)
        elif method == "ros":
            Xa, ya = augmented_flows(data, ros_windows(data, counts, seed), T)
        else:
            Xa, ya = smote_flows(data, counts, T, seed)
    else:
        sampler, info = build_sampler(method, data, prep, cfg, seed, out_dir, log)
        res["train_info"] = {k: v for k, v in info.items() if k != "config"}
        snap = lambda W: prep.snap(W) if len(W) else W
        Xe, ye = data[f"X_{eval_split}"], data[f"y_{eval_split}"]
        if full_metrics:
            n_syn = cfg["eval"]["n_syn_windows"]
            lab = rng.choice(ye, n_syn)                      # class proportions of the evaluation partition
            syn = snap(sampler(lab))
            np.savez_compressed(os.path.join(out_dir, "synthetic_sample.npz"), X=syn[:5000], y=lab[:5000])
            res["fidelity"] = fidelity(prep, Xe, syn, cfg["eval"]["ks_bootstrap"], cfg["eval"]["ks_subsample"], seed)
            res["temporal"] = temporal(prep, Xe, syn, seed=seed)
            res["discriminative_score"] = discriminative_score(Xe, syn, cfg["eval"]["disc_epochs"], seed)
            res["prdc"] = prdc(Xe, syn, cfg["eval"]["prdc_k"], cfg["eval"]["prdc_n"], seed)
            uni = np.repeat(np.arange(K), 1000)
            res["diversity"] = get_judge(ds, data, cfg)(snap(sampler(uni)), uni)
            res["tstr_macro_f1"] = tstr(syn, lab, Xe, ye, seed, cfg["eval"]["rf_trees"])
        else:
            n_val = min(len(Xe), 20000)
            sel = rng.choice(len(Xe), n_val, replace=False)
            syn = snap(sampler(ye[sel]))
            res["fidelity"] = fidelity(prep, Xe[sel], syn, 200, cfg["eval"]["ks_subsample"], seed)
            res["temporal"] = temporal(prep, Xe[sel], syn, seed=seed)
        extra = {k: (snap(sampler(np.full(n, k))) if n > 0 else None) for k, n in counts.items()}
        Xa, ya = augmented_flows(data, extra, T)
    res["augmentation_windows_added"] = {prep.classes[k]: int(n) for k, n in counts.items()} if method != "none" else {}
    res["ids"] = evaluate_ids(prep, data, Xa, ya, cfg, seed, eval_split)
    save_json(res, os.path.join(out_dir, "metrics.json"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


# --------------------------------------------------------------------------- cross-dataset transfer
_SYN = {"tot": "total", "totlen": "total length", "len": "length", "pkts": "packet", "pkt": "packet",
        "packets": "packet", "byts": "bytes", "cnt": "count", "var": "variance", "avg": "average",
        "seg": "segment", "backward": "bwd", "forward": "fwd", "blk": "bulk", "b": "bulk", "win": "win",
        "std": "std", "min": "min", "max": "max"}
_STOP = {"of", "s"}


def feature_key(name: str):
    toks = re.findall(r"[a-z]+", name.lower())
    out = []
    for t in toks:
        out += _SYN.get(t, t).split()
    return frozenset(t for t in out if t not in _STOP) | ({"per_s"} if "/s" in name.lower() else set())


_MAP18 = [(r"benign", "Benign"), (r"ddos|ddos attack|loic|hoic", "DDoS"), (r"hulk", "DoS Hulk"),
          (r"goldeneye", "DoS GoldenEye"), (r"slowloris", "DoS slowloris"), (r"slowhttptest", "DoS Slowhttptest"),
          (r"ftp", "FTP-Patator"), (r"ssh", "SSH-Patator"), (r"^bot$", "Bot")]


def map_target_class(name):
    for pat, tgt in _MAP18:
        if re.search(pat, name.lower()):
            return tgt
    return None


def transfer_setup(prep_s, prep_t):
    fmap = {}
    tkeys = {feature_key(c): j for j, c in enumerate(prep_t.feature_cols)}
    for i, c in enumerate(prep_s.feature_cols):
        k = feature_key(c)
        if k in tkeys:
            fmap[i] = tkeys[k]
    cls_s = [c for c in prep_s.classes if any(map_target_class(t) == c for t in prep_t.classes)]
    return fmap, cls_s


def _scale_subset(prep, O, idx):
    Z = O.copy()
    lm = prep.log_mask[idx]
    Z[:, lm] = np.sign(Z[:, lm]) * np.log1p(np.abs(Z[:, lm]))
    lo, hi = prep.lo[idx], prep.hi[idx]
    return np.clip(2 * (Z - lo) / np.where(hi > lo, hi - lo, 1) - 1, -1.5, 1.5).astype(np.float32)


def run_transfer(src, tgt, data_s, prep_s, data_t, prep_t, cfg, method, seed, results_root, log=print):
    fmap, shared = transfer_setup(prep_s, prep_t)
    si = sorted(fmap); ti = [fmap[i] for i in si]
    new_idx = {prep_s.classes.index(c): n for n, c in enumerate(shared)}
    T, d = prep_s.T, len(prep_s.feature_cols)

    def src_flows(X, y):
        keep = np.isin(y, list(new_idx))
        O = prep_s.inverse(X[keep]).reshape(-1, d)[:, si]
        return _scale_subset(prep_s, O, si), np.repeat(np.vectorize(new_idx.get)(y[keep]), T)
    Xs, ys = src_flows(data_s["X_train"], data_s["y_train"])
    if method != "none":
        G = Generator(d, T, len(prep_s.classes), **cfg["model"]).to(device())
        G.load_state_dict(torch.load(os.path.join(results_root, src, method, f"seed{seed}", "generator.pt"),
                                     map_location=device()))
        counts = augmentation_counts(data_s["y_train"], prep_s.classes, cfg["eval"]["aug_factor"])
        add_X, add_y = [], []
        for k, n in counts.items():
            if n > 0 and k in new_idx:
                W = prep_s.snap(generate(G, np.full(n, k), device(), seed=seed))
                Xa, ya = src_flows(W, np.full(n, k)); add_X.append(Xa); add_y.append(ya)
        if add_X:
            Xs, ys = np.concatenate([Xs] + add_X), np.concatenate([ys] + add_y)
    tmap = {j: new_idx[prep_s.classes.index(map_target_class(c))] for j, c in enumerate(prep_t.classes)
            if map_target_class(c) in shared}
    keep = np.isin(data_t["y_test"], list(tmap))
    Ot = prep_t.inverse(data_t["X_test"][keep]).reshape(-1, len(prep_t.feature_cols))[:, ti]
    Xt = _scale_subset(prep_s, Ot, si)
    yt = np.repeat(np.vectorize(tmap.get)(data_t["y_test"][keep]), T)
    from .ids import IDS, ids_report
    res = {"source": src, "target": tgt, "method": method, "seed": seed, "shared_classes": shared,
           "shared_features": [prep_s.feature_cols[i] for i in si],
           "unmatched_source_features": [c for i, c in enumerate(prep_s.feature_cols) if i not in fmap]}
    for kind in ("rf", "dnn"):
        clf = IDS(kind, len(shared), seed, cfg["eval"]["rf_trees"], cfg["eval"]["dnn_epochs"]).fit(Xs, ys)
        res[kind] = ids_report(yt, clf.predict(Xt), shared)
    save_json(res, os.path.join(results_root, "transfer", f"{src}_to_{tgt}", method, f"seed{seed}", "metrics.json"))
    return res
