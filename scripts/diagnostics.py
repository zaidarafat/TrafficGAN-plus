#!/usr/bin/env python
"""Diagnostic analyses of Sections 3.10 and 4.7 (Table 11 and Tables S6-S9). Needs only numpy, pandas and the saved results; no GPU.

  python scripts/diagnostics.py [--config configs/default.yaml] [--datasets cicids2017 unsw_nb15 cicids2018]

  S6  ASE proxy saturation: entropy (nats) of the batch-marginal prediction of the auxiliary head on generated
      batches at the selected checkpoint (log.csv), against its maximum ln K and against the class entropy
      measured by the independent random forest (metrics.json).
  S7  Exact repeats in real windows: per class, the share of continuous-feature increments that are exactly zero
      and the share of consecutive flow pairs that are identical in all features (test partition).
  S8  Zero-increment bound on increment-KS: for every generator with a saved synthetic sample, the observed
      increment-KS on that sample and the part of it that is forced by the zero atom alone,
      LB_j = max(|F_r(0) - F_s(0)|, |F_r(0-) - F_s(0-)|) >= |a_r,j - a_s,j| / 2, averaged over continuous features.
      The real-vs-real floor (training windows resampled to the test class proportions) is reported as reference.
  T11 Summary of S6 and S8 for the CNN-LSTM WGAN-GP and TrafficGAN+ (Table 11 of the paper).
  S9  TGP activity (runs trained with this package version or later; see the 'backbone' and 'diagnostic' stages):
      increment-gradient norms ||g_t|| at the interpolates, relative to the global gradient norm."""
# --- run from the package root, whatever the current directory ---
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_os.chdir(_ROOT)
if _ROOT not in _sys.path: _sys.path.insert(0, _ROOT)
import argparse, csv, glob, json, math, os, pickle
import numpy as np
import pandas as pd

R = "results"
P = os.path.join("data", "processed")
NAMES = {"mlp_gan": "MLP-GAN", "mlp_wgan": "MLP WGAN-GP", "ctgan": "CTGAN", "copulagan": "CopulaGAN",
         "tabddpm": "TabDDPM-style", "timegan": "TimeGAN", "dgan": "DoppelGANger", "wgan_gp": "CNN-LSTM WGAN-GP",
         "trafficgan": "TrafficGAN+", "no_tgp": "Without TGP", "no_ase": "Without ASE",
         "alt_sn": "TGP → spectral normalization", "alt_lp": "TGP → one-sided GP", "alt_acgan": "ASE → AC-GAN loss",
         "alt_pacgan": "ASE → PacGAN", "tr_wgan_gp": "Transformer WGAN-GP", "tr_tgp": "Transformer + TGP",
         "tr_ase": "Transformer + ASE", "tr_trafficgan": "Transformer + TGP + ASE",
         "diag_wgan_gp": "CNN-LSTM WGAN-GP (diagnostic run)", "diag_trafficgan": "TrafficGAN+ (diagnostic run)"}
DS = {"cicids2017": "CIC-IDS-2017", "unsw_nb15": "UNSW-NB15", "cicids2018": "CSE-CIC-IDS2018"}
GEN = ["mlp_gan", "mlp_wgan", "ctgan", "copulagan", "tabddpm", "timegan", "dgan", "wgan_gp", "trafficgan",
       "tr_wgan_gp", "tr_tgp", "tr_ase", "tr_trafficgan"]
CNN = ["trafficgan", "no_tgp", "no_ase", "wgan_gp", "alt_sn", "alt_lp", "alt_acgan", "alt_pacgan",
       "tr_wgan_gp", "tr_tgp", "tr_ase", "tr_trafficgan"]
EPS = 1e-9


def ms(v, p=3):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return "–"
    return f"{np.mean(v):.{p}f} ± {np.std(v, ddof=1):.{p}f}" if len(v) > 1 else f"{v[0]:.{p}f}"


def write(df, name, caption):
    os.makedirs(os.path.join(R, "tables"), exist_ok=True)
    df.to_csv(os.path.join(R, "tables", name + ".csv"), index=False)
    with open(os.path.join(R, "tables", name + ".md"), "w", encoding="utf-8") as fh:
        fh.write(f"**{caption}**\n\n" + df.to_markdown(index=False) + "\n")
    print(f"\n**{caption}**\n" + df.to_markdown(index=False))


def model_dirs(ds, m):
    base = os.path.join(R, ds, "diagnostic" if m.startswith("diag_") else "", m[5:] if m.startswith("diag_") else m)
    return sorted(glob.glob(os.path.join(base, "seed*")))


def best_row(d):
    try:
        rows = list(csv.DictReader(open(os.path.join(d, "log.csv"))))
        best = json.load(open(os.path.join(d, "train_info.json")))["best_iter"]
    except (OSError, KeyError, ValueError):
        return None
    r = [x for x in rows if int(x["iter"]) == int(best)]
    return r[0] if r else None


SUMMARY = {}


def s6_ase(datasets):
    out = []
    for ds in datasets:
        for m in CNN + ["diag_wgan_gp", "diag_trafficgan"]:
            a, h, ent, agr, K = [], [], [], [], None
            for d in model_dirs(ds, m):
                r = best_row(d)
                if r is None or "ase_nats" not in r:
                    continue
                info = json.load(open(os.path.join(d, "train_info.json")))
                K = info.get("K", K)
                a.append(float(r["ase_nats"]))
                mp = os.path.join(d, "metrics.json")
                if os.path.exists(mp):
                    mt = json.load(open(mp)).get("diversity") or {}
                    if "class_entropy_bits" in mt:
                        ent.append(mt["class_entropy_bits"]); agr.append(mt["label_agreement"])
            if not a:
                continue
            SUMMARY.setdefault(ds, {})[("ase", m)] = (100 * np.mean(a) / math.log(K), 100 * np.mean(ent) / math.log2(K) if ent else np.nan)
            out.append({"Dataset": DS[ds], "Variant": NAMES.get(m, m), "n seeds": len(a),
                        "ASE at checkpoint (nats)": ms(a), "Share of ln K (%)": f"{100 * np.mean(a) / math.log(K):.1f}",
                        "RF class entropy (bits)": ms(ent, 2),
                        "Share of log2 K (%)": f"{100 * np.mean(ent) / math.log2(K):.1f}" if ent else "–",
                        "Label agreement": ms(agr, 2)})
    if out:
        write(pd.DataFrame(out), "tableS6_ase_saturation",
              "Table S6. ASE proxy at the selected checkpoint versus the independent random-forest judge")


def load(ds):
    z = np.load(os.path.join(P, f"{ds}_T10.npz"))
    prep_p = os.path.join(P, f"{ds}_T10_prep.pkl")
    if os.path.exists(prep_p):
        prep = pickle.load(open(prep_p, "rb"))
        classes, disc = list(prep.classes), np.asarray(prep.discrete, bool)
    else:                                          # summary JSON is enough for these diagnostics
        s = json.load(open(os.path.join(P, f"{ds}_T10_summary.json")))
        classes = s["classes"]; disc = np.isin(s["features"], s["discrete"])
    return z, classes, np.where(~disc)[0]


def increments(X):
    return (X[:, 1:] - X[:, :-1]).reshape(-1, X.shape[-1])


def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    v = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, v, "right") / len(a) - np.searchsorted(b, v, "right") / len(b))))


def atom_bound(a, b):
    """max(|F_a(0) - F_b(0)|, |F_a(0-) - F_b(0-)|): the KS distance forced by the jump at zero."""
    fa0, fb0 = np.mean(a <= EPS), np.mean(b <= EPS)
    fa_, fb_ = np.mean(a < -EPS), np.mean(b < -EPS)
    return max(abs(fa0 - fb0), abs(fa_ - fb_))


def s7_s8(datasets, max_inc=200_000):
    rows7, rows8 = [], []
    for ds in datasets:
        try:
            z, classes, cont = load(ds)
        except OSError:
            print(f"  {ds}: processed data not found, skipped"); continue
        Xte, yte, Xtr, ytr = z["X_test"], z["y_test"], z["X_train"], z["y_train"]
        for k, c in enumerate(list(classes) + ["All classes"]):
            X = Xte if c == "All classes" else Xte[yte == k]
            if len(X) == 0:
                continue
            D = X[:, 1:] - X[:, :-1]
            rows7.append({"Dataset": DS[ds], "Class": c, "Test windows": len(X),
                          "Zero increments, continuous features (%)": f"{100 * (np.abs(D[:, :, cont]) < EPS).mean():.1f}",
                          "Identical consecutive flows (%)": f"{100 * np.all(np.abs(D) < EPS, axis=2).mean():.1f}"})
        rng = np.random.default_rng(0)
        dr = increments(Xte)
        dr = dr[rng.choice(len(dr), min(max_inc, len(dr)), replace=False)]

        def row(name, syn, n):
            ds_ = increments(syn)
            ds_ = ds_[rng.choice(len(ds_), min(max_inc, len(ds_)), replace=False)]
            inc = [ks(dr[:, j], ds_[:, j]) for j in cont]
            lb = [atom_bound(dr[:, j], ds_[:, j]) for j in cont]
            return {"Dataset": DS[ds], "Generator": name, "n samples": n,
                    "Zero increments, synthetic (%)": 100 * float((np.abs(ds_[:, cont]) < EPS).mean()),
                    "Increment-KS": float(np.mean(inc)), "Part forced by zero atom": float(np.mean(lb))}
        idx = np.concatenate([rng.choice(np.where(ytr == k)[0], int(np.sum(yte == k)), replace=True)
                              for k in range(len(classes)) if np.sum(yte == k) > 0 and np.any(ytr == k)])
        SUMMARY.setdefault(ds, {})["zero"] = 100 * float((np.abs(dr[:, cont]) < EPS).mean())
        r = row("Real training windows (floor)", Xtr[idx], 1)
        rows8.append(r)
        for m in GEN:
            vals = []
            for d in model_dirs(ds, m):
                f = os.path.join(d, "synthetic_sample.npz")
                if os.path.exists(f):
                    vals.append(row(NAMES[m], np.load(f)["X"], 1))
            if vals:
                agg = {k: vals[0][k] for k in ("Dataset", "Generator")}
                agg["n samples"] = len(vals)
                for k in ("Zero increments, synthetic (%)", "Increment-KS", "Part forced by zero atom"):
                    agg[k] = float(np.mean([v[k] for v in vals]))
                rows8.append(agg)
                SUMMARY.setdefault(ds, {})[("atom", m)] = (agg["Increment-KS"], agg["Part forced by zero atom"])
        print(f"  {ds}: real zero increments {100 * (np.abs(dr[:, cont]) < EPS).mean():.1f}%")
    if rows7:
        write(pd.DataFrame(rows7), "tableS7_zero_increments",
              "Table S7. Exact repeats in real test windows (T = 10)")
    if rows8:
        df = pd.DataFrame(rows8)
        df["Share of increment-KS (%)"] = (100 * df["Part forced by zero atom"] / df["Increment-KS"]).round(0)
        for k in ("Zero increments, synthetic (%)",):
            df[k] = df[k].map(lambda v: f"{v:.1f}")
        for k in ("Increment-KS", "Part forced by zero atom"):
            df[k] = df[k].map(lambda v: f"{v:.3f}")
        write(df, "tableS8_zero_atom_bound",
              "Table S8. Increment-KS and the part of it forced by the zero-increment atom (one synthetic sample "
              "of 5,000 windows per seed, averaged over seeds)")


def t11(datasets):
    rows = []
    for ds in datasets:
        S = SUMMARY.get(ds, {})
        if ("ase", "wgan_gp") not in S or ("ase", "trafficgan") not in S or "zero" not in S:
            continue
        sh = [100 * v[1] / v[0] for key, v in S.items() if isinstance(key, tuple) and key[0] == "atom"]
        tg = S.get(("atom", "trafficgan"))
        rows.append({"Dataset": DS[ds],
                     "ASE proxy, WGAN-GP / TrafficGAN+ (%)": f"{S[('ase', 'wgan_gp')][0]:.1f} / {S[('ase', 'trafficgan')][0]:.1f}",
                     "RF entropy, WGAN-GP / TrafficGAN+ (%)": f"{S[('ase', 'wgan_gp')][1]:.1f} / {S[('ase', 'trafficgan')][1]:.1f}",
                     "Zero increments in real test windows (%)": f"{S['zero']:.1f}",
                     "Increment-KS of TrafficGAN+ (zero-atom part)": f"{tg[0]:.3f} ({tg[1]:.3f})" if tg else "–",
                     "Zero-atom share, all generators (%)": f"{min(sh):.0f}–{max(sh):.0f}" if sh else "–"})
    if rows:
        write(pd.DataFrame(rows), "table11_diagnostics", "Table 11. Diagnostic summary (details in Tables S6–S8)")


def s9_tgp(datasets):
    out = []
    cols = ["gp_norm", "g_mean", "g_max", "g1_over_grad", "g_frac_above1", "tgp_value"]
    for ds in datasets:
        for m in CNN + ["diag_wgan_gp", "diag_trafficgan"]:
            vals = {c: [] for c in cols}
            n = 0
            for d in model_dirs(ds, m):
                try:
                    rows = list(csv.DictReader(open(os.path.join(d, "log.csv"))))
                except OSError:
                    continue
                if not rows or "g_mean" not in rows[0]:
                    continue
                n += 1
                for c in cols:            # average over the training checks of the run
                    vals[c].append(float(np.mean([float(r[c]) for r in rows])))
            if n:
                out.append({"Dataset": DS[ds], "Variant": NAMES.get(m, m), "n seeds": n,
                            "||∇D|| (GP target 1)": ms(vals["gp_norm"]), "mean_t ||g_t||": ms(vals["g_mean"]),
                            "max_t ||g_t||": ms(vals["g_max"]),
                            "||g_1|| / ||∇D|| (bound √(T−1) = 3)": ms(vals["g1_over_grad"]),
                            "Share ||g_t|| > 1 (%)": ms([100 * v for v in vals["g_frac_above1"]], 1),
                            "TGP value": ms(vals["tgp_value"], 4)})
    if out:
        write(pd.DataFrame(out), "tableS9_tgp_activity",
              "Table S9. Increment-gradient norms at the interpolates (mean over training checks)")
    else:
        print("\n(S9 skipped: no run logs with TGP statistics yet; run the 'backbone' or 'diagnostic' stage first)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--datasets", nargs="*", default=["cicids2017", "unsw_nb15", "cicids2018"])
    a = ap.parse_args()
    import yaml
    global R, P
    cfg = yaml.safe_load(open(a.config))
    R, P = cfg["paths"]["results_root"], cfg["paths"]["processed_root"]
    s6_ase(a.datasets)
    s7_s8(a.datasets)
    t11(a.datasets)
    s9_tgp(a.datasets)


if __name__ == "__main__":
    main()
