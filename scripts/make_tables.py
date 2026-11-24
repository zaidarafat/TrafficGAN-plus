#!/usr/bin/env python
"""Builds every table and figure of Section 4 from results/ and writes results/paper_numbers.json.

  python scripts/make_tables.py [--config configs/default.yaml] [--primary cicids2017]
       [--extra unsw_nb15 cicids2018] [--transfer cicids2017_to_cicids2018]
Outputs: results/tables/*.md|.csv, results/figures/*.png, results/paper_numbers.json"""
# --- run from the package root, whatever the current directory ---
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_os.chdir(_ROOT)
if _ROOT not in _sys.path: _sys.path.insert(0, _ROOT)
import argparse, glob, json, os, re, sys
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tg.utils import load_cfg

NAMES = {"none": "None", "ros": "ROS", "smote": "SMOTE", "mlp_gan": "MLP-GAN", "mlp_wgan": "MLP WGAN-GP",
         "ctgan": "CTGAN", "copulagan": "CopulaGAN", "tabddpm": "TabDDPM-style DDPM", "timegan": "TimeGAN",
         "dgan": "DoppelGANger", "wgan_gp": "CNN-LSTM WGAN-GP", "trafficgan": "TrafficGAN+",
         "no_tgp": "Without TGP", "no_ase": "Without ASE", "alt_sn": "TGP → spectral normalization",
         "alt_lp": "TGP → one-sided GP (WGAN-LP)", "alt_acgan": "ASE → AC-GAN class loss",
         "alt_pacgan": "ASE → PacGAN (pack size 2)"}
GEN = ["mlp_gan", "mlp_wgan", "ctgan", "copulagan", "tabddpm", "timegan", "dgan", "wgan_gp", "trafficgan"]
ALL7 = ["none", "ros", "smote"] + GEN
ABL = ["trafficgan", "no_tgp", "no_ase", "wgan_gp", "alt_sn", "alt_lp", "alt_acgan", "alt_pacgan"]
FEATS = [("Flow duration", r"^flow duration$|^dur$"), ("Packet length (mean)", r"^packet length mean$|^pkt len mean$|^smeansz$"),
         ("FIN flag count", r"^fin flag count$|^fin flag cnt$|^proto$"), ("Flow IAT (mean)", r"^flow iat mean$|^sintpkt$"),
         ("Forward packets", r"^total fwd packets$|^tot fwd pkts$|^spkts$"), ("Flow bytes/s", r"^flow bytes/s$|^flow byts/s$|^sload$")]


def runs(R, ds, m):
    out = {}
    for f in glob.glob(os.path.join(R, ds, m, "seed*", "metrics.json")):
        out[int(re.search(r"seed(\d+)", f).group(1))] = json.load(open(f))
    return out


def ms(v, p=3):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return "–"
    return f"{v.mean():.{p}f} ± {v.std(ddof=1):.{p}f}" if len(v) > 1 else f"{v.mean():.{p}f}"


def mean(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def g(r, path):
    for k in path.split("."):
        if r is None or k not in r:
            return None
        r = r[k]
    return r


def rel_gain(rm, rn, classes, metric="recall"):
    gains = []
    for c in classes:
        if c == "Benign":
            continue
        a, b = rm["ids"]["rf"]["per_class"][c][metric], rn["ids"]["rf"]["per_class"][c][metric]
        if b > 0:
            gains.append((a - b) / b * 100)
    return float(np.mean(gains)) if gains else None


def paired(ref, other, f):
    seeds = sorted(set(ref) & set(other))
    a = np.array([f(ref[s]) for s in seeds], float); b = np.array([f(other[s]) for s in seeds], float)
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    if len(a) < 2:
        return dict(n=len(a), p_t=None, p_w=None, dz=None, diff=None, ci=None)
    d = a - b
    sd = d.std(ddof=1)
    p_t = stats.ttest_rel(a, b).pvalue if sd > 0 else 1.0
    try:
        p_w = stats.wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p_w = None
    h = stats.t.ppf(0.975, len(d) - 1) * sd / np.sqrt(len(d))
    return dict(n=len(d), p_t=float(p_t), p_w=None if p_w is None else float(p_w),
                dz=float(d.mean() / sd) if sd > 0 else None, diff=float(d.mean()), ci=[float(d.mean() - h), float(d.mean() + h)])


def holm(ps):
    idx = [i for i, p in enumerate(ps) if p is not None]
    order = sorted(idx, key=lambda i: ps[i]); m = len(order); adj = [None] * len(ps); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def write(df, name, outdir, caption):
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, name + ".csv"), index=False)
    with open(os.path.join(outdir, name + ".md"), "w") as fh:
        fh.write(f"**{caption}**\n\n" + df.to_markdown(index=False) + "\n")
    print(f"\n## {caption}\n" + df.to_markdown(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--primary", default="cicids2017")
    ap.add_argument("--extra", nargs="*", default=["unsw_nb15", "cicids2018"])
    ap.add_argument("--transfer", default="cicids2017_to_cicids2018")
    a = ap.parse_args()
    cfg = load_cfg(a.config); R = cfg["paths"]["results_root"]; P = a.primary
    TD, FD = os.path.join(R, "tables"), os.path.join(R, "figures")
    os.makedirs(FD, exist_ok=True)
    allr = {m: runs(R, P, m) for m in set(ALL7 + ABL)}
    any_run = next((list(v.values())[0] for v in allr.values() if v), None)
    if any_run is None:
        sys.exit("no results found")
    classes = any_run["classes"]
    nums = {"primary": P, "n_seeds": {m: len(v) for m, v in allr.items()}}

    # ---------------- Table 4: feature-wise fidelity
    rows = []; fig3 = []
    base, tg = allr["wgan_gp"], allr["trafficgan"]
    feat_names = list(next(iter(tg.values()))["fidelity"]["per_feature"]) if tg else []
    for label, pat in FEATS:
        f = next((n for n in feat_names if re.search(pat, n.strip().lower())), None)
        if f is None:
            continue
        b = [r["fidelity"]["per_feature"][f][1] for r in base.values()]
        t = [r["fidelity"]["per_feature"][f][1] for r in tg.values()]
        metric = next(iter(tg.values()))["fidelity"]["per_feature"][f][0]
        red = (mean(b) - mean(t)) / mean(b) * 100 if b and t and mean(b) else None
        rows.append([f"{label} ({f})", metric, ms(b), ms(t), f"{red:.1f}" if red is not None else "–"])
        fig3.append((label, mean(b), mean(t)))
    b = [r["fidelity"]["mean"] for r in base.values()]; t = [r["fidelity"]["mean"] for r in tg.values()]
    if b and t:
        rows.append(["Mean over all features", "KS/TVD", ms(b), ms(t), f"{(mean(b)-mean(t))/mean(b)*100:.1f}"])
        fig3.append(("Mean (all features)", mean(b), mean(t)))
        nums["mean_ks"] = {"wgan_gp": ms(b), "trafficgan": ms(t)}
    write(pd.DataFrame(rows, columns=["Feature", "Metric", "CNN-LSTM WGAN-GP", "TrafficGAN+", "Reduction (%)"]),
          "table4_fidelity", TD, "Table 4. Feature-wise fidelity (test partition; mean ± SD over seeds)")

    # ---------------- Table 5: temporal & sample-level fidelity
    rows = []
    for m in GEN:
        rr = allr[m].values()
        rows.append([NAMES[m], ms([g(r, "fidelity.mean") for r in rr]), ms([g(r, "temporal.increment_ks") for r in rr]),
                     ms([g(r, "temporal.acf_error") for r in rr]), ms([g(r, "discriminative_score") for r in rr]),
                     ms([g(r, "prdc.precision") for r in rr]), ms([g(r, "prdc.recall") for r in rr]),
                     ms([g(r, "prdc.density") for r in rr]), ms([g(r, "prdc.coverage") for r in rr])])
    write(pd.DataFrame(rows, columns=["Method", "Mean KS/TVD ↓", "Increment-KS ↓", "ACF-error ↓", "Discriminative score ↓",
                                      "Precision ↑", "Recall ↑", "Density", "Coverage ↑"]),
          "table5_temporal", TD, "Table 5. Temporal and sample-level fidelity")

    # ---------------- Section 4.2: diversity
    rows = []
    for m in GEN:
        rr = allr[m].values()
        rows.append([NAMES[m], ms([g(r, "diversity.class_entropy_bits") for r in rr], 2),
                     ms([g(r, "diversity.class_entropy_normalized") for r in rr], 2),
                     ms([g(r, "diversity.label_agreement") for r in rr], 2)])
    kmax = g(next(iter(tg.values()), None), "diversity.max_entropy_bits") if tg else None
    write(pd.DataFrame(rows, columns=["Method", "Class entropy (bits) ↑", "Normalized entropy ↑", "Label agreement ↑"]),
          "table_diversity", TD, f"Class diversity of uniformly conditioned batches (maximum {kmax:.2f} bits)" if kmax else "Class diversity")
    nums["entropy"] = {m: ms([g(r, "diversity.class_entropy_bits") for r in allr[m].values()], 2) for m in ("wgan_gp", "trafficgan")}
    nums["entropy_max_bits"] = kmax

    # ---------------- Table 6: per-class RF, none vs TrafficGAN+
    rows = []; none = allr["none"]
    per = {}
    for c in classes:
        f0 = [r["ids"]["rf"]["per_class"][c]["f1"] for r in none.values()]
        f1 = [r["ids"]["rf"]["per_class"][c]["f1"] for r in tg.values()]
        seeds = sorted(set(none) & set(tg))
        rg = [(tg[s]["ids"]["rf"]["per_class"][c]["recall"] - none[s]["ids"]["rf"]["per_class"][c]["recall"]) /
              max(none[s]["ids"]["rf"]["per_class"][c]["recall"], 1e-9) * 100 for s in seeds] if c != "Benign" else []
        fc = (mean(f1) - mean(f0)) / mean(f0) * 100 if f0 and f1 and mean(f0) else None
        rows.append([c, ms(f0), ms(f1), ms(rg, 1) if rg else "–", f"{fc:+.1f}" if fc is not None else "–"])
        per[c] = rg
    m0 = [r["ids"]["rf"]["macro_f1"] for r in none.values()]; m1 = [r["ids"]["rf"]["macro_f1"] for r in tg.values()]
    seeds = sorted(set(none) & set(tg))
    mg = [rel_gain(tg[s], none[s], classes) for s in seeds]
    rows.append(["Macro average", ms(m0), ms(m1), ms(mg, 1) + " (attack classes)" if mg else "–",
                 f"{(mean(m1)-mean(m0))/mean(m0)*100:+.1f}" if m0 and m1 else "–"])
    write(pd.DataFrame(rows, columns=["Class", "F1 (no augmentation)", "F1 (TrafficGAN+ augmentation)",
                                      "Relative recall gain (%)", "F1 change (%)"]),
          "table6_perclass", TD, "Table 6. Per-class random-forest results on the test partition")
    nums["rf_macro_f1"] = {"none": ms(m0), "trafficgan": ms(m1)}
    fig5 = (classes, [mean([r["ids"]["rf"]["per_class"][c]["f1"] for r in none.values()]) for c in classes],
            [mean([r["ids"]["rf"]["per_class"][c]["f1"] for r in tg.values()]) for c in classes],
            [mean(per[c]) if per[c] else None for c in classes])

    # ---------------- Table 7: utility, all methods
    rows = []
    for m in ALL7:
        rr = allr[m].values()
        rows.append([NAMES[m], ms([g(r, "ids.rf.macro_f1") for r in rr]), ms([g(r, "ids.rf.attack_macro_recall") for r in rr]),
                     ms([g(r, "ids.dnn.macro_f1") for r in rr]), ms([g(r, "ids.dnn.attack_macro_recall") for r in rr]),
                     ms([g(r, "tstr_macro_f1") for r in rr]) if m in GEN else "n/a"])
    write(pd.DataFrame(rows, columns=["Augmentation", "RF macro-F1", "RF attack recall", "DNN macro-F1",
                                      "DNN attack recall", "TSTR macro-F1"]), "table7_utility", TD,
          "Table 7. Downstream utility (test partition)")

    # ---------------- Table 8: ablation + alternatives, with paired tests vs TrafficGAN+
    rows, tests = [], []
    for m in ABL:
        rr = allr[m]
        gains = [rel_gain(rr[s], none[s], classes) for s in sorted(set(rr) & set(none))]
        rows.append([NAMES[m] if m != "wgan_gp" else "Without TGP and ASE (CNN-LSTM WGAN-GP)",
                     ms([g(r, "fidelity.mean") for r in rr.values()]), ms([g(r, "ids.rf.macro_f1") for r in rr.values()]),
                     ms(gains, 1), ms([g(r, "diversity.class_entropy_bits") for r in rr.values()], 2)])
    for m in ABL[1:] + [x for x in GEN if x not in ABL]:
        for metric, f in [("mean KS", lambda r: g(r, "fidelity.mean")), ("RF macro-F1", lambda r: g(r, "ids.rf.macro_f1"))]:
            st = paired(tg, allr[m], f); tests.append([NAMES[m], metric, st])
    adj = holm([t[2]["p_t"] for t in tests])
    trows = [[n, met, s["n"], f"{s['diff']:+.4f}" if s["diff"] is not None else "–",
              f"[{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]" if s["ci"] else "–",
              f"{s['p_t']:.4f}" if s["p_t"] is not None else "–", f"{p:.4f}" if p is not None else "–",
              f"{s['p_w']:.4f}" if s["p_w"] is not None else "–", f"{s['dz']:.2f}" if s["dz"] is not None else "–"]
             for (n, met, s), p in zip(tests, adj)]
    write(pd.DataFrame(rows, columns=["Variant", "Mean KS/TVD ↓", "RF macro-F1 ↑", "Relative recall gain, attack classes (%) ↑",
                                      "Class entropy (bits) ↑"]), "table8_ablation", TD, "Table 8. Ablation and alternative regularizers")
    write(pd.DataFrame(trows, columns=["Compared with TrafficGAN+", "Metric", "n seeds", "Mean diff. (TrafficGAN+ − other)",
                                       "95% CI", "p (paired t)", "p (Holm)", "p (Wilcoxon)", "Cohen's d_z"]),
          "table_stats", TD, "Paired comparisons with TrafficGAN+ (Holm-adjusted over all rows)")
    fig6 = [(NAMES[m] if m != "wgan_gp" else "WGAN-GP", [g(r, "fidelity.mean") for r in allr[m].values()],
             [g(r, "ids.rf.macro_f1") for r in allr[m].values()]) for m in ABL[:4]]

    # ---------------- Table 9: other datasets + transfer
    rows = []
    for ds in a.extra:
        rs = {m: runs(R, ds, m) for m in ["none", "smote", "ctgan", "timegan", "dgan", "wgan_gp", "trafficgan"]}
        if not any(rs.values()):
            continue
        seq = max(["timegan", "dgan"], key=lambda m: mean([g(r, "ids.rf.macro_f1") for r in rs[m].values()]) or -1)
        for m in ["none", "smote", "ctgan", seq, "wgan_gp", "trafficgan"]:
            rr = rs[m].values()
            rows.append([ds, NAMES[m] + (" (best sequence baseline)" if m == seq else ""),
                         ms([g(r, "fidelity.mean") for r in rr]), ms([g(r, "temporal.increment_ks") for r in rr]),
                         ms([g(r, "diversity.class_entropy_bits") for r in rr], 2),
                         ms([g(r, "ids.rf.macro_f1") for r in rr]), ms([g(r, "ids.dnn.macro_f1") for r in rr])])
    for m in ["none", "wgan_gp", "trafficgan"]:
        fs = glob.glob(os.path.join(R, "transfer", a.transfer, m, "seed*", "metrics.json"))
        if fs:
            rr = [json.load(open(f)) for f in fs]
            rows.append([a.transfer.replace("_to_", " → "), NAMES[m], "n/a", "n/a", "n/a",
                         ms([r["rf"]["macro_f1"] for r in rr]), ms([r["dnn"]["macro_f1"] for r in rr])])
    if rows:
        write(pd.DataFrame(rows, columns=["Dataset", "Method", "Mean KS/TVD ↓", "Increment-KS ↓", "Class entropy (bits) ↑",
                                          "RF macro-F1 ↑", "DNN macro-F1 ↑"]), "table9_datasets", TD, "Table 9. Additional datasets")

    # ---------------- Robustness
    rows = []
    for m in ["none", "smote", "wgan_gp", "trafficgan"]:
        rr = allr[m].values()
        rows.append([NAMES[m]] + [ms([g(r, f"ids.{k}.{x}") for r in rr]) for k in ("rf", "dnn")
                                  for x in ("attack_macro_recall", "perturbed_attack_macro_recall", "robustness_ratio")])
    write(pd.DataFrame(rows, columns=["Augmentation", "RF recall (clean)", "RF recall (±5% timing)", "RF ratio",
                                      "DNN recall (clean)", "DNN recall (±5% timing)", "DNN ratio"]),
          "table_robustness", TD, "Attack macro-recall under ±5% timing perturbation")

    # ---------------- Table 10: cost
    rows = []
    for m, lab in [("wgan_gp", "CNN-LSTM WGAN-GP"), ("no_ase", "+ TGP"), ("trafficgan", "+ TGP + ASE (TrafficGAN+)")]:
        ti = [r.get("train_info", {}) for r in allr[m].values()]
        rows.append([lab, ms([x.get("seconds_per_iter") for x in ti], 3), ms([x.get("best_iter") for x in ti], 0),
                     ms([x.get("peak_gpu_mem_gb") for x in ti], 2), ms([x.get("gen_ms_per_window") for x in ti], 3)])
    write(pd.DataFrame(rows, columns=["Variant", "Time per generator iteration (s)", "Iteration of best checkpoint",
                                      "Peak GPU memory (GB)", "Generation time per window (ms)"]), "table10_cost", TD,
          "Table 10. Computational cost")

    # ---------------- Sensitivity
    rows = []
    for d in sorted(glob.glob(os.path.join(R, P, "sensitivity", "*"))):
        rr = [json.load(open(f)) for f in glob.glob(os.path.join(d, "seed*", "metrics.json"))]
        if rr:
            rows.append([os.path.basename(d), len(rr), ms([g(r, "fidelity.mean") for r in rr]),
                         ms([g(r, "temporal.increment_ks") for r in rr]), ms([g(r, "ids.rf.macro_f1") for r in rr])])
    if rows:
        write(pd.DataFrame(rows, columns=["Setting", "Seeds", "Val. mean KS/TVD", "Val. increment-KS", "Val. RF macro-F1"]),
              "table_sensitivity", TD, "Sensitivity analysis (validation partition)")

    # ---------------- figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    C1, C2 = "#E2672E", "#2A73C9"
    if fig3:
        fig, ax = plt.subplots(figsize=(9, 3.6)); x = np.arange(len(fig3)); w = 0.38
        ax.bar(x - w / 2, [v[1] for v in fig3], w, color=C1, label="CNN-LSTM WGAN-GP")
        ax.bar(x + w / 2, [v[2] for v in fig3], w, color=C2, label="TrafficGAN+")
        ax.set_xticks(x); ax.set_xticklabels([v[0] for v in fig3], fontsize=8); ax.set_ylabel("KS / TVD (lower is better)")
        ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(FD, "fig3_fidelity.png"), dpi=220)
    if tg and base:
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        vals = [[g(r, "diversity.class_entropy_bits") for r in base.values()], [g(r, "diversity.class_entropy_bits") for r in tg.values()]]
        ax.bar([0, 1], [mean(v) for v in vals], yerr=[np.std(v, ddof=1) if len(v) > 1 else 0 for v in vals], color=[C1, C2], capsize=5, width=0.55)
        if kmax:
            ax.axhline(kmax, ls="--", color="#666", lw=1); ax.text(1.32, kmax, f"max = log2 K = {kmax:.2f}", va="center", fontsize=8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["CNN-LSTM\nWGAN-GP", "TrafficGAN+"]); ax.set_ylabel("Class entropy (bits)")
        ax.set_xlim(-0.6, 2.2); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(FD, "fig4_entropy.png"), dpi=220)
    cl, f0, f1, rg = fig5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(cl)); w = 0.38
    ax1.bar(x - w / 2, [v or 0 for v in f0], w, color=C1, label="No augmentation"); ax1.bar(x + w / 2, [v or 0 for v in f1], w, color=C2, label="TrafficGAN+ augmentation")
    ax1.set_xticks(x); ax1.set_xticklabels(cl, rotation=30, ha="right", fontsize=8); ax1.set_ylabel("F1 (random forest)"); ax1.set_ylim(0, 1.05); ax1.legend(frameon=False, fontsize=8)
    att = [(c, v) for c, v in zip(cl, rg) if v is not None]
    ax2.bar(range(len(att)), [v for _, v in att], color="#1D9E73"); ax2.set_xticks(range(len(att))); ax2.set_xticklabels([c for c, _ in att], rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Relative recall gain (%)"); ax2.axhline(0, color="#444", lw=0.8)
    for a_ in (ax1, ax2): a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FD, "fig5_perclass.png"), dpi=220)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, k, lab in [(ax1, 1, "Mean KS/TVD (lower is better)"), (ax2, 2, "RF macro-F1 after augmentation")]:
        vals = [v[k] for v in fig6]
        ax.bar(range(4), [mean(v) or 0 for v in vals], yerr=[np.std(v, ddof=1) if len(v) > 1 else 0 for v in vals], color=C2, capsize=4)
        ax.set_xticks(range(4)); ax.set_xticklabels([v[0] for v in fig6], fontsize=8); ax.set_ylabel(lab); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FD, "fig6_ablation.png"), dpi=220)

    tables = {}
    for f in sorted(glob.glob(os.path.join(TD, "*.csv"))):
        tables[os.path.basename(f)[:-4]] = pd.read_csv(f).fillna("").astype(str).values.tolist()
    nums["tables"] = tables
    json.dump(nums, open(os.path.join(R, "paper_numbers.json"), "w"), indent=1, default=str)
    print(f"\nwrote {TD}, {FD} and {os.path.join(R, 'paper_numbers.json')}")


if __name__ == "__main__":
    main()
