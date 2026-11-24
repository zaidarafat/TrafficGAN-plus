#!/usr/bin/env python
"""Command-line entry point. Every stage skips runs whose metrics.json already exists (resumable).

  python scripts/run_experiments.py prepare     --dataset cicids2017 [--T 10]
  python scripts/run_experiments.py main        --dataset cicids2017 [--methods ...] [--seeds ...]
  python scripts/run_experiments.py sensitivity --dataset cicids2017
  python scripts/run_experiments.py transfer    --source cicids2017 --target cicids2018
  python scripts/make_tables.py
Add  --set train.n_iters=200 eval.ks_bootstrap=20 ...  to override configuration values."""
# --- run from the package root, whatever the current directory ---
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_os.chdir(_ROOT)
if _ROOT not in _sys.path: _sys.path.insert(0, _ROOT)
import argparse, os, sys, traceback, ast
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tg.utils import load_cfg, load_json
from tg.data import prepare, load_prepared
from tg.experiment import ALL_METHODS, run_one, run_transfer


def parse_sets(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = ast.literal_eval(v)
        except Exception:
            pass
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "main", "sensitivity", "transfer"])
    ap.add_argument("--dataset"); ap.add_argument("--source"); ap.add_argument("--target")
    ap.add_argument("--methods", nargs="*"); ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--T", type=int); ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", dest="sets")
    ap.add_argument("--data-dir", help="folder with the original CSVs of --dataset (overrides the config)")
    a = ap.parse_args()
    cfg = load_cfg(a.config, parse_sets(a.sets))
    R = cfg["paths"]["results_root"]

    if a.data_dir:
        cfg["paths"].setdefault("datasets", {})[a.dataset] = a.data_dir
    if a.stage == "prepare":
        Ts = [a.T] if a.T else [cfg["data"]["T"]]
        for T in Ts:
            print(f"== preparing {a.dataset} T={T}")
            prepare(a.dataset, cfg, cfg["paths"]["processed_root"], T)
        return

    if a.stage == "main":
        data, prep = load_prepared(a.dataset, cfg)
        methods = a.methods or ALL_METHODS
        seeds = a.seeds if a.seeds is not None else cfg["experiments"]["seeds"]
        for seed in seeds:
            for m in methods:
                out = os.path.join(R, a.dataset, m, f"seed{seed}")
                if os.path.exists(os.path.join(out, "metrics.json")):
                    continue
                print(f"== {a.dataset} {m} seed {seed}", flush=True)
                try:
                    run_one(a.dataset, m, seed, data, prep, cfg, out)
                except Exception:
                    traceback.print_exc()
                    os.makedirs(out, exist_ok=True)
                    open(os.path.join(out, "ERROR.txt"), "w").write(traceback.format_exc())
        return

    if a.stage == "sensitivity":
        ex = cfg["experiments"]
        runs = [(f"gamma_{g}", {"train.gamma_tgp": g}, None) for g in ex["gamma_grid"]]
        runs += [(f"beta_{b}", {"train.beta_ase": b}, None) for b in ex["beta_grid"]]
        runs += [("tgp_onesided", {"train.tgp_one_sided": True}, None)]
        runs += [(f"T_{t}", {}, t) for t in ex["T_grid"]]
        seeds = a.seeds if a.seeds is not None else ex["sensitivity_seeds"]
        for name, ov, T in runs:
            c2 = load_cfg(a.config, {**parse_sets(a.sets), **ov})
            try:
                data, prep = load_prepared(a.dataset, c2, T)
            except FileNotFoundError:
                print(f"== preparing {a.dataset} T={T}"); prepare(a.dataset, c2, c2["paths"]["processed_root"], T)
                data, prep = load_prepared(a.dataset, c2, T)
            for seed in seeds:
                out = os.path.join(R, a.dataset, "sensitivity", name, f"seed{seed}")
                if os.path.exists(os.path.join(out, "metrics.json")):
                    continue
                print(f"== sensitivity {name} seed {seed}", flush=True)
                try:
                    run_one(a.dataset, "trafficgan", seed, data, prep, c2, out, eval_split="val",
                            full_metrics=False)
                except Exception:
                    traceback.print_exc()
        return

    if a.stage == "transfer":
        ds, ps = load_prepared(a.source, cfg)
        dt, pt = load_prepared(a.target, cfg)
        seeds = a.seeds if a.seeds is not None else cfg["experiments"]["seeds"]
        for seed in seeds:
            for m in (a.methods or ["none", "wgan_gp", "trafficgan"]):
                out = os.path.join(R, "transfer", f"{a.source}_to_{a.target}", m, f"seed{seed}", "metrics.json")
                if os.path.exists(out):
                    continue
                print(f"== transfer {m} seed {seed}", flush=True)
                run_transfer(a.source, a.target, ds, ps, dt, pt, cfg, m, seed, R)


if __name__ == "__main__":
    main()
