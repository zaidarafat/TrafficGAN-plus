#!/usr/bin/env python
"""Zips everything needed to fill the manuscript: tables, figures, paper_numbers.json, dataset summaries,
train_info/log files and any ERROR.txt. Model checkpoints and synthetic samples are excluded."""
# --- run from the package root, whatever the current directory ---
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_os.chdir(_ROOT)
if _ROOT not in _sys.path: _sys.path.insert(0, _ROOT)
import glob, os, sys, zipfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tg.utils import load_cfg
cfg = load_cfg(sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml")
R, Pd = cfg["paths"]["results_root"], cfg["paths"]["processed_root"]
pats = [f"{R}/paper_numbers.json", f"{R}/tables/*", f"{R}/figures/*", f"{Pd}/*_summary.json",
        f"{R}/**/metrics.json", f"{R}/**/train_info.json", f"{R}/**/log.csv", f"{R}/**/ERROR.txt", "configs/default.yaml"]
if not os.path.exists(f"{R}/paper_numbers.json"):
    sys.exit("No results yet: run the experiments and scripts/make_tables.py first (see README section 3).")
out = "results_for_paper.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in pats:
        for f in glob.glob(p, recursive=True):
            z.write(f)
print("wrote", out, f"({os.path.getsize(out)/1e6:.1f} MB)")
