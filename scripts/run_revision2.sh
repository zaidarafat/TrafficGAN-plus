#!/usr/bin/env bash
# Second revision: stronger-backbone study (Section 4.8) and TGP-activity runs (Table S9). Resumable.
set -e
cd "$(dirname "$0")/.."
P="python scripts/run_experiments.py"
$P backbone --dataset cicids2017 &
$P diagnostic --dataset cicids2017 &
wait
python scripts/make_tables.py
python scripts/diagnostics.py
python scripts/export_for_paper.py
