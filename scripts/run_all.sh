#!/usr/bin/env bash
# Full experiment schedule for the revised manuscript. Resumable: re-run the same command after an interruption.
set -e
cd "$(dirname "$0")/.."
P="python scripts/run_experiments.py"
$P prepare --dataset cicids2017
$P main --dataset cicids2017
$P sensitivity --dataset cicids2017
for ds in unsw_nb15 cicids2018; do
  $P prepare --dataset $ds
  $P main --dataset $ds --methods none smote ctgan timegan dgan wgan_gp trafficgan
done
$P transfer --source cicids2017 --target cicids2018
python scripts/make_tables.py
python scripts/export_for_paper.py
