#!/usr/bin/env bash
# End-to-end check on tiny FAKE data (numbers are meaningless). ~5-15 min on CPU, faster on GPU.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
rm -rf tests/raw tests/processed tests/results
python scripts/make_synthetic_testdata.py tests/raw
python tests/test_tgp.py
C="--config tests/smoke.yaml"
P="python scripts/run_experiments.py"
for ds in cicids2017 unsw_nb15 cicids2018; do $P prepare --dataset $ds $C; done
$P main --dataset cicids2017 $C --seeds 0 1
for ds in unsw_nb15 cicids2018; do $P main --dataset $ds $C --methods none smote ctgan timegan dgan wgan_gp trafficgan --seeds 0; done
$P transfer --source cicids2017 --target cicids2018 $C --seeds 0 1
$P sensitivity --dataset cicids2017 $C
$P backbone --dataset cicids2017 $C
$P diagnostic --dataset cicids2017 $C
python scripts/make_tables.py $C
python scripts/diagnostics.py --config tests/smoke.yaml
python scripts/export_for_paper.py tests/smoke.yaml
if ls tests/results/*/*/seed*/ERROR.txt >/dev/null 2>&1; then echo "SMOKE TEST: errors found"; exit 1; fi
echo "SMOKE TEST PASSED"
