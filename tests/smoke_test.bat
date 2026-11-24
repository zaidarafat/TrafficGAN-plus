@echo off
REM End-to-end check on tiny FAKE data (numbers are meaningless). Windows version of smoke_test.sh.
cd /d %~dp0\..
set PYTHONPATH=.
if exist tests\raw rmdir /s /q tests\raw
if exist tests\processed rmdir /s /q tests\processed
if exist tests\results rmdir /s /q tests\results
python scripts\make_synthetic_testdata.py tests\raw || exit /b 1
python tests\test_tgp.py || exit /b 1
set C=--config tests/smoke.yaml
set P=python scripts\run_experiments.py
for %%d in (cicids2017 unsw_nb15 cicids2018) do (%P% prepare --dataset %%d %C% || exit /b 1)
%P% main --dataset cicids2017 %C% --seeds 0 1 || exit /b 1
for %%d in (unsw_nb15 cicids2018) do (%P% main --dataset %%d %C% --methods none smote ctgan timegan dgan wgan_gp trafficgan --seeds 0 || exit /b 1)
%P% transfer --source cicids2017 --target cicids2018 %C% --seeds 0 1 || exit /b 1
%P% sensitivity --dataset cicids2017 %C% || exit /b 1
%P% backbone --dataset cicids2017 %C% || exit /b 1
%P% diagnostic --dataset cicids2017 %C% || exit /b 1
python scripts\make_tables.py %C% || exit /b 1
python scripts\diagnostics.py %C% || exit /b 1
python scripts\export_for_paper.py tests/smoke.yaml || exit /b 1
dir /s /b tests\results\ERROR.txt >nul 2>&1 && (echo SMOKE TEST: errors found & exit /b 1)
echo SMOKE TEST PASSED
