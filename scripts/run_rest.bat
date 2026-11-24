@echo off
REM Remaining experiments with a reduced budget for the slow third-party baselines
REM (CTGAN, CopulaGAN, DoppelGANger: 5 seeds instead of 10). Resumable: run again after an interruption.
cd /d %~dp0\..
set P=python scripts\run_experiments.py
%P% main --dataset cicids2017 --methods ctgan copulagan dgan --seeds 0 1 2 3 4 || exit /b 1
%P% sensitivity --dataset cicids2017 || exit /b 1
for %%d in (unsw_nb15 cicids2018) do (
  if not exist data\processed\%%d_T10.npz %P% prepare --dataset %%d || exit /b 1
  %P% main --dataset %%d --methods none smote wgan_gp trafficgan timegan || exit /b 1
  %P% main --dataset %%d --methods ctgan dgan --seeds 0 1 2 3 4 || exit /b 1
)
%P% transfer --source cicids2017 --target cicids2018 || exit /b 1
python scripts\make_tables.py || exit /b 1
python scripts\export_for_paper.py
