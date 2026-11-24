@echo off
REM Full experiment schedule (Windows). Resumable: run again after an interruption.
cd /d %~dp0\..
set P=python scripts\run_experiments.py
%P% prepare --dataset cicids2017 || exit /b 1
%P% main --dataset cicids2017 || exit /b 1
%P% sensitivity --dataset cicids2017 || exit /b 1
for %%d in (unsw_nb15 cicids2018) do (
  %P% prepare --dataset %%d || exit /b 1
  %P% main --dataset %%d --methods none smote ctgan timegan dgan wgan_gp trafficgan || exit /b 1
)
%P% transfer --source cicids2017 --target cicids2018 || exit /b 1
python scripts\make_tables.py || exit /b 1
python scripts\export_for_paper.py
