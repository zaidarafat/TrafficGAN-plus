@echo off
REM Stream B: additional datasets, then the sensitivity analysis.
cd /d %~dp0\..
set P=python scripts\run_experiments.py
%P% main --dataset unsw_nb15  --methods wgan_gp trafficgan || exit /b 1
%P% main --dataset cicids2018 --methods wgan_gp trafficgan || exit /b 1
%P% sensitivity --dataset cicids2017 || exit /b 1
echo STREAM DONE
