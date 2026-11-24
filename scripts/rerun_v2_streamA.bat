@echo off
REM Stream A: CIC-IDS-2017 main comparison and ablation, then the transfer experiment.
cd /d %~dp0\..
set P=python scripts\run_experiments.py
%P% main --dataset cicids2017 --methods wgan_gp trafficgan no_tgp no_ase || exit /b 1
%P% main --dataset cicids2017 --methods alt_sn alt_lp alt_acgan alt_pacgan --seeds 0 1 2 3 4 || exit /b 1
%P% transfer --source cicids2017 --target cicids2018 || exit /b 1
echo STREAM DONE
