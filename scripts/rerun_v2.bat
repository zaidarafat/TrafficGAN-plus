@echo off
REM ===========================================================================
REM  TrafficGAN+ v2: re-run ONLY the CNN-LSTM runs with the corrected
REM  (projection-conditioned) critic. All other baselines are kept as they are.
REM  1) moves the old CNN-LSTM results to results_v1_unconditional\ (nothing deleted)
REM  2) starts two parallel streams (two new windows). Each stream is resumable:
REM     if a window closes, run this file again and it continues where it stopped.
REM ===========================================================================
cd /d %~dp0\..
if not exist results_v1_unconditional (
  echo Moving old CNN-LSTM results to results_v1_unconditional ...
  for %%m in (trafficgan no_tgp no_ase wgan_gp alt_sn alt_lp alt_acgan alt_pacgan sensitivity) do (
    if exist results\cicids2017\%%m (
      mkdir results_v1_unconditional\cicids2017 2>nul
      move results\cicids2017\%%m results_v1_unconditional\cicids2017\%%m >nul
    )
  )
  for %%d in (unsw_nb15 cicids2018) do (
    for %%m in (trafficgan wgan_gp) do (
      if exist results\%%d\%%m (
        mkdir results_v1_unconditional\%%d 2>nul
        move results\%%d\%%m results_v1_unconditional\%%d\%%m >nul
      )
    )
  )
  if exist results\transfer move results\transfer results_v1_unconditional\transfer >nul
)
set LOKY_MAX_CPU_COUNT=8
start "TrafficGAN v2 - stream A" cmd /k call scripts\rerun_v2_streamA.bat
start "TrafficGAN v2 - stream B" cmd /k call scripts\rerun_v2_streamB.bat
echo Two windows were opened. When BOTH say "STREAM DONE", run:
echo    python scripts\make_tables.py
echo    python scripts\export_for_paper.py
