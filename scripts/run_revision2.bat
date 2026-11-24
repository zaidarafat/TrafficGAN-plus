@echo off
REM ===========================================================================
REM  Second revision: stronger-backbone study (Section 4.8) and TGP-activity runs (Table S9).
REM  Two parallel streams (two new windows). Resumable: if a window closes, run this file again.
REM  Approximate time on one RTX 3090: about one day with both streams running.
REM  Stream A: Transformer backbone, 4 variants x 5 seeds, full test-set evaluation.
REM  Stream B: CNN-LSTM WGAN-GP and TrafficGAN+ re-runs, 3 seeds, logging increment-gradient norms.
REM ===========================================================================
cd /d %~dp0\..
set LOKY_MAX_CPU_COUNT=8
start "TrafficGAN revision 2 - stream A" cmd /k "python scripts\run_experiments.py backbone --dataset cicids2017 && echo STREAM A DONE"
start "TrafficGAN revision 2 - stream B" cmd /k "python scripts\run_experiments.py diagnostic --dataset cicids2017 && echo STREAM B DONE"
echo Two windows were opened. When BOTH say "STREAM ... DONE", run:
echo    python scripts\make_tables.py
echo    python scripts\diagnostics.py
echo    python scripts\export_for_paper.py
