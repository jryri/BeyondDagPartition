@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
".venv\Scripts\python.exe" -u bdp\diag.py compass data\riscv_hg.npz 0.10 > bdp\diag_compass_riscv.log 2>&1
