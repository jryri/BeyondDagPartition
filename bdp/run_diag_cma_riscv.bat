@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
".venv\Scripts\python.exe" -u bdp\diag.py cma data\riscv_hg.npz 0.10 > bdp\diag_cma_riscv.log 2>&1
