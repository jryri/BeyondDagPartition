@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
".venv\Scripts\python.exe" -u bdp\homotopy.py data\riscv_hg.npz 0.10 > bdp\homotopy_riscv.log 2>&1
