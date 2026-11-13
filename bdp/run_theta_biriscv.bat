@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
".venv\Scripts\python.exe" -u bdp\theta_opt.py data\biriscv_hg.npz 0.10 1000 > bdp\theta_opt_biriscv.log 2>&1
