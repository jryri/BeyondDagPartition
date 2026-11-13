@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
".venv\Scripts\python.exe" -u bdp\diag.py compass data\picosoc_hg.npz 0.10 > bdp\diag_compass_picosoc.log 2>&1
