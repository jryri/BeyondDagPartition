@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
".venv\Scripts\python.exe" -u bdp\homotopy.py data\picosoc_hg.npz 0.10 > bdp\homotopy_picosoc.log 2>&1
