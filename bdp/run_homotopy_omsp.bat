@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
".venv\Scripts\python.exe" -u bdp\homotopy.py data\omsp_hg.npz 0.10 > bdp\homotopy_omsp.log 2>&1
