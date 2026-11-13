@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
".venv\Scripts\python.exe" -u bdp\homotopy.py data\serv_hg.npz 0.10 > bdp\homotopy_serv.log 2>&1
