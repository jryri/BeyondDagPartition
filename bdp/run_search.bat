@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
".venv\Scripts\python.exe" -u bdp\search.py data\ariane133_hg.npz 0.10 1 > bdp\search_ariane133.log 2>&1
