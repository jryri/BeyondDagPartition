@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
".venv\Scripts\python.exe" -u bdp\large.py data\ariane133_hg.npz 0.10 > bdp\large_ariane133_r1.log 2>&1
".venv\Scripts\python.exe" -u bdp\large.py data\bp_quad_hg.npz 0.10 > bdp\large_bp_quad.log 2>&1
