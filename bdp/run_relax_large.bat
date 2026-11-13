@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
".venv\Scripts\python.exe" -u bdp\relax_large.py data\ariane133_hg.npz 0.10 > bdp\relax_ariane133.log 2>&1
".venv\Scripts\python.exe" -u bdp\relax_large.py data\bp_quad_hg.npz 0.10 > bdp\relax_bp_quad.log 2>&1
