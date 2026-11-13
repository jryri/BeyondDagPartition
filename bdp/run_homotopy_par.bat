@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
for %%d in (omsp picosoc aes riscv biriscv) do ".venv\Scripts\python.exe" -u bdp\homotopy_par.py data\%%d_hg.npz 0.10 12 > bdp\homotopy_par_%%d.log 2>&1
