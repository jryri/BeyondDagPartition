@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
for %%d in (omsp picosoc aes riscv biriscv) do ".venv\Scripts\python.exe" -u bdp\oracle2.py data\%%d_hg.npz 0.10 > bdp\oracle2_%%d.log 2>&1
for %%d in (serv omsp picosoc aes riscv biriscv) do ".venv\Scripts\python.exe" -u bdp\homotopy_par.py data\%%d_hg.npz 0.10 12 g > bdp\homotopy_g_%%d.log 2>&1
