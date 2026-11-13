@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
for %%d in (serv omsp picosoc aes riscv biriscv) do ".venv\Scripts\python.exe" -u bdp\homotopy_ul.py data\%%d_hg.npz 0.10 12 ul > bdp\homotopy_ul_%%d.log 2>&1
for %%d in (serv omsp picosoc aes riscv biriscv) do ".venv\Scripts\python.exe" -u bdp\homotopy_ul.py data\%%d_hg.npz 0.10 12 x > bdp\homotopy_plx_%%d.log 2>&1
