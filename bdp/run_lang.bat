@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
for %%d in (serv omsp picosoc aes riscv biriscv) do ".venv\Scripts\python.exe" -u bdp\lang.py data\%%d_hg.npz 12 >> bdp\lang.log 2>&1
