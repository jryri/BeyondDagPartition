@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
(for %%d in (serv omsp picosoc aes riscv biriscv) do ".venv\Scripts\python.exe" -u bdp\candsel.py data\%%d_hg.npz 0.10) > bdp\candsel_small.log 2>&1
