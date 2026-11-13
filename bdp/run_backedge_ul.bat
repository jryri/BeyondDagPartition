@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
for %%d in (serv omsp picosoc aes riscv biriscv) do ".venv\Scripts\python.exe" -u bdp\backedge_ul.py data\%%d_hg.npz 0.10 12 > bdp\backedge_ul_%%d.log 2>&1
