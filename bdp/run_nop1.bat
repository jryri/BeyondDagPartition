@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
for %%d in (serv omsp picosoc aes riscv biriscv) do ".venv\Scripts\python.exe" -u bdp\homotopy_dec.py data\%%d_hg.npz nop 1 12 >> bdp\dec_nop_1.log 2>&1
