@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
for %%k in (linf alpha kahn) do for %%d in (serv omsp picosoc aes riscv biriscv) do ".venv\Scripts\python.exe" -u bdp\homotopy_dec.py data\%%d_hg.npz %%k inf 12 >> bdp\dec_%%k.log 2>&1
for %%l in (4 1) do for %%d in (serv omsp picosoc aes riscv biriscv) do ".venv\Scripts\python.exe" -u bdp\homotopy_dec.py data\%%d_hg.npz nop %%l 12 >> bdp\dec_nop_%%l.log 2>&1
