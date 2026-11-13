@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
".venv\Scripts\python.exe" -u bdp\homotopy_dec.py data\biriscv_hg.npz linf inf 12 1000 1 >> bdp\var_linf.log 2>&1
for %%s in (2 3 4) do for %%d in (serv omsp picosoc aes riscv biriscv) do ".venv\Scripts\python.exe" -u bdp\homotopy_dec.py data\%%d_hg.npz linf inf 12 1000 %%s >> bdp\var_linf.log 2>&1
for %%s in (1 2) do for %%d in (serv omsp picosoc aes riscv biriscv) do ".venv\Scripts\python.exe" -u bdp\homotopy_dec.py data\%%d_hg.npz nop 1 12 1000 %%s >> bdp\var_nop1.log 2>&1
