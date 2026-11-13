@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
(for %%d in (serv omsp picosoc aes riscv biriscv) do ".venv\Scripts\python.exe" -u bdp\mono_dict.py data\%%d_hg.npz 0.10 32 128) > bdp\dict_small.log 2>&1
