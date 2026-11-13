@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
(for %%d in (serv omsp picosoc aes riscv biriscv) do ".venv\Scripts\python.exe" -u bdp\det.py compass data\%%d_hg.npz 0.10) > bdp\det_compass.log 2>&1
