@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
for %%d in (serv omsp picosoc aes riscv biriscv) do start "" /b cmd /c ".venv\Scripts\python.exe -u bdp\refsel.py data\%%d_hg.npz 0.10 > bdp\refsel_%%d.log 2>&1"
