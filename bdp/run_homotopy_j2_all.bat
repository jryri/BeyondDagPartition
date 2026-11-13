@echo off
cd /d "%~dp0.."
set PYTHONWARNINGS=ignore
for %%d in (serv omsp picosoc aes riscv biriscv) do start "" /b cmd /c ".venv\Scripts\python.exe -u bdp\homotopy.py data\%%d_hg.npz 0.10 j2 > bdp\homotopy_j2_%%d.log 2>&1"
