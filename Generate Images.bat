@echo off
REM Double-click this file to open the dataset generator GUI (no terminal needed).
REM Uses pythonw.exe from the 'trocr' conda environment so no console window appears.

set "PYW=C:\Users\limmi\.conda\envs\trocr\pythonw.exe"

if not exist "%PYW%" (
    echo Could not find the trocr environment Python at:
    echo   %PYW%
    echo.
    echo Please update the PYW path in this .bat file.
    pause
    exit /b 1
)

start "" "%PYW%" "%~dp0gui.py"
