@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Portable Windows launcher for the dataset-generator GUI.
REM Search project-local environments first, then the active Conda environment,
REM then the standard Python launchers on PATH. No user-specific path is used.

set "APP=%~dp0gui.py"

if exist "%~dp0.venv\Scripts\pythonw.exe" (
    start "" "%~dp0.venv\Scripts\pythonw.exe" "%APP%"
    exit /b 0
)

if exist "%~dp0venv\Scripts\pythonw.exe" (
    start "" "%~dp0venv\Scripts\pythonw.exe" "%APP%"
    exit /b 0
)

if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\pythonw.exe" (
    start "" "%CONDA_PREFIX%\pythonw.exe" "%APP%"
    exit /b 0
)

where pyw.exe >nul 2>nul
if not errorlevel 1 (
    start "" pyw.exe -3 "%APP%"
    exit /b 0
)

where pythonw.exe >nul 2>nul
if not errorlevel 1 (
    start "" pythonw.exe "%APP%"
    exit /b 0
)

where py.exe >nul 2>nul
if not errorlevel 1 (
    py.exe -3 "%APP%"
    set "GUI_EXIT=!errorlevel!"
    if not "!GUI_EXIT!"=="0" pause
    exit /b !GUI_EXIT!
)

where python.exe >nul 2>nul
if not errorlevel 1 (
    python.exe "%APP%"
    set "GUI_EXIT=!errorlevel!"
    if not "!GUI_EXIT!"=="0" pause
    exit /b !GUI_EXIT!
)

echo No usable Python installation was found.
echo.
echo Create .venv using the setup steps in README.md, or activate a Conda
echo environment that contains Pillow, NumPy, tqdm, and Tkinter.
pause
exit /b 1
