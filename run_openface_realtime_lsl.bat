@echo off
setlocal

cd /d "%~dp0"

set "VENV_DIR=%~dp0lsl_env"
set "ACTIVATE_BAT=%VENV_DIR%\Scripts\activate.bat"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "SCRIPT_PATH=%~dp0openface_realtime_lsl.py"

if not exist "%SCRIPT_PATH%" (
    echo Could not find script:
    echo %SCRIPT_PATH%
    pause
    exit /b 1
)

if exist "%ACTIVATE_BAT%" (
    call "%ACTIVATE_BAT%"
)

if not exist "%PYTHON_EXE%" (
    echo Could not find local virtual environment Python at:
    echo %PYTHON_EXE%
    echo.
    echo Expected virtual environment folder:
    echo %VENV_DIR%
    pause
    exit /b 1
)

"%PYTHON_EXE%" "%SCRIPT_PATH%"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Script exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
