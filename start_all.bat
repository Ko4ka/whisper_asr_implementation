@echo off
setlocal

REM Navigate to the specified directory
cd /d C:\Users\Alex\whisper_asr_implementation

REM Activate virtual environment
call venv\Scripts\activate

REM Start watchdog (spawns and restarts services as needed)
py watchdog.py

endlocal

