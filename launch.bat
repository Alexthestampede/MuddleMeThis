@echo off
REM MuddleMeThis Launch Script for Windows
REM Simple launcher for the MuddleMeThis application

echo 🎨 Starting MuddleMeThis...
echo.

REM Check if venv exists and activate it
if exist "venv\Scripts\activate.bat" (
    echo 📦 Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo 📦 Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Launch the application
echo 🚀 Launching MuddleMeThis...
echo 📱 Access at: http://localhost:7860
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py
pause
