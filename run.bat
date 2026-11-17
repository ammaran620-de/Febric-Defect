@echo off
REM One-click start for Windows: installs dependencies, then starts the dashboard.
python -m pip install -r requirements.txt
echo.
echo Open http://localhost:8000 in your browser. Press Ctrl+C here to stop.
python -m uvicorn src.app:app --port 8000
pause
