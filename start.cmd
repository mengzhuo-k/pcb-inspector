@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -c "import ultralytics, albumentations, flask, waitress, reportlab" >nul 2>&1
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto failed
)
echo Open http://127.0.0.1:8765 in your browser.
".venv\Scripts\python.exe" app.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo Startup failed. See the error above.
pause
exit /b 1
