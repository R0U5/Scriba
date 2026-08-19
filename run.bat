@echo off
rem RosettaAI launcher. Double-click this file, or run it with arguments:
rem     run.bat
rem     run.bat transcribe https://www.youtube.com/watch?v=aircAruvnKk
rem     run.bat doctor
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-windows.ps1" %*
set EXITCODE=%ERRORLEVEL%
if not "%EXITCODE%"=="0" (
    echo.
    echo RosettaAI stopped with exit code %EXITCODE%. The message above explains why.
    pause
)
exit /b %EXITCODE%
