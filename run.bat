@echo off
REM Windows batch script for running censor-engine with FFMPEG and keeping the window open

REM Set FFMPEG_BINARY if not already set
IF "%FFMPEG_BINARY%"=="" (
    SET FFMPEG_BINARY=C:\path\to\ffmpeg.exe
)

REM Run the Python script with uv, passing all command-line arguments
uv run python src\main.py %*

REM Keep the console open
pause