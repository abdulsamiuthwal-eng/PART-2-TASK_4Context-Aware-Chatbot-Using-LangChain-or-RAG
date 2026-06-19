@echo off
:: ============================================================
::  launch.bat — One-click launcher for RAG Research Assistant
::  Double-click this file to start the chatbot.
:: ============================================================

title RAG Research Assistant — Launcher
color 0A

echo.
echo  ============================================================
echo   RAG Research Assistant — Context-Aware Chatbot
echo   Powered by LangChain + Groq + ChromaDB
echo  ============================================================
echo.

:: ── Change to the script's own directory ──────────────────────────────────────
cd /d "%~dp0"

:: ── Check virtual environment exists ─────────────────────────────────────────
if not exist "venv\Scripts\activate.bat" (
    echo  [ERROR] Virtual environment not found at .\venv\
    echo.
    echo  Please run setup first:
    echo    python -m venv venv
    echo    venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: ── Activate virtual environment ──────────────────────────────────────────────
echo  [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

:: ── Load .env file (set GROQ_API_KEY) ────────────────────────────────────────
if exist ".env" (
    echo  [2/3] Loading environment variables from .env ...
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "line=%%A"
        if not "!line:~0,1!"=="#" (
            if not "%%A"=="" if not "%%B"=="" (
                set "%%A=%%B"
            )
        )
    )
) else (
    echo  [WARN] No .env file found. Using system environment variables.
    echo         Copy .env.example to .env and add your GROQ_API_KEY.
)

:: ── Check for API key ─────────────────────────────────────────────────────────
if "%GROQ_API_KEY%"=="" (
    echo.
    echo  [WARN] GROQ_API_KEY is not set.
    echo         You can enter it in the sidebar after the app launches.
    echo         Get a free key at: https://console.groq.com
    echo.
)

:: ── Launch Streamlit ──────────────────────────────────────────────────────────
echo  [3/3] Starting Streamlit app...
echo.
echo  ============================================================
echo   App URL: http://localhost:8501
echo   Press Ctrl+C in this window to stop the server.
echo  ============================================================
echo.

:: Enable delayed expansion for .env loading
setlocal enabledelayedexpansion

:: Re-load .env with delayed expansion enabled
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "line=%%A"
        if not "!line:~0,1!"=="#" (
            if not "%%A"=="" if not "%%B"=="" (
                set "%%A=%%B"
            )
        )
    )
)

:: Start the app
streamlit run app.py --server.headless false --browser.gatherUsageStats false

:: ── If Streamlit exits, pause so the user can see any errors ──────────────────
echo.
echo  [INFO] Streamlit server stopped.
pause
