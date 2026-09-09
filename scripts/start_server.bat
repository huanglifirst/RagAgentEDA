@echo off
setlocal

cd /d %~dp0\..

if "%HOST%"=="" set HOST=0.0.0.0
if "%PORT%"=="" set PORT=8000

echo [INFO] Starting RagAgentEDA server...
echo [INFO] Host=%HOST% Port=%PORT%
echo [INFO] UI:   http://127.0.0.1:%PORT%/ragagent
echo [INFO] Docs: http://127.0.0.1:%PORT%/docs

python -m uvicorn backend.app:app --host %HOST% --port %PORT%
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
  echo [ERROR] Server exited with code %EXIT_CODE%
)

exit /b %EXIT_CODE%
