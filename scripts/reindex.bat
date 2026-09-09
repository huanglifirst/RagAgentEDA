@echo off
setlocal

if "%BASE_URL%"=="" set BASE_URL=http://127.0.0.1:8000

echo [INFO] Rebuilding vector index at %BASE_URL%/v1/rag/reindex
curl -sS -X POST "%BASE_URL%/v1/rag/reindex"
set EXIT_CODE=%ERRORLEVEL%
echo.

if not "%EXIT_CODE%"=="0" (
  echo [ERROR] Reindex request failed with code %EXIT_CODE%
)

exit /b %EXIT_CODE%
