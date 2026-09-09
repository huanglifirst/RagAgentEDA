#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "[INFO] Starting RagAgentEDA server..."
echo "[INFO] Host=${HOST} Port=${PORT}"
echo "[INFO] UI:   http://127.0.0.1:${PORT}/ragagent"
echo "[INFO] Docs: http://127.0.0.1:${PORT}/docs"

python -m uvicorn backend.app:app --host "${HOST}" --port "${PORT}"
