#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

echo "[INFO] Rebuilding vector index at ${BASE_URL}/v1/rag/reindex"
curl -sS -X POST "${BASE_URL}/v1/rag/reindex"
echo
