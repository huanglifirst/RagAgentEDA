#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _request_json(url: str, method: str = "GET") -> dict:
    req = Request(url=url, method=method, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=120) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} {url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"URL error {url}: {exc.reason}") from exc


def _check_health(base_url: str) -> None:
    health = _request_json(f"{base_url}/health")
    ok = bool(health.get("ok"))
    if not ok:
        raise RuntimeError(f"/health check failed: {health}")
    print("[PASS] /health ok=true")
    print(
        "[INFO] model={model}, embedding_model={emb}, rerank_enabled={rerank}, vector_index_latest={latest}".format(
            model=health.get("model"),
            emb=health.get("embedding_model"),
            rerank=health.get("rerank_enabled"),
            latest=health.get("vector_index_latest"),
        )
    )


def _check_reindex(base_url: str) -> None:
    result = _request_json(f"{base_url}/v1/rag/reindex", method="POST")
    chunk_count = int(result.get("chunk_count", 0) or 0)
    vector_count = int(result.get("vector_count", 0) or 0)
    if chunk_count <= 0:
        raise RuntimeError(f"/v1/rag/reindex returned invalid chunk_count: {result}")
    if vector_count <= 0:
        raise RuntimeError(f"/v1/rag/reindex returned invalid vector_count: {result}")
    if vector_count != chunk_count:
        raise RuntimeError(
            f"/v1/rag/reindex mismatch: vector_count={vector_count}, chunk_count={chunk_count}, payload={result}"
        )
    print("[PASS] /v1/rag/reindex chunk_count>0 and vector_count==chunk_count")
    print(
        "[INFO] doc_count={doc}, chunk_count={chunk}, vector_count={vec}, fingerprint={fp}".format(
            doc=result.get("doc_count"),
            chunk=chunk_count,
            vec=vector_count,
            fp=result.get("fingerprint"),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="RagAgentEDA local smoke checks")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    print(f"[INFO] Smoke checking {base_url}")
    _check_health(base_url)
    _check_reindex(base_url)
    print("[PASS] Smoke checks completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
