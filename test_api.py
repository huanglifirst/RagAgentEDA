import os
from pathlib import Path

import requests


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


if __name__ == "__main__":
    load_env_file()

    api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    base_url = os.getenv("EMBEDDING_API_BASE", os.getenv("OPENAI_API_BASE", "https://a.fe8.cn/v1"))
    model_name = os.getenv("EMBEDDING_MODEL_TEXT", "text-embedding-v4")

    print(f"===== Embedding smoke test: {model_name} =====")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY is missing")
        raise SystemExit(1)

    request_url = f"{base_url.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "input": "测试模型连通性",
        "encoding_format": "float",
    }

    try:
        response = requests.post(url=request_url, headers=headers, json=payload, timeout=10)
        print(f"请求状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("[OK] embedding request succeeded")
            print(f"[INFO] vector_dim: {len(result['data'][0]['embedding'])}")
        else:
            print(f"[ERROR] request failed: {response.text}")
    except Exception as e:
        print(f"[ERROR] exception: {str(e)}")
