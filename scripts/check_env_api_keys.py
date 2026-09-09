#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from backend.config import settings  # noqa: E402
from backend.llm.client import OpenAICompatClient  # noqa: E402


@dataclass
class CheckResult:
    name: str
    ok: bool
    skipped: bool = False


def _configured(value: str | None) -> bool:
    return bool((value or "").strip())


def _redact(text: object) -> str:
    redacted = str(text)
    secret_values = {
        settings.openai_api_key,
        settings.embedding_api_key,
        settings.rerank_api_key,
        os.getenv("OPENAI_API_KEY", ""),
        os.getenv("EMBEDDING_API_KEY", ""),
        os.getenv("RERANK_API_KEY", ""),
    }
    for secret in secret_values:
        if secret:
            redacted = redacted.replace(secret, "<redacted>")
    return redacted


def _print_config(name: str, base_url: str, model: str, key: str) -> None:
    print(f"[INFO] {name} base_url={base_url.rstrip('/')}")
    print(f"[INFO] {name} model={model}")
    print(f"[INFO] {name} api_key_set={_configured(key)}")


def _model_id(item: object) -> str:
    if isinstance(item, dict):
        for key in ("id", "model", "name"):
            value = item.get(key)
            if value:
                return str(value)
    return str(item)


def _fetch_model_items(client: OpenAICompatClient) -> list[object]:
    models = client.list_models()
    model_items = models.get("data")
    if not isinstance(model_items, list):
        raise RuntimeError(f"/models returned unexpected payload: {OpenAICompatClient._brief(models)}")
    return model_items


def _run_required(name: str, check: Callable[[], None]) -> CheckResult:
    try:
        check()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {name}: {_redact(exc)}")
        return CheckResult(name=name, ok=False)
    print(f"[PASS] {name}")
    return CheckResult(name=name, ok=True)


def _check_chat() -> None:
    _print_config("chat", settings.openai_api_base, settings.model_name, settings.openai_api_key)
    if not _configured(settings.openai_api_key):
        raise RuntimeError("missing OPENAI_API_KEY")
    if not _configured(settings.openai_api_base):
        raise RuntimeError("missing OPENAI_API_BASE")
    if not _configured(settings.model_name):
        raise RuntimeError("missing MODEL_NAME")

    client = OpenAICompatClient(settings.openai_api_base, settings.openai_api_key)
    model_items = _fetch_model_items(client)
    print(f"[INFO] chat models_list_count={len(model_items)}")

    content = client.chat(
        settings.model_name,
        [{"role": "user", "content": "Reply with exactly: ok"}],
        temperature=0,
    )
    if not _configured(content):
        raise RuntimeError("chat completion returned empty content")
    print(f"[INFO] chat response_chars={len(content)}")


def _check_embedding() -> None:
    _print_config(
        "embedding",
        settings.embedding_api_base,
        settings.embedding_model_text,
        settings.embedding_api_key,
    )
    if not _configured(settings.embedding_api_key):
        raise RuntimeError("missing EMBEDDING_API_KEY and fallback OPENAI_API_KEY")
    if not _configured(settings.embedding_api_base):
        raise RuntimeError("missing EMBEDDING_API_BASE")
    if not _configured(settings.embedding_model_text):
        raise RuntimeError("missing EMBEDDING_MODEL_TEXT")

    client = OpenAICompatClient(settings.embedding_api_base, settings.embedding_api_key)
    vectors = client.embed(settings.embedding_model_text, ["api key smoke test"])
    if not vectors or not vectors[0]:
        raise RuntimeError("embedding returned empty vector")
    print(f"[INFO] embedding vector_dim={len(vectors[0])}")


def _check_rerank() -> CheckResult:
    if not settings.rerank_enabled:
        print("[SKIP] rerank: RERANK_ENABLED=false")
        return CheckResult(name="rerank", ok=True, skipped=True)

    def check() -> None:
        _print_config("rerank", settings.rerank_api_base, settings.rerank_model_text, settings.rerank_api_key)
        if not _configured(settings.rerank_api_key):
            raise RuntimeError("missing RERANK_API_KEY and fallback EMBEDDING_API_KEY/OPENAI_API_KEY")
        if not _configured(settings.rerank_api_base):
            raise RuntimeError("missing RERANK_API_BASE")
        if not _configured(settings.rerank_model_text):
            raise RuntimeError("missing RERANK_MODEL_TEXT")

        client = OpenAICompatClient(settings.rerank_api_base, settings.rerank_api_key)
        ranked = client.rerank(
            settings.rerank_model_text,
            "api key smoke test",
            ["api key smoke test", "unrelated document"],
            top_n=1,
        )
        if not ranked:
            raise RuntimeError("rerank returned empty result")
        print(f"[INFO] rerank top_index={ranked[0][0]} top_score={ranked[0][1]:.6f}")

    return _run_required("rerank", check)


def _list_models() -> CheckResult:
    def check() -> None:
        _print_config("models", settings.openai_api_base, "(list only)", settings.openai_api_key)
        if not _configured(settings.openai_api_key):
            raise RuntimeError("missing OPENAI_API_KEY")
        if not _configured(settings.openai_api_base):
            raise RuntimeError("missing OPENAI_API_BASE")

        client = OpenAICompatClient(settings.openai_api_base, settings.openai_api_key)
        model_ids = sorted({_model_id(item) for item in _fetch_model_items(client)})
        print(f"[INFO] models count={len(model_ids)}")
        for model_id in model_ids:
            print(model_id)
        print("[INFO] /models means visible to this key; it does not guarantee every model is callable.")

    return _run_required("models", check)


def _probe_chat_models(model_names: list[str]) -> CheckResult:
    def check() -> None:
        _print_config("chat_probe", settings.openai_api_base, ",".join(model_names), settings.openai_api_key)
        if not _configured(settings.openai_api_key):
            raise RuntimeError("missing OPENAI_API_KEY")
        if not _configured(settings.openai_api_base):
            raise RuntimeError("missing OPENAI_API_BASE")

        client = OpenAICompatClient(settings.openai_api_base, settings.openai_api_key)
        failed: list[str] = []
        for model_name in model_names:
            model_name = model_name.strip()
            if not model_name:
                continue
            try:
                content = client.chat(
                    model_name,
                    [{"role": "user", "content": "Reply with exactly: ok"}],
                    temperature=0,
                )
                if not _configured(content):
                    raise RuntimeError("empty response")
                print(f"[PASS] chat_probe model={model_name} response_chars={len(content)}")
            except Exception as exc:  # noqa: BLE001
                failed.append(model_name)
                print(f"[FAIL] chat_probe model={model_name}: {_redact(exc)}")
        if failed:
            raise RuntimeError(f"failed chat probe models: {','.join(failed)}")

    return _run_required("chat_probe", check)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check .env API key usability for RagAgentEDA")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print model IDs returned by OPENAI_API_BASE /models.",
    )
    parser.add_argument(
        "--probe-chat-model",
        action="append",
        default=[],
        metavar="MODEL",
        help="Run a minimal chat request against one model. Can be used multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(f"[INFO] repo_root={REPO_ROOT}")
    print("[INFO] .env values are loaded by backend.config; API keys are never printed.")

    if args.list_models or args.probe_chat_model:
        results = []
        if args.list_models:
            results.append(_list_models())
        if args.probe_chat_model:
            results.append(_probe_chat_models(args.probe_chat_model))
    else:
        results = [
            _run_required("chat", _check_chat),
            _run_required("embedding", _check_embedding),
            _check_rerank(),
        ]

    failed = [result.name for result in results if not result.ok]
    skipped = [result.name for result in results if result.skipped]
    if failed:
        print(f"[FAIL] completed with failed_checks={','.join(failed)}")
        return 1
    if skipped:
        print(f"[PASS] completed with skipped_checks={','.join(skipped)}")
    else:
        print("[PASS] completed all checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
