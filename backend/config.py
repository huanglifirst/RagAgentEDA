from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _load_env_file(path: Path = Path('.env')) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        raw = line.strip()
        if not raw or raw.startswith('#') or '=' not in raw:
            continue
        key, value = raw.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


_load_env_file()


@dataclass
class Settings:
    resource_dir: Path = Path(os.getenv('RAG_RESOURCE_DIR', 'Resource'))
    work_dir: Path = Path(os.getenv('RAG_WORK_DIR', './workdir'))
    vector_index_dir: Path = Path(os.getenv('RAG_VECTOR_INDEX_DIR', './workdir/vector_index'))

    # model settings from .env
    openai_api_base: str = os.getenv('OPENAI_API_BASE', 'https://ark.cn-beijing.volces.com/api/v3')
    embedding_api_base: str = os.getenv('EMBEDDING_API_BASE', 'https://a.fe8.cn/v1')
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    embedding_api_key: str = os.getenv('EMBEDDING_API_KEY', os.getenv('OPENAI_API_KEY', ''))
    model_name: str = os.getenv('MODEL_NAME', 'deepseek-v3-1-terminus')
    embedding_model_text: str = os.getenv('EMBEDDING_MODEL_TEXT', 'text-embedding-v4')
    embedding_model_vision: str = os.getenv('EMBEDDING_MODEL_VISION', 'doubao-embedding-vision')
    embedding_batch_size: int = int(os.getenv('EMBEDDING_BATCH_SIZE', '10'))
    embedding_retry_count: int = int(os.getenv('EMBEDDING_RETRY_COUNT', '4'))
    embedding_retry_delay_sec: float = float(os.getenv('EMBEDDING_RETRY_DELAY_SEC', '1.0'))
    rerank_enabled: bool = os.getenv('RERANK_ENABLED', 'true').strip().lower() in ('1', 'true', 'yes', 'on')
    rerank_model_text: str = os.getenv('RERANK_MODEL_TEXT', 'qwen3-rerank')
    rerank_api_base: str = os.getenv('RERANK_API_BASE', os.getenv('EMBEDDING_API_BASE', 'https://a.fe8.cn/v1'))
    rerank_api_key: str = os.getenv('RERANK_API_KEY', os.getenv('EMBEDDING_API_KEY', os.getenv('OPENAI_API_KEY', '')))
    rerank_topn_factor: int = int(os.getenv('RERANK_TOPN_FACTOR', '4'))

    # execution settings
    execution_mode: str = os.getenv('RAG_EXECUTION_MODE', 'local')  # local|ssh
    local_bashrc: str = os.getenv('RAG_LOCAL_BASHRC', '~/.bashrc')
    remote_bashrc: str = os.getenv('RAG_REMOTE_BASHRC', '~/.bashrc')
    ted_required_commands: str = os.getenv('RAG_TED_REQUIRED_COMMANDS', 'python')
    ted_required_python_modules: str = os.getenv('RAG_TED_REQUIRED_PY_MODULES', '')

    ssh_host: str = os.getenv('RAG_SSH_HOST', '127.0.0.1')
    ssh_user: str = os.getenv('RAG_SSH_USER', '')
    ssh_port: int = int(os.getenv('RAG_SSH_PORT', '22'))
    ssh_key_path: str = os.getenv('RAG_SSH_KEY_PATH', '')
    remote_work_dir: str = os.getenv('RAG_REMOTE_WORK_DIR', '/tmp/ragagent_tasks')


settings = Settings()
