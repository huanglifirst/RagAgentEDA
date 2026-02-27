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
        os.environ.setdefault(key, value)


_load_env_file()


@dataclass
class Settings:
    resource_dir: Path = Path(os.getenv('RAG_RESOURCE_DIR', 'Resource'))
    work_dir: Path = Path(os.getenv('RAG_WORK_DIR', './workdir'))
    vector_index_dir: Path = Path(os.getenv('RAG_VECTOR_INDEX_DIR', './workdir/vector_index'))

    # model settings from .env
    openai_api_base: str = os.getenv('OPENAI_API_BASE', 'https://ark.cn-beijing.volces.com/api/v3')
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    model_name: str = os.getenv('MODEL_NAME', 'deepseek-v3-1-terminus')
    embedding_model_text: str = os.getenv('EMBEDDING_MODEL_TEXT', 'doubao-embedding-large')
    embedding_model_vision: str = os.getenv('EMBEDDING_MODEL_VISION', 'doubao-embedding-vision')

    # execution settings
    execution_mode: str = os.getenv('RAG_EXECUTION_MODE', 'local')  # local|ssh
    local_bashrc: str = os.getenv('RAG_LOCAL_BASHRC', '~/.bashrc')

    ssh_host: str = os.getenv('RAG_SSH_HOST', '127.0.0.1')
    ssh_user: str = os.getenv('RAG_SSH_USER', '')
    ssh_port: int = int(os.getenv('RAG_SSH_PORT', '22'))
    ssh_key_path: str = os.getenv('RAG_SSH_KEY_PATH', '')
    remote_work_dir: str = os.getenv('RAG_REMOTE_WORK_DIR', '/tmp/ragagent_tasks')


settings = Settings()
