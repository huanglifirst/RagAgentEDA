from __future__ import annotations

from fastapi import FastAPI, HTTPException

from backend.agents.orchestrator import PipelineDeps, RagLangGraphPipeline
from backend.config import settings
from backend.rag.indexer import ResourceIndexer
from backend.rag.vector_store import EmbeddingRetriever
from backend.runner.ssh_runner import SSHConfig
from backend.schemas.api import RunTaskRequest, RunTaskResponse

app = FastAPI(title='RagAgent EDA Demo', version='0.3.0')

pipeline = RagLangGraphPipeline(
    PipelineDeps(
        resource_dir=settings.resource_dir,
        work_dir=settings.work_dir,
        ssh_cfg=SSHConfig(
            host=settings.ssh_host,
            user=settings.ssh_user,
            port=settings.ssh_port,
            key_path=settings.ssh_key_path,
            remote_work_dir=settings.remote_work_dir,
        ),
    )
)


@app.get('/health')
def health() -> dict:
    latest = settings.vector_index_dir / 'LATEST'
    return {
        'ok': True,
        'execution_mode': settings.execution_mode,
        'model': settings.model_name,
        'embedding_model': settings.embedding_model_text,
        'vector_index_dir': str(settings.vector_index_dir),
        'vector_index_latest': latest.read_text(encoding='utf-8').strip() if latest.exists() else None,
    }


@app.post('/v1/rag/reindex')
def reindex() -> dict:
    try:
        chunks = ResourceIndexer(settings.resource_dir).index()
        fingerprint = ResourceIndexer(settings.resource_dir).fingerprint()
        emb = EmbeddingRetriever.from_chunks(
            pipeline.client,
            settings.embedding_model_text,
            chunks,
            pipeline.vector_store,
            fingerprint,
        )
        return {
            'ok': True,
            'fingerprint': fingerprint,
            'chunk_count': len(emb.chunks),
            'vector_count': len(emb._vectors),
            'saved_dir': str(settings.vector_index_dir),
            'saved_file': str(settings.vector_index_dir / f'{fingerprint}.json'),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post('/v1/tasks/run', response_model=RunTaskResponse)
def run_task(req: RunTaskRequest) -> RunTaskResponse:
    try:
        return pipeline.run(req)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
