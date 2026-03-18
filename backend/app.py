from __future__ import annotations

from fastapi import FastAPI, HTTPException

from backend.agents.orchestrator import PipelineDeps, RagLangGraphPipeline
from backend.config import settings
from backend.rag.indexer import ResourceIndexer
from backend.rag.vector_store import EmbeddingRetriever
from backend.runner.ssh_runner import SSHConfig
from backend.schemas.api import RunTaskRequest, RunTaskResponse

app = FastAPI(title='RagAgent EDA Demo', version='0.3.0')


def _csv_tuple(raw: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(',') if part and part.strip())
    return values or default

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
            remote_bashrc=settings.remote_bashrc,
            required_commands=_csv_tuple(settings.ted_required_commands, default=('python',)),
            required_python_modules=_csv_tuple(settings.ted_required_python_modules),
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
        'model_api_base': settings.openai_api_base,
        'model_api_key_set': bool(settings.openai_api_key),
        'embedding_model': settings.embedding_model_text,
        'embedding_api_base': settings.embedding_api_base,
        'embedding_api_key_set': bool(settings.embedding_api_key),
        'rerank_enabled': settings.rerank_enabled,
        'rerank_model': settings.rerank_model_text,
        'rerank_api_base': settings.rerank_api_base,
        'rerank_api_key_set': bool(settings.rerank_api_key),
        'remote_bashrc': settings.remote_bashrc,
        'vector_index_dir': str(settings.vector_index_dir),
        'vector_index_latest': latest.read_text(encoding='utf-8').strip() if latest.exists() else None,
    }


@app.post('/v1/rag/reindex')
def reindex() -> dict:
    try:
        indexer = ResourceIndexer(settings.resource_dir)
        docs = indexer.list_docs()
        if not docs:
            raise RuntimeError(f'resource indexing failed: no eligible documents found in {settings.resource_dir}')

        chunks = indexer.index()
        if not chunks:
            raise RuntimeError(f'resource indexing failed: no chunks generated from {settings.resource_dir}')

        fingerprint = indexer.fingerprint()
        emb = EmbeddingRetriever.from_chunks(
            pipeline.embedding_client,
            settings.embedding_model_text,
            chunks,
            pipeline.vector_store,
            fingerprint,
            force_rebuild=True,
        )
        vector_count = len(emb._vectors)
        chunk_count = len(emb.chunks)
        if vector_count <= 0:
            raise RuntimeError('embedding build failed: invalid vector index (vector_count<=0)')
        if vector_count != chunk_count:
            raise RuntimeError(
                'embedding build failed: invalid vector index '
                f'(vector_count={vector_count}, chunk_count={chunk_count})'
            )
        return {
            'ok': True,
            'doc_count': len(docs),
            'fingerprint': fingerprint,
            'chunk_count': chunk_count,
            'vector_count': vector_count,
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
