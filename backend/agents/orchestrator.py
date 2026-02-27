from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from backend.agents.codegen import ScriptGenerator
from backend.config import settings
from backend.llm.client import OpenAICompatClient
from backend.rag.indexer import ResourceIndexer, Chunk
from backend.rag.retriever import HybridRetriever, ScoredChunk
from backend.rag.vector_store import EmbeddingRetriever, PersistentEmbeddingIndex
from backend.runner.local_runner import LocalTedRunner
from backend.runner.ssh_runner import SSHRunner, SSHConfig, new_task_id
from backend.schemas.api import EvidenceItem, RunTaskRequest, RunTaskResponse


class AgentState(TypedDict, total=False):
    task_id: str
    request: RunTaskRequest
    chunks: List[Chunk]
    evidence: List[ScoredChunk]
    generated_code: str
    run_status: str
    run_metrics: Dict[str, Any]
    run_stdout: str
    run_stderr: str
    run_error: str


@dataclass
class PipelineDeps:
    resource_dir: Path
    work_dir: Path
    ssh_cfg: SSHConfig


class RagLangGraphPipeline:
    def __init__(self, deps: PipelineDeps) -> None:
        self.deps = deps
        self.indexer = ResourceIndexer(deps.resource_dir)
        self.client = OpenAICompatClient(settings.openai_api_base, settings.openai_api_key)
        self.codegen = ScriptGenerator(self.client)
        self.vector_store = PersistentEmbeddingIndex(settings.vector_index_dir)

    def run(self, req: RunTaskRequest) -> RunTaskResponse:
        graph = self._build_graph()
        final = graph.invoke({'request': req})
        evidence = [
            EvidenceItem(source=s.chunk.source, score=round(s.score, 4), snippet=s.chunk.text[:180])
            for s in final['evidence']
        ]
        return RunTaskResponse(
            task_id=final['task_id'],
            status=final['run_status'],
            generated_code=final['generated_code'],
            metrics=final.get('run_metrics', {}),
            logs={'stdout': final.get('run_stdout', '')[-4000:], 'stderr': final.get('run_stderr', '')[-4000:]},
            evidence=evidence,
            error=final.get('run_error') or None,
        )

    def _build_graph(self):
        try:
            from langgraph.graph import StateGraph, END
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError('langgraph is required. Please install from requirements.txt') from exc

        g = StateGraph(AgentState)
        g.add_node('prepare', self._prepare)
        g.add_node('retrieve', self._retrieve)
        g.add_node('codegen', self._codegen)
        g.add_node('execute', self._execute)

        g.set_entry_point('prepare')
        g.add_edge('prepare', 'retrieve')
        g.add_edge('retrieve', 'codegen')
        g.add_edge('codegen', 'execute')
        g.add_edge('execute', END)
        return g.compile()

    def _prepare(self, state: AgentState) -> AgentState:
        task_id = new_task_id()
        self.deps.work_dir.mkdir(parents=True, exist_ok=True)
        (self.deps.work_dir / task_id).mkdir(parents=True, exist_ok=True)
        return {'task_id': task_id}

    def _retrieve(self, state: AgentState) -> AgentState:
        req = state['request']
        chunks = self.indexer.index()
        fingerprint = self.indexer.fingerprint()

        # stage-1: embedding retrieval (fallback to lexical if api unavailable)
        try:
            emb = EmbeddingRetriever.from_chunks(
                self.client,
                settings.embedding_model_text,
                chunks,
                self.vector_store,
                fingerprint,
            )
            emb_hits = emb.search(req.query, top_k=max(20, req.top_k * 4))
            candidates = [h.chunk for h in emb_hits]
            if not candidates:
                candidates = chunks
        except Exception:  # noqa: BLE001
            candidates = chunks

        # stage-2: keyword/bm25 rerank in candidate pool
        hybrid = HybridRetriever(candidates)
        reranked = hybrid.retrieve(req.query, top_k=req.top_k)
        return {'chunks': chunks, 'evidence': reranked}

    def _codegen(self, state: AgentState) -> AgentState:
        req = state['request']
        script = self.codegen.generate(req.query, req.circuit_description, state['evidence'])
        task_dir = self.deps.work_dir / state['task_id']
        local_script = task_dir / 'main.py'
        local_script.write_text(script.code, encoding='utf-8')
        return {'generated_code': script.code}

    def _execute(self, state: AgentState) -> AgentState:
        task_dir = self.deps.work_dir / state['task_id']
        local_script = task_dir / 'main.py'
        if settings.execution_mode == 'ssh':
            runner = SSHRunner(self.deps.ssh_cfg)
        else:
            runner = LocalTedRunner(settings.local_bashrc)

        out = runner.run_script(local_script=local_script, task_id=state['task_id'])
        return {
            'run_status': out.status,
            'run_metrics': out.metrics,
            'run_stdout': out.stdout,
            'run_stderr': out.stderr,
            'run_error': out.error,
        }
