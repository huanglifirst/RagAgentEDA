from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from backend.agents.codegen import CodegenError, ScriptGenerator
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
    retrieval_warning: str
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
        self.chat_client = OpenAICompatClient(settings.openai_api_base, settings.openai_api_key)
        self.embedding_client = OpenAICompatClient(settings.embedding_api_base, settings.embedding_api_key)
        self.rerank_client = OpenAICompatClient(settings.rerank_api_base, settings.rerank_api_key)
        self.codegen = ScriptGenerator(self.chat_client)
        self.vector_store = PersistentEmbeddingIndex(settings.vector_index_dir)
        self.required_commands = self._csv_tuple(settings.ted_required_commands, default=('python',))
        self.required_python_modules = self._csv_tuple(settings.ted_required_python_modules)

    def run(self, req: RunTaskRequest) -> RunTaskResponse:
        graph = self._build_graph(req.execute)
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

    def _build_graph(self, execute: bool):
        try:
            from langgraph.graph import StateGraph, END
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError('langgraph is required. Please install from requirements.txt') from exc

        g = StateGraph(AgentState)
        g.add_node('prepare', self._prepare)
        g.add_node('retrieve', self._retrieve)
        g.add_node('codegen', self._codegen)
        if execute:
            g.add_node('preflight', self._preflight)
            g.add_node('execute', self._execute)

        g.set_entry_point('prepare')
        g.add_edge('prepare', 'retrieve')
        g.add_edge('retrieve', 'codegen')
        if execute:
            g.add_edge('codegen', 'preflight')
            g.add_edge('preflight', 'execute')
            g.add_edge('execute', END)
        else:
            g.add_edge('codegen', END)
        return g.compile()

    def _prepare(self, state: AgentState) -> AgentState:
        task_id = new_task_id()
        self.deps.work_dir.mkdir(parents=True, exist_ok=True)
        (self.deps.work_dir / task_id).mkdir(parents=True, exist_ok=True)
        return {'task_id': task_id}

    def _retrieve(self, state: AgentState) -> AgentState:
        req = state['request']
        chunks = self.indexer.index()
        if not chunks:
            raise RuntimeError(f'resource indexing failed: no eligible documents found in {self.deps.resource_dir}')
        fingerprint = self.indexer.fingerprint()
        rerank_top_n = max(20, req.top_k * max(1, settings.rerank_topn_factor))

        # stage-1: embedding retrieval (fallback to lexical if api unavailable)
        retrieval_warning = ''
        try:
            emb = EmbeddingRetriever.from_chunks(
                self.embedding_client,
                settings.embedding_model_text,
                chunks,
                self.vector_store,
                fingerprint,
            )
            emb_hits = emb.search(req.query, top_k=rerank_top_n)
            candidates = [h.chunk for h in emb_hits]
            if not candidates:
                candidates = chunks
                retrieval_warning = 'embedding retrieval returned no hits; using lexical fallback'
        except Exception as exc:  # noqa: BLE001
            candidates = chunks
            retrieval_warning = f'embedding retrieval unavailable: {exc}; using lexical fallback'

        # stage-2: rerank model (fallback to lexical rerank)
        reranked: List[ScoredChunk]
        if settings.rerank_enabled and candidates:
            try:
                docs = [c.text for c in candidates]
                ranked = self.rerank_client.rerank(
                    model=settings.rerank_model_text,
                    query=req.query,
                    documents=docs,
                    top_n=min(len(docs), rerank_top_n),
                )
                reranked = []
                for idx, score in ranked:
                    if 0 <= idx < len(candidates):
                        reranked.append(ScoredChunk(chunk=candidates[idx], score=score))
                if not reranked:
                    raise RuntimeError('rerank returned no usable entries')
                reranked = reranked[:req.top_k]
            except Exception as exc:  # noqa: BLE001
                if retrieval_warning:
                    retrieval_warning = f"{retrieval_warning}; rerank unavailable: {exc}; using lexical rerank fallback"
                else:
                    retrieval_warning = f"rerank unavailable: {exc}; using lexical rerank fallback"
                hybrid = HybridRetriever(candidates)
                reranked = hybrid.retrieve(req.query, top_k=req.top_k)
        else:
            hybrid = HybridRetriever(candidates)
            reranked = hybrid.retrieve(req.query, top_k=req.top_k)
        return {'chunks': chunks, 'evidence': reranked, 'retrieval_warning': retrieval_warning}

    def _codegen(self, state: AgentState) -> AgentState:
        req = state['request']
        try:
            script = self.codegen.generate(req.query, req.circuit_description, state['evidence'], execute=req.execute)
        except CodegenError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise CodegenError(f'code generation failed: {exc}') from exc
        task_dir = self.deps.work_dir / state['task_id']
        local_script = task_dir / 'main.py'
        local_script.write_text(script.code, encoding='utf-8')
        next_state: AgentState = {'generated_code': script.code}
        if not req.execute:
            next_state['run_status'] = 'generated'
            next_state['run_metrics'] = {}
            next_state['run_stdout'] = ''
            next_state['run_stderr'] = state.get('retrieval_warning', '')
        return next_state

    def _preflight(self, state: AgentState) -> AgentState:  # noqa: ARG002
        if settings.execution_mode == 'ssh':
            runner = SSHRunner(self.deps.ssh_cfg)
        else:
            runner = LocalTedRunner(
                settings.local_bashrc,
                required_commands=self.required_commands,
                required_python_modules=self.required_python_modules,
            )

        out = runner.preflight()
        if not out.ok:
            detail = out.error or out.stderr.strip() or out.stdout.strip() or 'unknown preflight failure'
            raise RuntimeError(f'preflight failed: {detail}')
        return {}

    def _execute(self, state: AgentState) -> AgentState:
        task_dir = self.deps.work_dir / state['task_id']
        local_script = task_dir / 'main.py'
        if settings.execution_mode == 'ssh':
            runner = SSHRunner(self.deps.ssh_cfg)
        else:
            runner = LocalTedRunner(
                settings.local_bashrc,
                required_commands=self.required_commands,
                required_python_modules=self.required_python_modules,
            )

        out = runner.run_script(local_script=local_script, task_id=state['task_id'])
        stderr = out.stderr
        if state.get('retrieval_warning'):
            stderr = f"{state['retrieval_warning']}\n{stderr}".strip()
        return {
            'run_status': out.status,
            'run_metrics': out.metrics,
            'run_stdout': out.stdout,
            'run_stderr': stderr,
            'run_error': out.error,
        }

    @staticmethod
    def _csv_tuple(raw: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
        values = tuple(part.strip() for part in raw.split(',') if part and part.strip())
        return values or default
