# RagAgentEDA (Phase-1 Demo)

后端 Demo：`Embedding RAG -> LangGraph Agent流程 -> 代码生成 -> 执行 -> 回传指标`。

## 你问的两个关键点
1. **Resource 全量向量化代码在哪？**
   - `backend/rag/indexer.py`：扫描 `Resource/` 下 `md/html/txt` 并切块
   - `backend/rag/vector_store.py`：调用方舟 embedding 做向量化，并持久化到本地
   - `backend/agents/orchestrator.py` `_retrieve`：任务中自动加载/构建向量索引

2. **向量化后保存在哪？**
   - 默认目录：`./workdir/vector_index/`
   - 文件名：`<fingerprint>.json`
   - 最新索引指针：`./workdir/vector_index/LATEST`
   - 可通过 `RAG_VECTOR_INDEX_DIR` 修改路径

## Ark(火山方舟)调用规范（OpenAI兼容）
- Base URL: `https://ark.cn-beijing.volces.com/api/v3`
- Chat: `POST /chat/completions`
- Embedding: `POST /embeddings`
- Model 列表检查: `GET /models`
- Header: `Authorization: Bearer <OPENAI_API_KEY>`

项目客户端实现见：`backend/llm/client.py`

## 启动
```bash
pip install -r requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

## 手工重建向量索引
```bash
curl -X POST 'http://127.0.0.1:8000/v1/rag/reindex'
```

## 调用示例
```bash
curl -X POST 'http://127.0.0.1:8000/v1/tasks/run' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "请测试该运放电路的带宽",
    "circuit_description": "两级运放，输出带负载",
    "top_k": 6
  }'
```

## 执行模式
- `RAG_EXECUTION_MODE=local`：本机 `source ~/.bashrc` 后执行。
- `RAG_EXECUTION_MODE=ssh`：通过 ssh/scp 到远端执行。

## 注意
当前 `backend/agents/codegen.py` 仍是 demo 模板，需要你按 Resource 手册把 TODO 段替换为真实 pyted API 调用映射。
