# RagAgentEDA (Phase-1 Demo)

后端 Demo：`Embedding RAG -> LangGraph Agent流程 -> 代码生成 -> 执行 -> 回传指标`。

## 当前实现概览
1. **Resource 全量向量化代码在哪？**
   - `backend/rag/indexer.py`：扫描 `Resource/` 下文档并切块
   - `backend/rag/vector_store.py`：调用 embedding / rerank 相关客户端并维护向量索引
   - `backend/agents/orchestrator.py` `_retrieve`：任务中检索与二阶段重排

2. **向量化后保存在哪？**
   - 默认目录：`./workdir/vector_index/`
   - 文件名：`<fingerprint>.json`
   - 最新索引指针：`./workdir/vector_index/LATEST`
   - 可通过 `RAG_VECTOR_INDEX_DIR` 修改路径

3. **支持的语料文件类型与切块策略**
   - 文件类型：`md/markdown/html/htm/txt`
   - HTML：正文抽取 + 模板噪声过滤 + 标题层级分段
   - 切块：heading 感知分段后滑窗切块，`size=1200`、`overlap=200`
   - 过滤：长度 `<50` 的片段不入索引

## API Base 配置（OpenAI兼容）
- Chat/Base（用于 `MODEL_NAME`，如 `deepseek-v3-1-terminus`）: `OPENAI_API_BASE=https://ark.cn-beijing.volces.com/api/v3`
- Embedding/Base（用于 `EMBEDDING_MODEL_TEXT`）: `EMBEDDING_API_BASE=https://a.fe8.cn/v1`
- Chat/Key: `OPENAI_API_KEY=...`
- Embedding/Key: `EMBEDDING_API_KEY=...`（未设置时回退到 `OPENAI_API_KEY`）
- Embedding 批处理参数：
  - `EMBEDDING_BATCH_SIZE`（默认 10）
  - `EMBEDDING_RETRY_COUNT`（默认 4）
  - `EMBEDDING_RETRY_DELAY_SEC`（默认 1.0）
- Rerank（二阶段重排）参数：
  - `RERANK_ENABLED`（默认 `true`）
  - `RERANK_MODEL_TEXT`（默认 `qwen3-rerank`）
  - `RERANK_API_BASE`（默认回退到 `EMBEDDING_API_BASE`）
  - `RERANK_API_KEY`（默认回退到 `EMBEDDING_API_KEY`）
  - `RERANK_TOPN_FACTOR`（默认 `4`，`top_n=max(20, top_k*factor)`）
- Chat: `POST /chat/completions`
- Embedding: `POST /embeddings`
- Rerank: `POST /rerank`
- Model 列表检查: `GET /models`
- Header: `Authorization: Bearer <对应 Base 的 Key>`
  - Chat base 用 `OPENAI_API_KEY`
  - Embedding base 用 `EMBEDDING_API_KEY`（未配置时回退 `OPENAI_API_KEY`）
  - Rerank base 用 `RERANK_API_KEY`（未配置时回退 `EMBEDDING_API_KEY`）

项目客户端实现见：`backend/llm/client.py`

## Embedding 模型默认值与排障
- 默认文本 embedding 模型：`EMBEDDING_MODEL_TEXT=text-embedding-v4`
- 推荐先检查当前 key 可用模型：
  ```bash
  curl -H "Authorization: Bearer <OPENAI_API_KEY>" \
    https://a.fe8.cn/v1/models
  ```
- 如果出现 `InvalidEndpointOrModel.NotFound`：
  - 说明该 key 对该模型未开通或模型名不匹配
  - 将 `.env` 的 `EMBEDDING_MODEL_TEXT` 改为你账号实际可用的 embedding 模型
  - 再执行 `test_api.py` 或 `/v1/rag/reindex` 验证连通性

## 启动
```bash
pip install -r requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

## 健康检查
```bash
curl http://127.0.0.1:8000/health
```

返回会包含：
- `model/model_api_base/model_api_key_set`
- `embedding_model/embedding_api_base/embedding_api_key_set`
- `rerank_enabled/rerank_model/rerank_api_base/rerank_api_key_set`
- `vector_index_latest`

## 手工重建向量索引
```bash
curl -X POST 'http://127.0.0.1:8000/v1/rag/reindex'
```

返回中会包含：
- `doc_count`
- `chunk_count`
- `vector_count`
- `fingerprint`
- `saved_file`

当前 `reindex` 行为：
- 强制重建索引（不复用旧缓存）
- 向量有效性校验：`vector_count > 0` 且 `vector_count == chunk_count`
- 不满足则接口返回 500
- 索引文件包含元信息：`embedding_model`、`embedding_api_base`、`created_at`

## 调用示例
```bash
curl -X POST 'http://127.0.0.1:8000/v1/tasks/run' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "请测试该运放电路的带宽",
    "circuit_description": "两级运放，输出带负载",
    "top_k": 6,
    "execute": false
  }'
```

当 `execute=false` 时，接口只做检索和代码生成，返回 `generated_code + evidence`，不执行脚本。

检索流程（`execute=false`）：
- stage-1：向量召回（embedding）
- stage-2：`qwen3-rerank` 二阶段重排
- rerank 失败：回退 lexical rerank，并在 `logs.stderr` 留 warning

## 执行模式
- `RAG_EXECUTION_MODE=local`：本机 `source ~/.bashrc` 后执行。
- `RAG_EXECUTION_MODE=ssh`：通过 ssh/scp 到远端执行。

## 注意
- `execute=true` 时，当前仅 `opamp_bandwidth` 接入真实模板执行（依赖 `TED_BANDWIDTH_CMD`）。
- 其它测试类型（如 SFDR/环路增益）尚未接入真实模板执行。
- `execute=false` 仅做检索+代码生成，不执行脚本。

## 检索回归脚本
已内置回归用例与脚本：
- `workdir/retrieval_regression_cases.json`
- `workdir/run_retrieval_regression.py`

运行：
```bash
python workdir/run_retrieval_regression.py
```

验收规则：
- 核心用例：必须全部 PASS
- 扩展用例：至少 1 条 PASS
- 若出现 `embedding retrieval unavailable` 或 `using lexical fallback`，会被判为检索路径降级
