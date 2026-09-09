# TokenOps 智能运营平台需求文档

编制日期：2026-06-09  
版本：v0.1 MVP  
适用周期：10 天二次开发  
基础项目：`RagAgentEDA`

## 1. 产品名称

**TokenOps 智能运营平台**

副标题：**大模型成本、限流与故障定位助手**

产品定位：  
面向江苏电信内部 AI 应用和智能体的运营治理平台，通过采集大模型调用日志，完成 Token 消耗统计、成本核算、异常检测、故障定位、限流建议、模型路由建议和运营报告生成。

## 2. 建设目标

### 2.1 MVP 目标

1. 支持导入模型调用日志。
2. 自动计算 Token、成本、成功率、延迟等指标。
3. 支持按应用、部门、地市、模型维度钻取。
4. 识别 Token 异常消耗、失败率异常、延迟异常、上下文超长、重试异常等问题。
5. 通过 RAG 问答解释异常原因和处理建议。
6. 生成限流、降级、Prompt 优化和预算控制建议。
7. 输出成本分摊账单和异常分析报告。
8. 形成可路演的端到端闭环。

### 2.2 非目标

MVP 阶段不做以下事项：

- 不直接改造真实生产模型网关。
- 不接入真实计费系统。
- 不实现复杂权限体系。
- 不做实时流式日志处理。
- 不自动下发限流策略到生产环境。
- 不做跨省多租户 SaaS 化部署。

这些能力作为后续扩展。

## 3. 技术栈

### 3.1 现有项目技术栈

| 层级 | 技术 |
| --- | --- |
| 后端框架 | FastAPI |
| 前端演示 | Gradio |
| 数据模型 | Pydantic |
| 本地存储 | SQLite、JSON 文件 |
| RAG | 文档切块、Embedding、Rerank、检索证据 |
| 大模型接口 | OpenAI 兼容接口 |
| 任务编排 | 当前项目内置 Pipeline |
| 执行器 | Local Runner、SSH Runner |

### 3.2 TokenOps 新增技术栈

| 层级 | 建议技术 |
| --- | --- |
| 日志导入 | CSV/JSON 文件上传 |
| 指标计算 | Python 聚合逻辑，MVP 使用内存 + SQLite |
| 异常检测 | 规则引擎 + 环比/同比阈值 |
| 报告生成 | Markdown/HTML 模板 + LLM 摘要 |
| 可视化 | Gradio DataFrame、Plot、HTML 卡片 |
| 配置管理 | `.env` + YAML/JSON 配置 |
| 测试 | pytest 或脚本级 smoke check |

### 3.3 推荐目录结构

```text
backend/
  tokenops/
    __init__.py
    schemas.py
    store.py
    importer.py
    metrics.py
    anomaly.py
    advisor.py
    report.py
    sample_data.py
  ui/
    gradio_tokenops.py
Resource/
  tokenops/
    token运营规范.md
    模型调用错误码.md
    限流策略SOP.md
    Prompt优化指南.md
workdir/
  tokenops/
    tokenops.db
    reports/
    sample_logs/
```

说明：MVP 可以与现有 `RagAgentEDA` 共用 RAG 能力，也可以先独立建立 `Resource/tokenops/` 运营知识库。

## 4. 用户角色

| 角色 | 核心诉求 | MVP 权限 |
| --- | --- | --- |
| 省公司 AI 运营管理员 | 看全省总览、定位异常、输出报告 | 查看全部数据 |
| 地市 AI 运营人员 | 查看本地市应用成本和异常 | MVP 中用筛选模拟 |
| 应用负责人 | 查看自己应用的调用质量和成本 | MVP 中用应用筛选模拟 |
| 财务/经营人员 | 查看成本分摊和预算超限 | 查看成本报表 |
| 开发/运维人员 | 定位错误码、优化 Prompt、调整策略 | 查看异常详情和建议 |

MVP 阶段可不做登录，只提供筛选条件模拟不同角色视图。

## 5. 数据需求

### 5.1 模型调用日志字段

MVP 建议字段如下：

| 字段 | 类型 | 说明 | 示例 |
| --- | --- | --- | --- |
| request_id | string | 请求唯一 ID | req_20260609_0001 |
| timestamp | datetime | 调用时间 | 2026-06-09 09:30:00 |
| app_id | string | 应用 ID | app_customer_qa |
| app_name | string | 应用名称 | 客服知识助手 |
| department | string | 部门 | 客服中心 |
| city | string | 地市 | 南京 |
| scenario | string | 场景 | 客服问答 |
| model_name | string | 模型名称 | deepseek-v4-pro |
| model_tier | string | 模型档位 | high/mid/small |
| prompt_template | string | Prompt 模板 ID | policy_summary_v3 |
| input_tokens | int | 输入 Token | 3200 |
| output_tokens | int | 输出 Token | 850 |
| total_tokens | int | 总 Token | 4050 |
| latency_ms | int | 响应耗时 | 2300 |
| status | string | success/error/timeout | success |
| error_code | string | 错误码 | context_length_exceeded |
| retry_count | int | 重试次数 | 0 |
| user_type | string | 用户类型 | employee/customer |
| trace_id | string | 链路追踪 ID | trace_xxx |

### 5.2 模型价格配置

MVP 使用配置化单价：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| model_name | string | 模型名称 |
| input_price_per_1k | float | 输入 Token 每千价格 |
| output_price_per_1k | float | 输出 Token 每千价格 |
| currency | string | 币种 |
| effective_date | date | 生效日期 |

成本计算公式：

```text
cost = input_tokens / 1000 * input_price_per_1k
     + output_tokens / 1000 * output_price_per_1k
```

### 5.3 应用主数据

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| app_id | string | 应用 ID |
| app_name | string | 应用名称 |
| owner | string | 负责人 |
| department | string | 归属部门 |
| city | string | 地市 |
| priority | string | high/medium/low |
| monthly_budget | float | 月预算 |
| business_value | string | high/medium/low |

## 6. 模块功能说明

### 6.1 数据导入模块

功能目标：导入大模型调用日志和模型价格配置。

MVP 功能：

- 支持上传 CSV 日志。
- 支持上传 JSON 日志。
- 支持生成模拟日志。
- 校验必填字段。
- 自动补齐 `total_tokens`。
- 将日志写入 SQLite。
- 展示导入成功数量、失败数量、错误原因。

验收标准：

- 上传 1000 条样例日志可正常入库。
- 缺失关键字段时给出明确错误。
- 重复 `request_id` 可跳过或覆盖，策略可配置。

### 6.2 指标总览模块

功能目标：展示 AI 应用整体运营态势。

核心指标：

- 总调用量。
- 总 Token。
- 输入 Token。
- 输出 Token。
- 总成本。
- 成功率。
- 平均延迟。
- P95 延迟。
- 异常请求数。
- 活跃应用数。

筛选维度：

- 日期范围。
- 应用。
- 部门。
- 地市。
- 模型。
- 场景。

页面展示：

- 顶部 KPI 卡片。
- Token 趋势图。
- 成本趋势图。
- 应用排行表。
- 模型使用分布。

### 6.3 成本分析模块

功能目标：回答“钱花在哪、谁在花、是否值得”。

功能点：

- 按应用统计成本。
- 按部门统计成本。
- 按地市统计成本。
- 按模型统计成本。
- 按 Prompt 模板统计成本。
- 识别高成本低价值应用。
- 识别单次请求成本异常。
- 输出成本分摊账单。

建议指标：

| 指标 | 说明 |
| --- | --- |
| app_cost | 应用总成本 |
| avg_cost_per_request | 单次请求平均成本 |
| cost_per_success | 单次成功请求成本 |
| token_per_request | 单次请求平均 Token |
| budget_usage_rate | 预算使用率 |

策略建议：

- 高成本摘要类任务优先降级到中小模型。
- 超长输入先做摘要压缩。
- 低优先级应用设置日预算。
- 失败重试高的应用设置熔断。

### 6.4 异常检测模块

功能目标：自动识别大模型运营异常。

MVP 异常规则：

| 异常类型 | 判定逻辑 | 输出建议 |
| --- | --- | --- |
| Token 突增 | 当前周期 Token 较上一周期增长超过阈值 | 检查业务活动、循环调用、Prompt 变更 |
| 成本超预算 | 预算使用率超过 80%/100% | 启动预算预警或限流 |
| 失败率升高 | 错误率超过阈值 | 查看错误码和模型状态 |
| 延迟异常 | P95 延迟超过阈值 | 检查上下文长度和模型排队 |
| 上下文超长 | input_tokens 超过阈值 | 增加摘要压缩或截断 |
| 重试异常 | retry_count 平均值升高 | 检查应用重试策略 |
| 高价模型滥用 | 低价值场景频繁调用高档模型 | 推荐模型路由降级 |

异常输出字段：

- anomaly_id。
- anomaly_type。
- severity。
- app_id。
- time_window。
- evidence_metrics。
- suspected_reason。
- recommended_action。

### 6.5 故障定位模块

功能目标：将异常指标转化为可执行的排查结论。

输入：

- 异常检测结果。
- 原始调用日志片段。
- 错误码统计。
- 应用主数据。
- Token 运营知识库检索结果。

输出：

- 异常摘要。
- 影响范围。
- 疑似根因。
- 证据链。
- 建议处理动作。
- 建议工单内容。

示例：

```text
异常：客服知识助手失败率异常升高
影响：南京、苏州两地客服问答场景
根因推断：context_length_exceeded 占错误请求 74%，集中在 policy_summary_v3 Prompt
证据：平均输入 Token 从 4200 升至 18500，P95 延迟从 3.2s 升至 11.6s
建议：增加文档摘要压缩步骤，限制单次输入不超过 12000 Token，失败重试从 3 次降至 1 次
```

### 6.6 限流策略建议模块

功能目标：根据业务优先级和异常状态生成限流建议。

建议类型：

1. 全局限流。
2. 应用级限流。
3. 部门级预算限流。
4. Prompt 模板级限流。
5. 异常重试熔断。
6. 高峰时段配额保护。

MVP 输出策略 JSON：

```json
{
  "strategy_id": "limit_app_customer_qa_20260609",
  "target": {
    "type": "app",
    "app_id": "app_customer_qa"
  },
  "rule": {
    "max_qps": 20,
    "daily_token_budget": 5000000,
    "max_retry_count": 1
  },
  "reason": "Token cost increased by 85% and retry_count is abnormal",
  "expected_effect": "reduce daily token cost by 20%-35%"
}
```

### 6.7 模型路由建议模块

功能目标：减少高成本模型误用。

模型路由规则：

- 简单分类、格式转换、摘要草稿可用小模型。
- 长文本复杂推理使用中高档模型。
- 高价值客户、投诉处理、政企场景优先保障高档模型。
- 超长文档先摘要，再进入高档模型。
- 批处理离线任务可安排低峰时段。

输出：

- 当前模型使用是否合理。
- 推荐目标模型。
- 预估节省成本。
- 风险说明。

### 6.8 Prompt 优化建议模块

功能目标：发现高成本、低效果或易失败的 Prompt。

指标：

- Prompt 模板调用量。
- 平均输入 Token。
- 平均输出 Token。
- 失败率。
- 用户反馈。
- 单次平均成本。

建议动作：

- 删除重复上下文。
- 增加输入摘要。
- 固定输出格式。
- 限制回答长度。
- 拆分多目标 Prompt。
- 引入缓存。

### 6.9 RAG 智能问答模块

功能目标：让运营人员用自然语言查询运营数据和处理策略。

问题示例：

- “昨天哪个应用 Token 消耗最高？”
- “南京地市本周成本上涨的原因是什么？”
- “哪些 Prompt 应该优化？”
- “客服助手失败率升高怎么处理？”
- “低优先级应用如何设置限流？”

回答结构：

1. 直接结论。
2. 关键数据。
3. 异常原因。
4. 建议动作。
5. 证据来源。

实现方式：

- 日志指标先由规则聚合为结构化摘要。
- RAG 检索 Token 运营知识库。
- 大模型基于“指标摘要 + 检索证据”生成解释。

### 6.10 报告生成模块

功能目标：一键输出管理层可读报告。

报告类型：

- Token 运营日报。
- 应用成本分摊报告。
- 异常分析报告。
- 限流策略建议报告。
- 模型路由优化报告。

报告格式：

- MVP：Markdown。
- 可选：HTML。
- 后续：Word/PDF。

报告结构：

```text
1. 本期概览
2. 核心指标
3. 成本排行
4. 异常事件
5. 根因分析
6. 策略建议
7. 预期收益
8. 后续跟踪项
```

### 6.11 系统管理模块

MVP 功能：

- 模型价格配置。
- 应用主数据配置。
- 异常阈值配置。
- 样例数据重置。

后续功能：

- 用户权限。
- 操作审计。
- 策略审批。
- 网关配置下发。

## 7. 页面需求

### 7.1 页面 1：TokenOps 工作台

内容：

- 时间范围选择。
- 总调用量、总 Token、总成本、成功率、P95 延迟。
- Token 趋势。
- 成本趋势。
- 应用成本 TOP10。
- 异常事件列表。

### 7.2 页面 2：成本分摊

内容：

- 按部门/地市/应用切换。
- 成本排行榜。
- 预算使用率。
- 高成本 Prompt 列表。
- 导出账单按钮。

### 7.3 页面 3：异常诊断

内容：

- 异常列表。
- 异常详情。
- 证据指标。
- 疑似根因。
- 建议动作。
- 生成工单文本。

### 7.4 页面 4：策略建议

内容：

- 限流建议。
- 模型降级建议。
- Prompt 优化建议。
- 预算控制建议。
- 策略 JSON 预览。

### 7.5 页面 5：智能问答与报告

内容：

- 自然语言问题输入。
- RAG 回答。
- 证据来源。
- 生成日报/异常报告按钮。
- Markdown 报告预览。

## 8. API 需求

### 8.1 日志导入

```http
POST /v1/tokenops/logs/import
```

请求：

- multipart 文件上传，支持 CSV/JSON。

响应：

```json
{
  "ok": true,
  "imported": 1000,
  "failed": 0,
  "message": "import success"
}
```

### 8.2 生成样例数据

```http
POST /v1/tokenops/sample-data
```

响应：

```json
{
  "ok": true,
  "generated": 5000
}
```

### 8.3 指标总览

```http
GET /v1/tokenops/metrics/overview
```

查询参数：

- start_date。
- end_date。
- app_id。
- department。
- city。
- model_name。

### 8.4 成本分析

```http
GET /v1/tokenops/costs
```

返回：

- 成本汇总。
- 分组排行。
- 高成本 Prompt。
- 预算使用率。

### 8.5 异常检测

```http
POST /v1/tokenops/anomalies/detect
```

返回：

- 异常列表。
- 严重级别。
- 证据指标。
- 建议动作。

### 8.6 异常解释

```http
POST /v1/tokenops/anomalies/explain
```

请求：

```json
{
  "anomaly_id": "anomaly_001"
}
```

响应：

```json
{
  "summary": "客服知识助手失败率异常升高",
  "root_cause": "上下文超长导致 context_length_exceeded",
  "evidence": [],
  "recommendations": []
}
```

### 8.7 智能问答

```http
POST /v1/tokenops/ask
```

请求：

```json
{
  "question": "本周哪个应用成本异常，原因是什么？"
}
```

### 8.8 报告生成

```http
POST /v1/tokenops/reports/generate
```

请求：

```json
{
  "report_type": "daily",
  "start_date": "2026-06-01",
  "end_date": "2026-06-09"
}
```

响应：

```json
{
  "ok": true,
  "report_path": "workdir/tokenops/reports/daily_20260609.md",
  "markdown": "# Token 运营日报..."
}
```

## 9. 核心算法和规则

### 9.1 成本计算

```text
input_cost = input_tokens / 1000 * input_price_per_1k
output_cost = output_tokens / 1000 * output_price_per_1k
total_cost = input_cost + output_cost
```

### 9.2 成功率

```text
success_rate = success_requests / total_requests
```

### 9.3 P95 延迟

对筛选范围内 `latency_ms` 排序后取 95 分位。

### 9.4 Token 突增

```text
growth_rate = (current_tokens - baseline_tokens) / baseline_tokens
```

当 `growth_rate > threshold` 时判定异常。MVP 默认阈值 50%。

### 9.5 预算超限

```text
budget_usage_rate = period_cost / monthly_budget
```

- 大于 80%：预警。
- 大于 100%：严重。

### 9.6 高价模型滥用

判定条件示例：

```text
model_tier = high
AND business_value = low
AND avg_input_tokens < 1500
AND success_rate > 95%
```

建议：降级到中小模型。

## 10. 非功能需求

### 10.1 性能

MVP：

- 支持 1 万条日志本地导入和分析。
- 指标查询响应小于 3 秒。
- 报告生成小于 30 秒。

后续：

- 支持百万级日志。
- 支持异步任务和缓存。

### 10.2 可靠性

- 导入失败不影响已有数据。
- 大模型不可用时，报告生成使用模板兜底。
- RAG 检索失败时，异常检测仍可基于规则输出。

### 10.3 安全

MVP：

- 不导入用户敏感明文。
- 样例数据全部脱敏或模拟。
- 日志中 Prompt 内容可只保存模板 ID。

后续：

- 接入统一认证。
- 数据分级授权。
- 操作审计。
- 敏感字段脱敏。

### 10.4 可扩展性

- 日志字段可扩展。
- 模型价格可配置。
- 异常规则可配置。
- 策略建议模板可扩展。
- 后续可接入真实模型网关。

## 11. MVP 验收标准

| 编号 | 验收项 | 标准 |
| --- | --- | --- |
| A1 | 样例数据 | 可生成不少于 5000 条调用日志 |
| A2 | 日志导入 | CSV/JSON 至少支持一种 |
| A3 | 指标看板 | 展示调用量、Token、成本、成功率、延迟 |
| A4 | 多维分析 | 支持应用、部门、地市、模型维度 |
| A5 | 成本分摊 | 可输出应用和部门成本排行 |
| A6 | 异常检测 | 至少支持 5 类异常 |
| A7 | 故障定位 | 可对异常生成原因解释和处理建议 |
| A8 | 策略建议 | 可生成限流、降级、Prompt 优化建议 |
| A9 | 智能问答 | 可回答至少 5 个运营问题 |
| A10 | 报告生成 | 可生成 Markdown 异常报告 |
| A11 | 路演闭环 | 可完成“导入日志 -> 发现异常 -> 解释原因 -> 输出策略 -> 生成报告” |

## 12. 10 天开发计划

| 天数 | 工作内容 | 交付物 |
| --- | --- | --- |
| 第 1 天 | 梳理字段、准备知识库、设计样例数据 | 字段表、知识库文档、样例日志方案 |
| 第 2 天 | 实现 TokenOps schemas/store/importer | SQLite 表、导入接口 |
| 第 3 天 | 实现 metrics/costs 聚合 | 指标 API、成本 API |
| 第 4 天 | 实现 anomaly 规则引擎 | 异常检测 API |
| 第 5 天 | 实现 advisor 故障解释和策略建议 | 异常解释、策略 JSON |
| 第 6 天 | 实现 report 生成 | Markdown 报告 |
| 第 7 天 | 开发 Gradio TokenOps 页面 | 工作台 UI |
| 第 8 天 | 打通端到端演示流程 | 演示脚本、样例场景 |
| 第 9 天 | 路演材料与录屏 | PPT 素材、备份视频 |
| 第 10 天 | 联调、修复、彩排 | 可交付版本 |

## 13. 路演演示脚本

建议使用一个清晰的故事线：

### 场景设定

江苏电信内部已有客服助手、装维助手、办公助手、云网运维助手、政企方案助手等多个 AI 应用。最近全省 Token 成本上涨，部分应用出现响应变慢和失败率升高。

### 演示步骤

1. 进入 TokenOps 工作台，展示总调用量、总 Token、总成本。
2. 发现“客服知识助手”成本环比上涨 85%。
3. 点击异常详情，系统显示失败率、延迟、错误码和 Prompt 模板分布。
4. 系统定位根因：某 Prompt 上下文过长，导致失败和重试放大。
5. 系统生成建议：上下文摘要压缩、重试降级、高峰限流、部分请求切换中小模型。
6. 一键生成异常分析报告和成本分摊账单。
7. 结尾强调：TokenOps 不是单个 AI 应用，而是治理所有 AI 应用的运营底座。

## 14. 与现有项目的映射关系

| 现有能力 | TokenOps 复用方式 |
| --- | --- |
| `backend/rag/indexer.py` | 构建 Token 运营知识库 |
| `backend/rag/vector_store.py` | 运营规则和故障 SOP 检索 |
| `backend/agents/qa_agent.py` | TokenOps 智能问答 |
| `backend/agents/query_rewriter.py` | 优化运营问题检索表达 |
| `backend/storage/qa_feedback_store.py` | 复用反馈和历史思路 |
| `backend/app.py` | 增加 TokenOps API |
| `backend/ui/gradio_ragagent.py` | 参考 Gradio 工作台布局 |
