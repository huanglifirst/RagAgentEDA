from __future__ import annotations

from datetime import datetime
import json
from html import escape
import os
import re
from typing import Any, List, Tuple

from fastapi import FastAPI

from backend.agents.query_rewriter import QueryRewriteResult, QueryRewriter
from backend.agents.qa_agent import RagQaAgent
from backend.config import settings
from backend.schemas.api import EvidenceItem, RagAskResponse
from backend.storage import QaFeedbackStore

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

_CSS = """
body, .gradio-container {
  background: #f6f7fb !important;
  color: #111827 !important;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
}
.gradio-container {
  max-width: none !important;
}
.status-badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  border: 1px solid #d1d5db;
  background: #f3f4f6;
  color: #1f2937;
}
.status-answered { border-color: #c7f0d8; background: #ecfdf3; color: #116149; }
.status-not-found { border-color: #fcd9aa; background: #fff7ed; color: #9a3412; }
.status-error { border-color: #fecaca; background: #fef2f2; color: #991b1b; }
.card {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #ffffff;
  padding: 12px;
}
.section-title {
  margin: 4px 0 8px;
  font-size: 28px;
  font-weight: 700;
  color: #111827;
}
.evidence-title {
  margin: 14px 0 10px;
  font-size: 18px;
  font-weight: 700;
  color: #111827;
}
.evidence-item {
  margin-top: 12px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #ffffff;
}
.evidence-summary {
  list-style: none;
  cursor: pointer;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  padding: 12px;
  color: #111827;
  user-select: none;
}
.evidence-summary:hover {
  background: #f8fafc;
}
.evidence-summary::-webkit-details-marker {
  display: none;
}
.evidence-summary::marker {
  content: "";
}
.evidence-summary::before {
  content: "▸";
  color: #64748b;
  font-size: 12px;
  transform-origin: center;
  transition: transform 0.15s ease;
}
.evidence-item[open] .evidence-summary::before {
  transform: rotate(90deg);
}
.evidence-item[open] .evidence-summary {
  border-bottom: 1px solid #e5e7eb;
}
.evidence-content {
  padding: 12px;
}
.meta-line {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  color: #111827;
}
.chip {
  display: inline-block;
  max-width: 100%;
  padding: 2px 8px;
  border-radius: 6px;
  border: 1px solid #d1d5db;
  background: #f8fafc;
  color: #1f2937;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 12px;
  overflow-wrap: anywhere;
}
.snippet-block {
  margin: 0;
  white-space: pre-wrap;
  overflow-x: auto;
  padding: 12px;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
  background: #f8fafc;
  color: #0f172a;
  font-size: 14px;
  line-height: 1.55;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
}
.answer-md {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #ffffff;
  padding: 12px 14px;
}
.answer-md :is(h1, h2, h3, h4) {
  margin: 10px 0 8px;
  color: #111827;
}
.answer-md :is(p, ul, ol) {
  margin: 8px 0;
}
.answer-md code {
  background: #f3f4f6;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  padding: 0 4px;
}
.answer-md pre code {
  display: block;
  white-space: pre-wrap;
  padding: 10px 12px;
  border-radius: 8px;
}
.snippet-container {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #f8fafc;
  padding: 10px 12px;
}
.snippet-text {
  white-space: pre-wrap;
  color: #111827;
  font-size: 14px;
  line-height: 1.6;
}
.snippet-table-wrap {
  overflow-x: auto;
}
.snippet-table {
  width: 100%;
  border-collapse: collapse;
  background: #ffffff;
  color: #111827;
  font-size: 13px;
  line-height: 1.5;
}
.snippet-table th,
.snippet-table td {
  border: 1px solid #d1d5db;
  padding: 6px 8px;
  text-align: left;
  vertical-align: top;
  white-space: pre-wrap;
  word-break: break-word;
}
.snippet-table thead th {
  background: #f1f5f9;
  font-weight: 600;
}
.snippet-code {
  margin: 0;
  white-space: pre;
  overflow-x: auto;
  overflow-y: hidden;
  overflow-wrap: normal;
  word-break: normal;
  background: #eef2f7;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  padding: 10px 12px;
  color: #0f172a;
  font-size: 14px;
  line-height: 1.55;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
}
.snippet-sep {
  height: 10px;
}
.warning {
  border: 1px solid #f7d9a2;
  background: #fff9ed;
  color: #8a4a03;
  border-radius: 8px;
  padding: 10px 12px;
  margin-bottom: 12px;
}

/* RagAgent professional workbench redesign */
:root {
  --ra-bg: #eef3f2;
  --ra-bg-soft: #f8faf9;
  --ra-surface: #ffffff;
  --ra-surface-raised: rgba(255, 255, 255, 0.94);
  --ra-ink: #17201f;
  --ra-muted: #667370;
  --ra-line: #d9e2df;
  --ra-line-strong: #bccbc7;
  --ra-accent: #0f766e;
  --ra-accent-2: #2563eb;
  --ra-success: #147c4f;
  --ra-warning: #a25b10;
  --ra-danger: #b42318;
  --ra-radius: 8px;
  --ra-radius-sm: 6px;
  --ra-shadow: 0 18px 42px rgba(31, 48, 45, 0.11);
  --ra-shadow-soft: 0 8px 22px rgba(31, 48, 45, 0.07);
}

body,
.gradio-container {
  background:
    linear-gradient(120deg, rgba(15, 118, 110, 0.09), transparent 34%),
    linear-gradient(270deg, rgba(37, 99, 235, 0.08), transparent 30%),
    var(--ra-bg) !important;
  color: var(--ra-ink) !important;
  font-family: "Aptos", "Segoe UI", "Microsoft YaHei UI", "Helvetica Neue", Arial, sans-serif !important;
}

html,
body {
  width: 100% !important;
  min-height: 100vh !important;
  margin: 0 !important;
  overflow-x: hidden;
}

.gradio-container {
  width: 100% !important;
  max-width: none !important;
  min-height: 100vh !important;
  padding: clamp(14px, 1.25vw, 24px) !important;
  box-sizing: border-box !important;
}

.gradio-container .contain,
.gradio-container .wrap,
.gradio-container main {
  width: 100% !important;
  max-width: none !important;
}

.gradio-container footer {
  display: none !important;
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin: 0 0 16px;
  padding: 16px 18px;
  border: 1px solid var(--ra-line);
  border-radius: var(--ra-radius);
  background: rgba(255, 255, 255, 0.86);
  box-shadow: var(--ra-shadow-soft);
  backdrop-filter: blur(10px);
}

.app-kicker,
.panel-kicker {
  color: var(--ra-accent);
  font-size: 12px;
  font-weight: 760;
  letter-spacing: 0;
  text-transform: uppercase;
}

.app-title {
  margin-top: 3px;
  color: var(--ra-ink);
  font-size: 26px;
  line-height: 1.15;
  font-weight: 780;
}

.app-status {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}

.runtime-pill,
.route-pill {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  min-height: 30px;
  padding: 5px 10px;
  border: 1px solid var(--ra-line);
  border-radius: 999px;
  background: var(--ra-bg-soft);
  color: var(--ra-muted);
  font-size: 12px;
  font-weight: 680;
}

.runtime-dot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: var(--ra-success);
  box-shadow: 0 0 0 4px rgba(20, 124, 79, 0.12);
}

.workspace-layout {
  align-items: stretch !important;
  gap: 16px !important;
  min-height: calc(100vh - 138px);
}

.workspace-panel {
  min-width: 0 !important;
  padding: 16px;
  border: 1px solid var(--ra-line);
  border-radius: var(--ra-radius);
  background: var(--ra-surface-raised);
  box-shadow: var(--ra-shadow);
}

.query-console {
  border-left: 4px solid var(--ra-accent);
}

.result-console {
  border-left: 4px solid var(--ra-accent-2);
}

.panel-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.panel-title {
  margin-top: 2px;
  color: var(--ra-ink);
  font-size: 18px;
  line-height: 1.2;
  font-weight: 760;
}

.panel-tag {
  flex: 0 0 auto;
  padding: 4px 8px;
  border: 1px solid var(--ra-line);
  border-radius: 999px;
  background: #f2f7f6;
  color: var(--ra-muted);
  font-size: 12px;
  font-weight: 650;
}

.action-row {
  gap: 10px !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select {
  border-color: var(--ra-line) !important;
  border-radius: var(--ra-radius-sm) !important;
  background: #fbfdfc !important;
  color: var(--ra-ink) !important;
}

.gradio-container textarea:focus,
.gradio-container input:focus {
  border-color: var(--ra-accent) !important;
  box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.13) !important;
}

.gradio-container input[type="radio"],
.gradio-container input[type="checkbox"] {
  position: relative;
  width: 18px !important;
  height: 18px !important;
  border: 1px solid var(--ra-line-strong) !important;
  background: #ffffff !important;
  accent-color: var(--ra-accent) !important;
}

.gradio-container input[type="radio"]:checked,
.gradio-container input[type="checkbox"]:checked {
  border-color: var(--ra-accent) !important;
  background:
    radial-gradient(circle at center, var(--ra-accent) 0 42%, transparent 46%) !important;
}

.gradio-container label:has(input[type="radio"]:checked),
.gradio-container label:has(input[type="checkbox"]:checked) {
  border-color: var(--ra-accent) !important;
  background: #e7f4f2 !important;
  color: var(--ra-ink) !important;
  box-shadow: inset 0 0 0 1px rgba(15, 118, 110, 0.18), 0 8px 18px rgba(15, 118, 110, 0.10) !important;
}

.gradio-container label:has(input[type="radio"]:checked) *,
.gradio-container label:has(input[type="checkbox"]:checked) * {
  color: var(--ra-ink) !important;
}

.gradio-container button {
  border-radius: var(--ra-radius-sm) !important;
  font-weight: 720 !important;
  transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease, background 0.12s ease;
}

.gradio-container button:hover {
  transform: translateY(-1px);
}

.gradio-container .primary-action button,
.gradio-container button.primary-action {
  border-color: transparent !important;
  background: linear-gradient(135deg, var(--ra-accent), var(--ra-accent-2)) !important;
  color: #ffffff !important;
  box-shadow: 0 10px 20px rgba(15, 118, 110, 0.20);
}

.gradio-container .secondary-action button,
.gradio-container button.secondary-action {
  border-color: var(--ra-line-strong) !important;
  background: #ffffff !important;
  color: var(--ra-ink) !important;
}

.feedback-row {
  gap: 10px !important;
}

.feedback-row button {
  min-width: 104px;
}

.card,
.warning,
.answer-md,
.evidence-item,
.snippet-container,
.snippet-code {
  border-radius: var(--ra-radius) !important;
}

.card {
  border-color: var(--ra-line);
  background: #f9fbfb;
  color: var(--ra-ink);
  box-shadow: none;
}

.warning {
  border-color: #f1c27b;
  background: #fff6e8;
  color: var(--ra-warning);
}

.status-badge {
  padding: 5px 10px;
  border-radius: 999px;
  font-size: 12px;
  letter-spacing: 0;
  text-transform: uppercase;
}

.status-answered {
  border-color: rgba(20, 124, 79, 0.22);
  background: #eaf7f0;
  color: var(--ra-success);
}

.status-not-found {
  border-color: rgba(162, 91, 16, 0.22);
  background: #fff4e5;
  color: var(--ra-warning);
}

.status-error {
  border-color: rgba(180, 35, 24, 0.22);
  background: #fff0ee;
  color: var(--ra-danger);
}

.answer-md {
  min-height: 210px;
  border-color: var(--ra-line);
  background: #ffffff;
  color: var(--ra-ink);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.75);
}

.answer-md :is(h1, h2, h3, h4) {
  color: var(--ra-ink);
}

.answer-md code {
  border-color: var(--ra-line);
  background: #eef5f4;
}

.evidence-title {
  margin: 18px 0 10px;
  color: var(--ra-ink);
  font-size: 18px;
  line-height: 1.2;
  font-weight: 760;
}

.evidence-item {
  overflow: hidden;
  border-color: var(--ra-line);
  background: #ffffff;
  box-shadow: var(--ra-shadow-soft);
}

.evidence-summary {
  display: grid;
  grid-template-columns: auto auto minmax(0, 1fr) auto;
  gap: 10px;
  align-items: center;
  padding: 12px 14px;
  color: var(--ra-ink);
}

.evidence-summary:hover {
  background: #f4f8f7;
}

.evidence-summary::before {
  content: ">";
  width: 18px;
  height: 18px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--ra-line);
  border-radius: 999px;
  color: var(--ra-accent);
  font-size: 11px;
  font-weight: 800;
}

.evidence-index {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 999px;
  background: #e7f4f2;
  color: var(--ra-accent);
  font-size: 12px;
  font-weight: 780;
}

.evidence-source {
  min-width: 0;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.evidence-source-label {
  color: var(--ra-muted);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}

.chip {
  border-color: var(--ra-line);
  background: #f4f8f7;
  color: #20302e;
}

.source-chip {
  min-width: 0;
  flex: 1 1 220px;
}

.score-chip {
  background: #edf4ff;
  color: #1d4ed8;
}

.history-card {
  margin-top: 14px;
  border: 1px solid var(--ra-line);
  border-radius: var(--ra-radius);
  background:
    linear-gradient(180deg, rgba(248, 252, 251, 0.96), rgba(255, 255, 255, 0.98));
  padding: 12px 12px 10px;
  box-shadow: var(--ra-shadow-soft);
}

.history-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
}

.history-title {
  color: var(--ra-ink);
  font-size: 15px;
  font-weight: 760;
}

.history-refresh button {
  min-width: 58px !important;
  min-height: 30px !important;
  padding: 4px 12px !important;
  border-radius: 999px !important;
  border-color: var(--ra-line) !important;
  background: #f7fbfa !important;
  color: var(--ra-accent) !important;
  font-size: 12px !important;
  box-shadow: none !important;
}

.history-refresh button:hover {
  background: #e7f4f2 !important;
  border-color: rgba(15, 118, 110, 0.36) !important;
}

.history-list {
  margin-top: 2px;
}

.history-list .wrap {
  max-height: 220px;
  overflow-y: auto;
  overflow-x: hidden;
  display: flex;
  flex-direction: column;
  flex-wrap: nowrap;
  gap: 6px;
  padding-right: 2px;
}

.history-list .wrap > div {
  width: 100% !important;
  display: flex !important;
  flex-direction: column !important;
  flex-wrap: nowrap !important;
  gap: 6px !important;
}

.history-list .wrap::-webkit-scrollbar {
  width: 6px;
}

.history-list .wrap::-webkit-scrollbar-thumb {
  background: #c9d7d3;
  border-radius: 999px;
}

.history-list label {
  width: 100% !important;
  max-width: 100% !important;
  margin: 0 !important;
  padding: 8px 10px !important;
  border: 1px solid var(--ra-line) !important;
  border-radius: var(--ra-radius-sm) !important;
  background: #ffffff !important;
  color: var(--ra-ink) !important;
  box-shadow: none !important;
  transition: background 0.12s ease, border-color 0.12s ease, transform 0.12s ease;
}

.history-list label:hover {
  transform: translateY(-1px);
  border-color: rgba(15, 118, 110, 0.30) !important;
  background: #f4f8f7 !important;
}

.history-list label:has(input[type="radio"]:checked) {
  border-color: var(--ra-accent) !important;
  background: #e7f4f2 !important;
  box-shadow: inset 3px 0 0 var(--ra-accent) !important;
}

.history-list input[type="radio"] {
  margin-top: 2px !important;
}

.history-list span,
.history-list p {
  min-width: 0;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  line-height: 1.35;
  font-size: 13px;
}

.history-note {
  margin-top: 9px;
  color: var(--ra-muted);
  font-size: 12px;
  line-height: 1.45;
}

.evidence-content {
  padding: 14px;
  background: #fbfdfc;
}

.snippet-container {
  border-color: var(--ra-line);
  background: #f7fbfa;
}

.snippet-text {
  color: var(--ra-ink);
}

.snippet-code {
  max-width: 100%;
  border-color: var(--ra-line);
  background: #f4f8f7;
  color: #17201f;
}

.snippet-code code {
  background: transparent !important;
  border: 0 !important;
  color: #17201f !important;
  text-shadow: none !important;
}

.snippet-table {
  background: #ffffff;
}

.snippet-table thead th {
  background: #edf5f3;
}

@media (max-width: 900px) {
  .gradio-container {
    padding: 12px !important;
  }

  .app-header,
  .panel-head {
    flex-direction: column;
    align-items: flex-start;
  }

  .app-status {
    justify-content: flex-start;
  }

  .workspace-layout {
    flex-direction: column !important;
  }

  .workspace-layout > div {
    min-width: 0 !important;
    width: 100% !important;
  }

  .workspace-panel {
    padding: 13px;
  }

  .evidence-summary {
    grid-template-columns: auto auto minmax(0, 1fr);
  }

  .score-chip {
    grid-column: 3;
    width: fit-content;
  }
}
"""


def _status_badge(status: str) -> str:
    raw = (status or "unknown").strip().lower()
    klass = "status-badge"
    if raw == "answered":
        klass += " status-answered"
    elif raw == "not_found":
        klass += " status-not-found"
    elif raw == "error":
        klass += " status-error"
    value = escape(raw)
    return f'<span class="{klass}">{value}</span>'


def _warning_html(warning: str) -> str:
    if not warning:
        return ""
    return f'<div class="warning">{escape(warning)}</div>'


_REWRITE_MODE_CONSERVATIVE_LABEL = "\u4fdd\u5b88\u578b"
_REWRITE_MODE_AGGRESSIVE_LABEL = "\u6fc0\u8fdb\u578b"
_REWRITE_MODE_LABEL_TO_KEY = {
    _REWRITE_MODE_CONSERVATIVE_LABEL: "conservative",
    _REWRITE_MODE_AGGRESSIVE_LABEL: "aggressive",
}
_REWRITE_MODE_KEY_TO_LABEL = {value: key for key, value in _REWRITE_MODE_LABEL_TO_KEY.items()}
_FINAL_SOURCE_ORIGINAL_LABEL = "\u539f\u59cb query"
_FINAL_SOURCE_REWRITE_LABEL = "rewrite \u7ed3\u679c"
_HISTORY_LIMIT = 50
_USER_ID_JS = """
() => {
  const key = "ragagent_user_id";
  let userId = window.localStorage.getItem(key);
  if (!userId) {
    if (window.crypto && window.crypto.randomUUID) {
      userId = `browser-${window.crypto.randomUUID()}`;
    } else {
      userId = `browser-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
    }
    window.localStorage.setItem(key, userId);
  }
  return userId;
}
"""


def _rewrite_mode_key(label: str) -> str:
    return _REWRITE_MODE_LABEL_TO_KEY.get((label or "").strip(), "conservative")


def _rewrite_status_html(strategy: str = "", warning: str = "", mode: str = "") -> str:
    if not strategy and not warning:
        return '<div class="card">点击 Rewrite 生成检索增强式候选 query，不会覆盖原始输入。</div>'

    mode_label = _REWRITE_MODE_KEY_TO_LABEL.get(mode, "")
    strategy_map = {
        "pass_through_precise": "原 query 已足够精确，直接保留原文。",
        "llm_rewrite": "已生成检索增强式 rewrite 候选，可继续手动编辑。",
        "fallback_original": "rewrite 未生成可用候选，已回退为原 query。",
        "legacy_rewrite": "历史记录来自旧版 rewrite，原始 rewrite 元数据不可恢复。",
    }
    if strategy == "llm_rewrite" and mode_label:
        primary = f"已生成{mode_label} rewrite 候选，可继续手动编辑。"
    else:
        primary = strategy_map.get(strategy, strategy or "rewrite 已处理。")
    html = f'<div class="card">{escape(primary)}</div>'
    if warning:
        html += f'<div class="warning">{escape(warning)}</div>'
    return html


def _normalize_newlines(text: str) -> str:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in raw.split("\n"))


def _is_list_line(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*•]|\d+[.)]|[一二三四五六七八九十]+[、.])\s+", line.strip()))


_PARAM_HEADER_RE = re.compile(r"^(参数|序号|说明|类型|默认值|返回值|用途|示例|备注|字段|名称|单位|可选值|取值|含义)$")
_PARAM_TYPE_WORD_RE = re.compile(
    r"^(?:str|int|float|bool|dict|list|tuple|set|none|true|false|null|any|waveform|torchvariable)$",
    re.IGNORECASE,
)
_NUMBERED_PARAM_ROW_RE = re.compile(r"^\d+\s+\S+")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")
_FENCED_SNIPPET_RE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\n[\s\S]*?```")


def _looks_like_param_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _PARAM_HEADER_RE.fullmatch(stripped):
        return True
    if _NUMBERED_PARAM_ROW_RE.match(stripped):
        return True
    if _PARAM_TYPE_WORD_RE.fullmatch(stripped):
        return True
    return False


def _is_markdown_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.count("|") >= 2


def _is_markdown_table_separator(line: str) -> bool:
    return bool(_TABLE_SEPARATOR_RE.fullmatch(line.strip()))


def _contains_markdown_table(lines: List[str]) -> bool:
    for idx in range(len(lines) - 1):
        if _is_markdown_table_line(lines[idx]) and _is_markdown_table_separator(lines[idx + 1]):
            return True
    return False


def _is_parameter_style_paragraph(lines: List[str]) -> bool:
    stripped_lines = [line.strip() for line in lines if line.strip()]
    if not stripped_lines:
        return False
    header_hits = sum(1 for line in stripped_lines if _PARAM_HEADER_RE.fullmatch(line))
    numbered_hits = sum(1 for line in stripped_lines if _NUMBERED_PARAM_ROW_RE.match(line))
    type_hits = sum(1 for line in stripped_lines if _PARAM_TYPE_WORD_RE.fullmatch(line))

    if numbered_hits >= 2:
        return True
    if header_hits >= 3 and len(stripped_lines) >= 5:
        return True
    if header_hits >= 2 and (numbered_hits + type_hits) >= 1:
        return True
    return False


def _line_code_score(line: str) -> int:
    raw = line.rstrip("\n")
    stripped = raw.strip()
    if not stripped:
        return 0
    if stripped.startswith("Section:"):
        return 0
    if _is_list_line(stripped):
        return 0
    if _looks_like_param_line(stripped):
        return 0
    if _is_markdown_table_line(stripped):
        return 0
    if len(stripped) <= 24 and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", stripped):
        return 0

    score = 0
    lowered = stripped.lower()
    if raw.startswith(("    ", "\t")):
        score += 1
    if stripped.startswith(("```", "~~~", "$ ", ">>>")):
        score += 3
    if re.match(r"^(def|class|import|from|return|if|elif|else:|for|while|try:|except|finally:|with|@)\b", lowered):
        score += 3
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*.+$", stripped):
        score += 2
    if re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(", stripped) and any(ch in stripped for ch in "=):"):
        score += 2
    if re.search(r"[{}[\];]|=>|->|::", stripped):
        score += 1
    if re.search(r"\b(python|pip|git|curl|pytest|setenv)\b", lowered) and " " in stripped:
        score += 2
    if stripped.startswith(("#", "//")):
        score += 1
    return score


def _is_code_like_line(line: str) -> bool:
    return _line_code_score(line) >= 2


def _split_paragraphs(text: str) -> List[List[str]]:
    chunks = [p for p in re.split(r"\n\s*\n+", text) if p.strip()]
    paragraphs: List[List[str]] = []
    for chunk in chunks:
        lines = [line for line in chunk.split("\n") if line.strip()]
        if lines:
            paragraphs.append(lines)
    return paragraphs


def _classify_paragraph(lines: List[str]) -> str:
    if _is_parameter_style_paragraph(lines):
        return "text"
    scores = [_line_code_score(line) for line in lines]
    code_flags = [score >= 2 for score in scores]
    strong_flags = [score >= 3 for score in scores]
    code_count = sum(code_flags)
    if code_count == 0:
        return "text"
    max_streak = 0
    streak = 0
    for flag in code_flags:
        if flag:
            streak += 1
            if streak > max_streak:
                max_streak = streak
            continue
        streak = 0
    if max_streak >= 2 and code_count >= 2:
        return "code"
    if sum(strong_flags) >= 2 and code_count >= 2:
        return "code"
    if code_count >= 3 and (code_count / max(1, len(lines))) >= 0.60:
        return "code"
    return "text"


def _smart_join(prev_text: str, next_text: str) -> str:
    if not prev_text:
        return next_text
    if not next_text:
        return prev_text
    if re.search(r"[\u4e00-\u9fff]$", prev_text) and re.match(r"^[\u4e00-\u9fff]", next_text):
        return prev_text + next_text
    if prev_text.endswith(("(", "（", "[", "【", "/", "-", "_")):
        return prev_text + next_text
    if next_text.startswith((")", "）", "]", "】", ",", "，", ".", "。", ":", "：", ";", "；", "!", "！", "?", "？")):
        return prev_text + next_text
    return f"{prev_text} {next_text}"


def _is_heading_like(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _is_list_line(stripped):
        return True
    if stripped.endswith(("：", ":")):
        return True
    if re.fullmatch(r"(参数|说明|返回值|类型|默认值|用途|示例|备注)", stripped):
        return True
    return False


def _merge_text_lines(lines: List[str]) -> str:
    stripped_lines = [line.strip() for line in lines if line.strip()]
    if not stripped_lines:
        return ""

    paired: List[str] = []
    idx = 0
    while idx < len(stripped_lines):
        current = stripped_lines[idx]
        nxt = stripped_lines[idx + 1] if idx + 1 < len(stripped_lines) else ""
        if current.endswith(("：", ":")) and nxt and not _is_heading_like(nxt):
            paired.append(f"{current} {nxt}")
            idx += 2
            continue
        paired.append(current)
        idx += 1

    merged_lines: List[str] = []
    current_text = ""
    for line in paired:
        if _is_heading_like(line):
            if current_text:
                merged_lines.append(current_text)
                current_text = ""
            merged_lines.append(line)
            continue
        current_text = _smart_join(current_text, line) if current_text else line

    if current_text:
        merged_lines.append(current_text)
    return "\n".join(merged_lines).strip()


def _format_text_block(lines: List[str]) -> str:
    stripped_lines = [line.strip() for line in lines if line.strip()]
    if not stripped_lines:
        return ""
    if _contains_markdown_table(stripped_lines):
        return "\n".join(stripped_lines)
    if _is_parameter_style_paragraph(stripped_lines):
        return "\n".join(stripped_lines)
    return _merge_text_lines(stripped_lines)


def _split_text_and_table_blocks(text: str) -> List[Tuple[str, str]]:
    lines = text.split("\n")
    blocks: List[Tuple[str, str]] = []
    text_buf: List[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if idx + 1 < len(lines) and _is_markdown_table_line(line) and _is_markdown_table_separator(lines[idx + 1]):
            if text_buf:
                text_block = "\n".join(text_buf).strip()
                if text_block:
                    blocks.append(("text", text_block))
                text_buf = []
            table_lines = [line, lines[idx + 1]]
            idx += 2
            while idx < len(lines) and _is_markdown_table_line(lines[idx]):
                table_lines.append(lines[idx])
                idx += 1
            table_block = "\n".join(table_lines).strip()
            if table_block:
                blocks.append(("table", table_block))
            continue
        text_buf.append(line)
        idx += 1
    if text_buf:
        text_block = "\n".join(text_buf).strip()
        if text_block:
            blocks.append(("text", text_block))
    return blocks


def _parse_markdown_table_row(line: str) -> List[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [cell.strip() for cell in stripped.split("|")]
    return [cell.replace(r"\|", "|").replace("<br>", "\n") for cell in cells]


def _render_markdown_table_html(table_text: str) -> str:
    lines = [line for line in table_text.split("\n") if line.strip()]
    if len(lines) < 2:
        return f'<div class="snippet-text">{escape(table_text)}</div>'
    header = _parse_markdown_table_row(lines[0])
    data_rows = [_parse_markdown_table_row(line) for line in lines[2:]]
    max_cols = max([len(header)] + [len(row) for row in data_rows] + [1])
    header += [""] * (max_cols - len(header))
    normalized_rows = [row + [""] * (max_cols - len(row)) for row in data_rows]

    thead = "<thead><tr>" + "".join(f"<th>{escape(cell)}</th>" for cell in header) + "</tr></thead>"
    tbody_rows = []
    for row in normalized_rows:
        tbody_rows.append("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>")
    tbody = "<tbody>" + "".join(tbody_rows) + "</tbody>"
    return f'<div class="snippet-table-wrap"><table class="snippet-table">{thead}{tbody}</table></div>'


def _split_paragraph_with_line_correction(lines: List[str]) -> List[Tuple[str, str]]:
    if not lines:
        return []
    if _is_parameter_style_paragraph(lines):
        return [("text", _format_text_block(lines))]
    para_type = _classify_paragraph(lines)
    if para_type == "code":
        return [("code", "\n".join(lines).strip())]

    scores = [_line_code_score(line) for line in lines]
    mark_code = [False] * len(lines)
    run_start = -1
    for idx, score in enumerate(scores):
        if score >= 2:
            if run_start < 0:
                run_start = idx
        else:
            if run_start >= 0 and (idx - run_start) >= 2:
                for j in range(run_start, idx):
                    mark_code[j] = True
            run_start = -1
    if run_start >= 0 and (len(lines) - run_start) >= 2:
        for j in range(run_start, len(lines)):
            mark_code[j] = True

    for idx, score in enumerate(scores):
        if score >= 4:
            mark_code[idx] = True

    blocks: List[Tuple[str, str]] = []
    current_kind = "code" if mark_code[0] else "text"
    current_lines: List[str] = [lines[0]]
    for idx in range(1, len(lines)):
        kind = "code" if mark_code[idx] else "text"
        if kind == current_kind:
            current_lines.append(lines[idx])
            continue
        block_text = "\n".join(current_lines).strip()
        if block_text:
            if current_kind == "text":
                block_text = _format_text_block(current_lines)
            blocks.append((current_kind, block_text))
        current_kind = kind
        current_lines = [lines[idx]]

    final_text = "\n".join(current_lines).strip()
    if final_text:
        if current_kind == "text":
            final_text = _format_text_block(current_lines)
        blocks.append((current_kind, final_text))
    return blocks


def _split_unfenced_snippet_blocks(snippet: str) -> List[Tuple[str, str]]:
    raw = _normalize_newlines(snippet).strip()
    if not raw:
        return []
    paragraphs = _split_paragraphs(raw)
    blocks: List[Tuple[str, str]] = []
    for paragraph in paragraphs:
        blocks.extend(_split_paragraph_with_line_correction(paragraph))
    merged: List[Tuple[str, str]] = []
    for kind, text in blocks:
        if not text.strip():
            continue
        if merged and merged[-1][0] == kind:
            merged[-1] = (kind, f"{merged[-1][1]}\n\n{text}")
        else:
            merged.append((kind, text))
    return merged


def _split_snippet_blocks(snippet: str) -> List[Tuple[str, str]]:
    raw = _normalize_newlines(snippet).strip()
    if not raw:
        return []

    blocks: List[Tuple[str, str]] = []
    last = 0
    for match in _FENCED_SNIPPET_RE.finditer(raw):
        prefix = raw[last:match.start()].strip()
        if prefix:
            blocks.extend(_split_unfenced_snippet_blocks(prefix))

        fenced = match.group(0).strip("\n")
        lines = fenced.split("\n")
        if len(lines) >= 2:
            code_lines = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
            code = "\n".join(code_lines).strip("\n")
            if code:
                blocks.append(("code", code))
        last = match.end()

    tail = raw[last:].strip()
    if tail:
        blocks.extend(_split_unfenced_snippet_blocks(tail))

    merged: List[Tuple[str, str]] = []
    for kind, text in blocks:
        if not text.strip():
            continue
        if merged and merged[-1][0] == kind:
            separator = "\n\n" if kind == "text" else "\n"
            merged[-1] = (kind, f"{merged[-1][1]}{separator}{text}")
        else:
            merged.append((kind, text))
    return merged


_REFLOW_KEYWORD_RE = re.compile(r"\s+(?=(?:def|class|from|import|with|if|for|while|try|except|finally|return)\b)")


def _should_reflow_compact_code(text: str) -> bool:
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return False
    if len(lines) > 6:
        return False
    if _is_parameter_style_paragraph(lines):
        return False
    longest = max(len(line) for line in lines)
    total_len = sum(len(line) for line in lines)
    if longest < 150 and total_len < 320:
        return False
    if longest < 180 and len(lines) > 2:
        return False
    code_marker = re.search(r"[{}();=]|@[A-Za-z_]|->|::", text)
    keyword_marker = re.search(r"\b(def|class|import|from|with|if|for|while|try|except|return)\b", text)
    assign_marker = re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*=", text)
    return bool(code_marker or keyword_marker or assign_marker)


def _gentle_reflow_line(line: str) -> str:
    work = line
    work = re.sub(r";\s*", ";\n", work)
    work = re.sub(r"\s+#\s*", "\n# ", work)
    work = re.sub(r"\s+(?=@[A-Za-z_])", "\n", work)
    work = re.sub(r"\s+(?=[A-Za-z_][A-Za-z0-9_]*\s*=)", "\n", work)
    work = _REFLOW_KEYWORD_RE.sub("\n", work)
    work = re.sub(r"\n{3,}", "\n\n", work)
    return work


def _gentle_reflow_code_block(text: str) -> str:
    if not _should_reflow_compact_code(text):
        return text
    fixed_lines: List[str] = []
    for line in text.split("\n"):
        if len(line) >= 180:
            fixed_lines.append(_gentle_reflow_line(line))
        else:
            fixed_lines.append(line)
    result = "\n".join(fixed_lines)
    return re.sub(r"\n{3,}", "\n\n", result).strip("\n")


def _render_evidence(evidence: List[EvidenceItem]) -> str:
    if not evidence:
        return '<div class="card empty-card">No evidence returned.</div>'

    html_items: List[str] = []
    for idx, item in enumerate(evidence, 1):
        source = escape(item.source)
        score = f"{item.score:.6f}"
        details_open_attr = " open" if idx == 1 else ""
        snippet_blocks = _split_snippet_blocks(item.snippet)
        snippet_html_parts: List[str] = []
        for block_type, block_text in snippet_blocks:
            if block_type == "code":
                safe = escape(block_text)
                snippet_html_parts.append(f'<pre class="snippet-code"><code>{safe}</code></pre>')
            else:
                sub_blocks = _split_text_and_table_blocks(block_text)
                if not sub_blocks:
                    safe = escape(block_text)
                    snippet_html_parts.append(f'<div class="snippet-text">{safe}</div>')
                for sub_type, sub_text in sub_blocks:
                    if sub_type == "table":
                        snippet_html_parts.append(_render_markdown_table_html(sub_text))
                    else:
                        safe = escape(sub_text)
                        snippet_html_parts.append(f'<div class="snippet-text">{safe}</div>')
        snippet_html = (
            '<div class="snippet-container">'
            + '<div class="snippet-sep"></div>'.join(snippet_html_parts or ['<div class="snippet-text">_(empty)_</div>'])
            + "</div>"
        )
        html_items.append(
            (
                f'<details class="evidence-item"{details_open_attr}>'
                '<summary class="evidence-summary">'
                f'<span class="evidence-index">{idx}</span>'
                '<span class="evidence-source">'
                '<span class="evidence-source-label">source</span>'
                f'<span class="chip source-chip">{source}</span>'
                '</span>'
                f'<span class="chip score-chip">score {score}</span>'
                '</summary>'
                f'<div class="evidence-content">{snippet_html}</div>'
                '</details>'
            )
        )
    return "".join(html_items)


def build_ragagent_ui(qa_agent: RagQaAgent, query_rewriter: QueryRewriter):
    import gradio as gr

    feedback_store = QaFeedbackStore(settings.rag_qa_feedback_db)

    def _serialize_evidence(evidence: List[EvidenceItem]) -> str:
        rows = []
        for item in evidence:
            if hasattr(item, "model_dump"):
                rows.append(item.model_dump())
            else:
                rows.append(
                    {
                        "source": item.source,
                        "score": item.score,
                        "snippet": item.snippet,
                    }
                )
        return json.dumps(rows, ensure_ascii=False)

    def _persist_qa_log(
        user_id: str,
        question: str,
        original_question: str,
        rewrite_query: str,
        final_query_source: str,
        rewrite_mode_label: str,
        rewrite_strategy: str,
        rewrite_warning: str,
        answer: str,
        status: str,
        warning: str,
        evidence: List[EvidenceItem],
        source: str,
    ) -> tuple[int | None, str]:
        try:
            safe_final_query_source = (final_query_source or "").strip()
            if safe_final_query_source not in {_FINAL_SOURCE_ORIGINAL_LABEL, _FINAL_SOURCE_REWRITE_LABEL}:
                safe_final_query_source = (
                    _FINAL_SOURCE_REWRITE_LABEL if source == "ragagent_ui_rewrite" else _FINAL_SOURCE_ORIGINAL_LABEL
                )
            safe_rewrite_query = (rewrite_query or "").strip()
            if (
                not safe_rewrite_query
                and safe_final_query_source == _FINAL_SOURCE_REWRITE_LABEL
                and source == "ragagent_ui_rewrite"
                and (question or "").strip() != (original_question or "").strip()
            ):
                safe_rewrite_query = (question or "").strip()
            record_id = feedback_store.insert_qa_log(
                user_id=user_id,
                question=question,
                original_question=original_question,
                rewrite_query=safe_rewrite_query,
                final_query_source=safe_final_query_source,
                rewrite_mode=_rewrite_mode_key(rewrite_mode_label),
                rewrite_strategy=rewrite_strategy,
                rewrite_warning=rewrite_warning,
                answer=answer,
                status=status,
                warning=warning,
                evidence_json=_serialize_evidence(evidence),
                source=source,
            )
            return record_id, ""
        except Exception as exc:  # noqa: BLE001
            return None, f"qa feedback store unavailable: {exc}"

    def _feedback_hint_html(record_id: int | None) -> str:
        if record_id is None:
            return '<div class="warning">问答已返回，但评价记录写入失败，请检查 SQLite 权限或路径。</div>'
        return '<div class="card">请点击“有用”或“无用”提交评价（可改选，最后一次覆盖之前评价）。</div>'

    def _history_label(row: dict[str, Any]) -> str:
        created_at = str(row.get("created_at") or "")
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00")).astimezone()
            created = dt.strftime("%m-%d %H:%M")
        except Exception:
            created = created_at[:16] or "unknown time"
        status = str(row.get("status") or "unknown")
        question_text = str(row.get("original_question") or row.get("question") or "").replace("\n", " ").strip()
        if len(question_text) > 40:
            question_text = question_text[:40].rstrip() + "..."
        return f"{status} · {created} · {question_text or '(empty)'}"

    def _history_choices(user_id: str) -> list[tuple[str, str]]:
        rows = feedback_store.list_history(user_id, limit=_HISTORY_LIMIT)
        return [(_history_label(row), str(row["id"])) for row in rows]

    def _refresh_history(user_id: str, selected_record_id: int | str | None = None):
        try:
            choices = _history_choices(user_id)
            values = {value for _, value in choices}
            selected = str(selected_record_id) if selected_record_id is not None else None
            value = selected if selected in values else None
            note = f"最近 {len(choices)} 条，仅当前浏览器。"
            return gr.update(choices=choices, value=value), f'<div class="history-note">{escape(note)}</div>'
        except Exception as exc:  # noqa: BLE001
            return gr.update(choices=[], value=None), f'<div class="warning">历史记录加载失败：{escape(str(exc))}</div>'

    def _initial_history(user_id: str):
        safe_user_id = QaFeedbackStore._normalize_user_id(user_id)
        history_update, history_note = _refresh_history(safe_user_id)
        return safe_user_id, history_update, history_note

    def _parse_evidence_json(raw: str) -> List[EvidenceItem]:
        try:
            payload = json.loads(raw or "[]")
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        evidence: List[EvidenceItem] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                evidence.append(EvidenceItem(**item))
            except Exception:
                continue
        return evidence

    def _load_history_record(user_id: str, record_id: str | int | None):
        if not record_id:
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(visible=False),
                gr.update(),
                gr.update(),
                gr.update(),
                None,
                '<div class="card">请选择一条历史对话。</div>',
            )
        try:
            row = feedback_store.get_history_record(user_id, int(record_id))
            if row is None:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(value='<div class="warning">历史记录不存在或不属于当前浏览器。</div>', visible=True),
                    _status_badge("error"),
                    "_(empty)_",
                    _render_evidence([]),
                    None,
                    '<div class="warning">历史记录读取失败。</div>',
                )
            evidence = _parse_evidence_json(str(row.get("evidence_json") or "[]"))
            rewrite_mode_key = str(row.get("rewrite_mode") or "")
            rewrite_mode_label = _REWRITE_MODE_KEY_TO_LABEL.get(rewrite_mode_key, _REWRITE_MODE_AGGRESSIVE_LABEL)
            final_source = str(row.get("final_query_source") or "")
            if final_source not in {_FINAL_SOURCE_ORIGINAL_LABEL, _FINAL_SOURCE_REWRITE_LABEL}:
                final_source = _FINAL_SOURCE_REWRITE_LABEL if row.get("rewrite_query") else _FINAL_SOURCE_ORIGINAL_LABEL
            warning = str(row.get("warning") or "")
            answer = str(row.get("answer") or "")
            return (
                row.get("original_question") or row.get("question") or "",
                row.get("rewrite_query") or "",
                gr.update(value=final_source),
                _rewrite_status_html(
                    strategy=str(row.get("rewrite_strategy") or ""),
                    warning=str(row.get("rewrite_warning") or ""),
                    mode=rewrite_mode_key,
                ),
                str(row.get("rewrite_strategy") or ""),
                str(row.get("rewrite_warning") or ""),
                row.get("original_question") or row.get("question") or "",
                gr.update(value=rewrite_mode_label),
                gr.update(value=_warning_html(warning), visible=bool(warning)),
                _status_badge(str(row.get("status") or "unknown")),
                answer if answer.strip() else "_(empty)_",
                _render_evidence(evidence),
                int(row["id"]),
                _feedback_hint_html(int(row["id"])),
            )
        except Exception as exc:  # noqa: BLE001
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(value=f'<div class="warning">历史记录读取失败：{escape(str(exc))}</div>', visible=True),
                _status_badge("error"),
                "_(empty)_",
                _render_evidence([]),
                None,
                '<div class="warning">历史记录读取失败。</div>',
            )

    def _submit_feedback(record_id: int | None, user_id: str, feedback: str) -> str:
        if record_id is None:
            return '<div class="warning">请先提问再评价。</div>'
        try:
            ok = feedback_store.update_feedback(int(record_id), feedback, user_id=user_id)
            if not ok:
                return f'<div class="warning">评价失败：记录不存在（id={record_id}）。</div>'
            label = "有用" if feedback == "useful" else "无用"
            return f'<div class="card">已记录：{label}（可改选）。</div>'
        except Exception as exc:  # noqa: BLE001
            return f'<div class="warning">评价失败：{escape(str(exc))}</div>'

    with gr.Blocks(css=_CSS, title="TED文档问答助手", theme=gr.themes.Soft()) as demo:

        gr.HTML(
            """
            <div class="app-header">
              <div>
                <div class="app-kicker">RagAgent EDA</div>
                <div class="app-title">TED文档问答助手</div>
              </div>
              <div class="app-status">
                <span class="runtime-pill"><span class="runtime-dot"></span>Ready</span>
                <span class="route-pill">/ragagent</span>
              </div>
            </div>
            """
        )
        with gr.Row(elem_classes=["workspace-layout"]):
            with gr.Column(scale=4, min_width=340, elem_classes=["workspace-panel", "query-console"]):
                gr.HTML(
                    """
                    <div class="panel-head">
                      <div>
                        <div class="panel-kicker">Query Console</div>
                        <div class="panel-title">检索输入</div>
                      </div>
                      <span class="panel-tag">Rewrite</span>
                    </div>
                    """
                )
                question = gr.Textbox(
                    label="Question",
                    placeholder="输入 TED / EDA 文档相关问题",
                    lines=5,
                    max_lines=9,
                )
                rewrite_mode = gr.Radio(
                    label="Rewrite 模式",
                    choices=[_REWRITE_MODE_CONSERVATIVE_LABEL, _REWRITE_MODE_AGGRESSIVE_LABEL],
                    value=_REWRITE_MODE_AGGRESSIVE_LABEL,
                )
                with gr.Row(elem_classes=["action-row"]):
                    rewrite_btn = gr.Button("Rewrite", elem_classes=["secondary-action"])
                    ask_btn = gr.Button("Ask", variant="primary", elem_classes=["primary-action"])
                rewrite_result = gr.Textbox(
                    label="Rewrite Result",
                    placeholder="Rewrite result will appear here",
                    lines=5,
                    max_lines=9,
                    interactive=True,
                )
                final_query_source = gr.Radio(
                    label="\u6700\u7ec8\u8f93\u5165\u6765\u6e90",
                    choices=[_FINAL_SOURCE_ORIGINAL_LABEL, _FINAL_SOURCE_REWRITE_LABEL],
                    value=_FINAL_SOURCE_ORIGINAL_LABEL,
                )
                rewrite_meta_html = gr.HTML(_rewrite_status_html())
                rewrite_base_query_state = gr.State(value="")
                rewrite_strategy_state = gr.State(value="")
                rewrite_warning_state = gr.State(value="")
                user_id_state = gr.Textbox(value="legacy", visible=False, elem_id="ragagent-user-id")
                with gr.Group(elem_classes=["history-card"]):
                    with gr.Row(elem_classes=["history-head"]):
                        gr.HTML('<div class="history-title">历史对话</div>')
                        refresh_history_btn = gr.Button("刷新", elem_classes=["history-refresh"])
                    history_select = gr.Radio(
                        label="",
                        choices=[],
                        value=None,
                        interactive=True,
                        show_label=False,
                        elem_classes=["history-list"],
                    )
                    history_status_html = gr.HTML('<div class="history-note">正在读取当前浏览器历史。</div>')

            with gr.Column(scale=7, min_width=460, elem_classes=["workspace-panel", "result-console"]):
                gr.HTML(
                    """
                    <div class="panel-head">
                      <div>
                        <div class="panel-kicker">Answer Console</div>
                        <div class="panel-title">回答与证据</div>
                      </div>
                      <span class="panel-tag">RAG</span>
                    </div>
                    """
                )
                warning_html = gr.HTML(visible=False)
                status_html = gr.HTML()
                answer_md = gr.Markdown(elem_classes=["answer-md"])
                record_id_state = gr.State(value=None)
                with gr.Row(elem_classes=["feedback-row"]):
                    useful_btn = gr.Button("有用", elem_classes=["secondary-action"])
                    useless_btn = gr.Button("无用", elem_classes=["secondary-action"])
                feedback_html = gr.HTML('<div class="card">请先提问，再点击“有用/无用”评价。</div>')
                gr.HTML('<div class="evidence-title">Evidence</div>')
                evidence_html = gr.HTML()

        def _rewrite(input_question: str, selected_mode: str):
            query = (input_question or "").strip()
            mode = _rewrite_mode_key(selected_mode)
            if not query:
                warning = "question is empty"
                return "", _rewrite_status_html(warning=warning, mode=mode), "", gr.update(value=_FINAL_SOURCE_ORIGINAL_LABEL), "", warning
            try:
                result: QueryRewriteResult = query_rewriter.rewrite(query, scene="qa", mode=mode)
                return (
                    result.rewritten_query,
                    _rewrite_status_html(strategy=result.strategy, warning=result.warning, mode=mode),
                    result.original_query,
                    gr.update(value=_FINAL_SOURCE_REWRITE_LABEL),
                    result.strategy,
                    result.warning,
                )
            except Exception as exc:  # noqa: BLE001
                warning = str(exc)
                return "", _rewrite_status_html(warning=warning, mode=mode), "", gr.update(value=_FINAL_SOURCE_ORIGINAL_LABEL), "", warning

        def _clear_rewrite_state(_: str):
            return "", gr.update(value=_FINAL_SOURCE_ORIGINAL_LABEL), _rewrite_status_html(), "", "", ""

        def _ask(
            input_question: str,
            rewrite_query: str,
            selected_source: str,
            rewrite_base_query: str,
            selected_rewrite_mode: str,
            rewrite_strategy: str,
            rewrite_warning: str,
            user_id: str,
        ):
            raw_query = (input_question or "").strip()
            rewrite_candidate = (rewrite_query or "").strip()
            source_label = (selected_source or _FINAL_SOURCE_ORIGINAL_LABEL).strip()

            def ask_response(_record_id: int | None, values: tuple):
                return values

            if source_label == _FINAL_SOURCE_REWRITE_LABEL:
                if not rewrite_candidate:
                    warning_text = "rewrite result is empty"
                    record_id, store_warn = _persist_qa_log(
                        user_id=user_id,
                        question=raw_query,
                        original_question=raw_query,
                        rewrite_query=rewrite_candidate,
                        final_query_source=source_label,
                        rewrite_mode_label=selected_rewrite_mode,
                        rewrite_strategy=rewrite_strategy,
                        rewrite_warning=rewrite_warning,
                        answer="",
                        status="error",
                        warning=warning_text,
                        evidence=[],
                        source="ragagent_ui_rewrite",
                    )
                    merged_warning = warning_text
                    if store_warn:
                        merged_warning = f"{merged_warning}; {store_warn}"
                    return ask_response(
                        record_id,
                        (
                            gr.update(value=f'<div class="warning">{escape(merged_warning)}</div>', visible=True),
                            _status_badge("error"),
                            "_(empty)_",
                            _render_evidence([]),
                            record_id,
                            _feedback_hint_html(record_id),
                        ),
                    )
                raw_query_key = query_rewriter.normalize_query(raw_query)
                rewrite_base_query_key = query_rewriter.normalize_query(rewrite_base_query or "")
                if rewrite_base_query_key != raw_query_key:
                    warning_text = "original query changed after rewrite; please rewrite again or use original query"
                    record_id, store_warn = _persist_qa_log(
                        user_id=user_id,
                        question=raw_query,
                        original_question=raw_query,
                        rewrite_query=rewrite_candidate,
                        final_query_source=source_label,
                        rewrite_mode_label=selected_rewrite_mode,
                        rewrite_strategy=rewrite_strategy,
                        rewrite_warning=rewrite_warning,
                        answer="",
                        status="error",
                        warning=warning_text,
                        evidence=[],
                        source="ragagent_ui_rewrite",
                    )
                    merged_warning = warning_text
                    if store_warn:
                        merged_warning = f"{merged_warning}; {store_warn}"
                    return ask_response(
                        record_id,
                        (
                            gr.update(value=f'<div class="warning">{escape(merged_warning)}</div>', visible=True),
                            _status_badge("error"),
                            "_(empty)_",
                            _render_evidence([]),
                            record_id,
                            _feedback_hint_html(record_id),
                        ),
                    )
                query = rewrite_candidate
                qa_source = "ragagent_ui_rewrite"
            else:
                query = raw_query
                qa_source = "ragagent_ui_original"

            if not query:
                warning_text = "question is empty"
                record_id, store_warn = _persist_qa_log(
                    user_id=user_id,
                    question=query,
                    original_question=raw_query,
                    rewrite_query=rewrite_candidate,
                    final_query_source=source_label,
                    rewrite_mode_label=selected_rewrite_mode,
                    rewrite_strategy=rewrite_strategy,
                    rewrite_warning=rewrite_warning,
                    answer="",
                    status="error",
                    warning=warning_text,
                    evidence=[],
                    source=qa_source,
                )
                merged_warning = warning_text
                if store_warn:
                    merged_warning = f"{merged_warning}; {store_warn}"
                return ask_response(
                    record_id,
                    (
                        gr.update(value=f'<div class="warning">{escape(merged_warning)}</div>', visible=True),
                        _status_badge("error"),
                        "_(empty)_",
                        _render_evidence([]),
                        record_id,
                        _feedback_hint_html(record_id),
                    ),
                )
            try:
                result: RagAskResponse = qa_agent.ask(query)
                record_id, store_warn = _persist_qa_log(
                    user_id=user_id,
                    question=query,
                    original_question=raw_query,
                    rewrite_query=rewrite_candidate,
                    final_query_source=source_label,
                    rewrite_mode_label=selected_rewrite_mode,
                    rewrite_strategy=rewrite_strategy,
                    rewrite_warning=rewrite_warning,
                    answer=result.answer if (result.answer or "").strip() else "",
                    status=result.status,
                    warning=result.warning or "",
                    evidence=result.evidence,
                    source=qa_source,
                )
                merged_warning = result.warning or ""
                if store_warn:
                    merged_warning = f"{merged_warning}; {store_warn}" if merged_warning else store_warn
                return ask_response(
                    record_id,
                    (
                        gr.update(value=_warning_html(merged_warning), visible=bool(merged_warning)),
                        _status_badge(result.status),
                        result.answer if (result.answer or "").strip() else "_(empty)_",
                        _render_evidence(result.evidence),
                        record_id,
                        _feedback_hint_html(record_id),
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                error_text = str(exc)
                record_id, store_warn = _persist_qa_log(
                    user_id=user_id,
                    question=query,
                    original_question=raw_query,
                    rewrite_query=rewrite_candidate,
                    final_query_source=source_label,
                    rewrite_mode_label=selected_rewrite_mode,
                    rewrite_strategy=rewrite_strategy,
                    rewrite_warning=rewrite_warning,
                    answer="",
                    status="error",
                    warning=error_text,
                    evidence=[],
                    source=qa_source,
                )
                merged_warning = error_text
                if store_warn:
                    merged_warning = f"{merged_warning}; {store_warn}"
                return ask_response(
                    record_id,
                    (
                        gr.update(value=f'<div class="warning">{escape(merged_warning)}</div>', visible=True),
                        _status_badge("error"),
                        "_(empty)_",
                        _render_evidence([]),
                        record_id,
                        _feedback_hint_html(record_id),
                    ),
                )

        rewrite_btn.click(
            _rewrite,
            inputs=[question, rewrite_mode],
            outputs=[
                rewrite_result,
                rewrite_meta_html,
                rewrite_base_query_state,
                final_query_source,
                rewrite_strategy_state,
                rewrite_warning_state,
            ],
        )
        question.input(
            _clear_rewrite_state,
            inputs=[question],
            outputs=[
                rewrite_result,
                final_query_source,
                rewrite_meta_html,
                rewrite_base_query_state,
                rewrite_strategy_state,
                rewrite_warning_state,
            ],
        )
        rewrite_mode.change(
            _clear_rewrite_state,
            inputs=[rewrite_mode],
            outputs=[
                rewrite_result,
                final_query_source,
                rewrite_meta_html,
                rewrite_base_query_state,
                rewrite_strategy_state,
                rewrite_warning_state,
            ],
        )
        ask_event = ask_btn.click(
            _ask,
            inputs=[
                question,
                rewrite_result,
                final_query_source,
                rewrite_base_query_state,
                rewrite_mode,
                rewrite_strategy_state,
                rewrite_warning_state,
                user_id_state,
            ],
            outputs=[
                warning_html,
                status_html,
                answer_md,
                evidence_html,
                record_id_state,
                feedback_html,
            ],
        )
        ask_event.then(
            _refresh_history,
            inputs=[user_id_state],
            outputs=[history_select, history_status_html],
            show_progress="hidden",
        )
        submit_event = question.submit(
            _ask,
            inputs=[
                question,
                rewrite_result,
                final_query_source,
                rewrite_base_query_state,
                rewrite_mode,
                rewrite_strategy_state,
                rewrite_warning_state,
                user_id_state,
            ],
            outputs=[
                warning_html,
                status_html,
                answer_md,
                evidence_html,
                record_id_state,
                feedback_html,
            ],
        )
        submit_event.then(
            _refresh_history,
            inputs=[user_id_state],
            outputs=[history_select, history_status_html],
            show_progress="hidden",
        )
        demo.load(
            _initial_history,
            inputs=[user_id_state],
            outputs=[user_id_state, history_select, history_status_html],
            js=_USER_ID_JS,
            show_progress="hidden",
        )
        refresh_history_btn.click(
            _refresh_history,
            inputs=[user_id_state, history_select],
            outputs=[history_select, history_status_html],
            show_progress="hidden",
        )
        history_select.change(
            _load_history_record,
            inputs=[user_id_state, history_select],
            outputs=[
                question,
                rewrite_result,
                final_query_source,
                rewrite_meta_html,
                rewrite_strategy_state,
                rewrite_warning_state,
                rewrite_base_query_state,
                rewrite_mode,
                warning_html,
                status_html,
                answer_md,
                evidence_html,
                record_id_state,
                feedback_html,
            ],
        )
        useful_btn.click(
            lambda rid, uid: _submit_feedback(rid, uid, "useful"),
            inputs=[record_id_state, user_id_state],
            outputs=[feedback_html],
        )
        useless_btn.click(
            lambda rid, uid: _submit_feedback(rid, uid, "useless"),
            inputs=[record_id_state, user_id_state],
            outputs=[feedback_html],
        )
    return demo


def mount_ragagent_ui(
    app: FastAPI,
    qa_agent: RagQaAgent,
    query_rewriter: QueryRewriter,
    path: str = "/ragagent",
) -> FastAPI:
    import gradio as gr

    demo = build_ragagent_ui(qa_agent, query_rewriter)
    return gr.mount_gradio_app(app, demo, path=path)
