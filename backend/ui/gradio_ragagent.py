from __future__ import annotations

from datetime import datetime
import json
from html import escape
from pathlib import Path
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

_CSS = Path(__file__).with_name("workbench.css").read_text(encoding="utf-8")

_EMPTY_ANSWER = """
<div class="answer-welcome">
  <div class="welcome-symbol" aria-hidden="true"><span></span><span></span><span></span></div>
  <div class="welcome-eyebrow">FROM DOCUMENTS TO ANSWERS</div>
  <h3>让每个答案，都有据可循。</h3>
  <p>从左侧输入一个问题，或选择示例开始。<br>在这里阅读回答，并展开原文核对细节。</p>
  <div class="welcome-steps"><span>01&nbsp; 提出问题</span><i>→</i><span>02&nbsp; 检索文档</span><i>→</i><span>03&nbsp; 查看依据</span></div>
</div>
"""
_SCROLL_TO_ANSWER_JS = """
() => {
  if (window.matchMedia("(max-width: 800px)").matches) {
    document.getElementById("answer-status")?.scrollIntoView({
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth",
      block: "start"
    });
  }
}
"""
_EXAMPLE_QUESTIONS = [
    "如何使用 TED 配置瞬态仿真？",
    "如何测量运放的带宽？",
    "TED 中如何设置电压源的参数？",
]



def _status_badge(status: str) -> str:
    raw = (status or "unknown").strip().lower()
    klass = "status-badge"
    if raw == "answered":
        klass += " status-answered"
    elif raw == "not_found":
        klass += " status-not-found"
    elif raw == "error":
        klass += " status-error"
    value = escape({"answered": "已生成回答", "not_found": "未找到相关依据", "error": "暂时无法回答"}.get(raw, raw))
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
        return '<div class="card">可选步骤：优化问题表达后再提问，也可以直接使用原始问题。</div>'

    mode_label = _REWRITE_MODE_KEY_TO_LABEL.get(mode, "")
    strategy_map = {
        "pass_through_precise": "原始问题已足够明确，已保留原文。",
        "llm_rewrite": "已生成改写建议，可以继续编辑后提问。",
        "fallback_original": "暂未生成可用建议，已保留原始问题。",
        "legacy_rewrite": "历史记录来自旧版 rewrite，原始 rewrite 元数据不可恢复。",
    }
    if strategy == "llm_rewrite" and mode_label:
        primary = f"已生成{mode_label}改写建议，可以继续编辑后提问。"
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
        return '<div class="card empty-card">暂无可展示的引用片段。可以补充函数名、参数或具体任务后提问。</div>'

    html_items: List[str] = []
    for idx, item in enumerate(evidence, 1):
        source = escape(item.source)
        score = f"{item.score:.3f}"
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
                '<span class="evidence-source-label">文档来源</span>'
                f'<span class="chip source-chip">{source}</span>'
                '</span>'
                f'<span class="chip score-chip">相关度 {score}</span>'
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
        return '<div class="feedback-note">这次回答有帮助吗？评价可以随时修改。</div>'

    def _history_label(row: dict[str, Any]) -> str:
        created_at = str(row.get("created_at") or "")
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00")).astimezone()
            created = dt.strftime("%m-%d %H:%M")
        except Exception:
            created = created_at[:16] or "unknown time"
        raw_status = str(row.get("status") or "unknown")
        status = {"answered": "已回答", "not_found": "无依据", "error": "未完成"}.get(raw_status, raw_status)
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
            note = f"最近 {len(choices)} 条 · 当前浏览器的记录" if choices else "还没有对话。第一次提问后，记录会保存在这里。"
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
                    "暂无回答内容。",
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
                answer if answer.strip() else "暂无回答内容。",
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
                "暂无回答内容。",
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

    theme = gr.themes.Base(
        primary_hue="emerald", neutral_hue="slate",
        font=["Aptos", "Microsoft YaHei UI", "sans-serif"],
        font_mono=["Cascadia Code", "Consolas", "monospace"],
    ).set(
        body_background_fill="#f4f5f1", body_background_fill_dark="#f4f5f1",
        body_text_color="#20332e", body_text_color_dark="#20332e",
        block_background_fill="#ffffff", block_background_fill_dark="#ffffff",
        block_border_color="#dfe5df", block_border_color_dark="#dfe5df",
        input_background_fill="#f8faf7", input_background_fill_dark="#f8faf7",
        input_border_color="#dfe5df", input_border_color_dark="#dfe5df",
        button_primary_background_fill="#1e5946", button_primary_background_fill_dark="#1e5946",
        button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
    )
    with gr.Blocks(css=_CSS, title="RagAgent EDA · 文档工作台", theme=theme) as demo:
        gr.HTML(
            """
            <header class="app-header">
              <div class="brand"><span class="brand-mark" aria-hidden="true">R<span>·</span></span>
                <div class="brand-name">RagAgent<span>EDA KNOWLEDGE WORKSPACE</span></div>
              </div>
              <div class="header-context"><span class="nav-current">文档工作台</span><span class="header-divider"></span><span>TED / EDA</span></div>
            </header>
            <section class="workspace-intro">
              <div><div class="app-kicker">KNOWLEDGE, WITH CONTEXT</div>
                <h1>把技术文档，变成你的工作伙伴<span>。</span></h1>
                <p>查用法、理参数、找示例。从 TED 文档中获取回答，回到原文确认依据。</p>
              </div>
              <div class="intro-index" aria-hidden="true"><span>WORKSPACE</span><strong>01 / QA</strong></div>
            </section>
            """
        )
        with gr.Row(elem_classes=["workspace-layout"]):
            with gr.Column(scale=4, min_width=320, elem_classes=["workspace-panel", "query-console"]):
                gr.HTML('<div class="panel-head"><div><span class="panel-kicker">01 / ASK</span><h2>从一个问题开始</h2></div><span class="panel-tag">TED 文档</span></div>')
                question = gr.Textbox(
                    label="你的问题", placeholder="例如：如何配置瞬态仿真，并获取输出波形？",
                    lines=4, max_lines=10, elem_id="question-input",
                )
                ask_btn = gr.Button("检索并回答  →", variant="primary", elem_classes=["primary-action"])
                gr.HTML('<div class="examples-heading">也可以试试这些问题</div>')
                example_buttons = [gr.Button(text, size="sm", elem_classes=["example-question"]) for text in _EXAMPLE_QUESTIONS]
                with gr.Accordion("优化提问 · 可选", open=False, elem_classes=["rewrite-options"]) as rewrite_options:
                    gr.HTML('<p class="helper-text">问题不够明确时，可先生成改写建议。原始输入会保留。</p>')
                    rewrite_mode = gr.Radio(
                        label="改写方式", choices=[_REWRITE_MODE_CONSERVATIVE_LABEL, _REWRITE_MODE_AGGRESSIVE_LABEL],
                        value=_REWRITE_MODE_AGGRESSIVE_LABEL,
                    )
                    rewrite_btn = gr.Button("生成改写建议", elem_classes=["secondary-action"])
                    rewrite_result = gr.Textbox(
                        label="改写建议", placeholder="生成后可在这里继续编辑", lines=3, max_lines=9, interactive=True,
                    )
                    final_query_source = gr.Radio(
                        label="本次提问使用", choices=[("原始问题", _FINAL_SOURCE_ORIGINAL_LABEL), ("改写后的问题", _FINAL_SOURCE_REWRITE_LABEL)],
                        value=_FINAL_SOURCE_ORIGINAL_LABEL,
                    )
                    rewrite_meta_html = gr.HTML(_rewrite_status_html())
                rewrite_base_query_state = gr.State(value="")
                rewrite_strategy_state = gr.State(value="")
                rewrite_warning_state = gr.State(value="")
                user_id_state = gr.Textbox(value="legacy", visible=False, elem_id="ragagent-user-id")
                with gr.Group(elem_classes=["history-card"]):
                    with gr.Row(elem_classes=["history-head"]):
                        gr.HTML('<h3 class="history-title">最近对话</h3>')
                        refresh_history_btn = gr.Button("刷新", size="sm", min_width=52, scale=0, elem_classes=["history-refresh"])
                    history_select = gr.Radio(label="历史对话", choices=[], value=None, interactive=True, show_label=False, elem_classes=["history-list"])
                    history_status_html = gr.HTML('<div class="history-note">正在读取历史对话…</div>')

            with gr.Column(scale=7, min_width=440, elem_classes=["workspace-panel", "result-console"]):
                gr.HTML('<div class="panel-head"><div><span class="panel-kicker">02 / EXPLORE</span><h2>回答与发现</h2></div><span class="panel-tag">基于文档的回答</span></div>')
                warning_html = gr.HTML(visible=False, elem_classes=["warning-region"])
                status_html = gr.HTML('<span class="status-badge status-idle">等待提问</span>', elem_id="answer-status")
                answer_md = gr.Markdown(value=_EMPTY_ANSWER, elem_classes=["answer-md"])
                record_id_state = gr.State(value=None)
                with gr.Row(elem_classes=["feedback-row"]):
                    feedback_html = gr.HTML('<div class="feedback-note">回答后可评价，帮助改进问答质量。</div>')
                    useful_btn = gr.Button("有帮助", size="sm", min_width=76, scale=0, elem_classes=["feedback-action"])
                    useless_btn = gr.Button("需改进", size="sm", min_width=76, scale=0, elem_classes=["feedback-action"])
                gr.HTML('<div class="evidence-heading"><h3>参考依据 <span>SOURCES</span></h3><p>展开片段，核对原文</p></div>')
                evidence_html = gr.HTML(_render_evidence([]), elem_id="evidence-results")
        gr.HTML('<div class="workspace-footer"><span>RagAgent EDA</span><span>以文档为依据 · 让技术知识触手可及</span></div>')

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
                    warning_text = "还没有改写建议，请先生成建议，或切换为原始问题。"
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
                            "暂无回答内容。",
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
                            "暂无回答内容。",
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
                warning_text = "请先输入一个问题，或选择上方的示例。"
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
                        "暂无回答内容。",
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
                        result.answer if (result.answer or "").strip() else "暂无回答内容。",
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
                        "暂无回答内容。",
                        _render_evidence([]),
                        record_id,
                        _feedback_hint_html(record_id),
                    ),
                )

        def _choose_example(example: str):
            return (example, *_clear_rewrite_state(""))

        for example_button, example_text in zip(example_buttons, _EXAMPLE_QUESTIONS):
            example_button.click(
                lambda text=example_text: _choose_example(text), inputs=[],
                outputs=[question, rewrite_result, final_query_source, rewrite_meta_html,
                         rewrite_base_query_state, rewrite_strategy_state, rewrite_warning_state],
                show_progress="hidden",
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
        ask_event.then(fn=None, js=_SCROLL_TO_ANSWER_JS, show_progress="hidden")
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
        submit_event.then(fn=None, js=_SCROLL_TO_ANSWER_JS, show_progress="hidden")
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
