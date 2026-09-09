from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import hashlib
import re


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str
    block_id: str | None = None
    block_type: str = "text"
    part_index: int = 0
    part_count: int = 1


@dataclass
class _ChunkDraft:
    text: str
    block_id: str | None = None
    block_type: str = "text"
    part_index: int = 0
    part_count: int = 1


_HEADING_TAGS: Dict[str, int] = {
    "h1": 1,
    "h2": 2,
    "h3": 3,
    "h4": 4,
    "h5": 5,
    "h6": 6,
}

_BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "main",
    "ul",
    "ol",
    "li",
    "table",
    "thead",
    "tbody",
    "tr",
    "td",
    "th",
    "pre",
    "blockquote",
    "br",
    "hr",
    "dl",
    "dt",
    "dd",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
}

_NOISE_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "nav",
    "aside",
    "header",
    "footer",
    "form",
    "button",
    "svg",
}

_NOISE_MARKERS = (
    "navbar",
    "sidebar",
    "breadcrumb",
    "skip-link",
    "slimsearch",
    "vp-sidebar",
    "vp-navbar",
    "color-mode-switch",
    "print-button",
    "toggle-sidebar",
)

_MAIN_MARKERS = (
    "main-content",
    "theme-hope-content",
    "vp-page",
    "article-body",
    "doc-content",
    "markdown-body",
    "post-content",
    "content-body",
)


_FENCED_CODE_RE = re.compile(r"```[^\n]*\n[\s\S]*?```")
_INDEXER_SCHEMA_VERSION = "html-inline-code-list-v2"

_NO_SPACE_BEFORE = set(
    ",.:;!?)]}"
    "\uFF0C\u3002\uFF1A\uFF1B\uFF01\uFF1F\u3001\uFF09\u3011\u300B\u3009\u300D\u300F"
)
_NO_SPACE_AFTER = set("([{" "\uFF08\u3010\u300A\u3008\u300C\u300E")


def _is_cjk_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _needs_inline_space(prev_text: str, next_text: str) -> bool:
    if not prev_text or not next_text:
        return False
    if prev_text[-1].isspace():
        return False
    first = next_text[0]
    last = prev_text[-1]
    if first.isspace() or first in _NO_SPACE_BEFORE:
        return False
    if last in _NO_SPACE_AFTER:
        return False
    if _is_cjk_char(last) and _is_cjk_char(first):
        return False
    return True


def _append_inline_text(parts: List[str], text: str) -> None:
    if parts and _needs_inline_space(parts[-1], text):
        parts.append(" ")
    parts.append(text)


def _normalize_plain_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized


def _normalize_fenced_code_block(block: str) -> str:
    return block.replace("\r\n", "\n").replace("\r", "\n").strip("\n")


def _normalize_text_preserving_code(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    parts: List[str] = []
    last = 0
    for match in _FENCED_CODE_RE.finditer(normalized):
        parts.append(_normalize_plain_text(normalized[last:match.start()]))
        parts.append(_normalize_fenced_code_block(match.group(0)))
        last = match.end()
    parts.append(_normalize_plain_text(normalized[last:]))
    joined = "".join(parts)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


@dataclass
class _SectionCollector:
    heading_stack: List[str]
    sections: List[Tuple[str, str]]
    body_parts: List[str]
    heading_level: int | None
    heading_parts: List[str]

    @classmethod
    def create(cls) -> "_SectionCollector":
        return cls(heading_stack=[], sections=[], body_parts=[], heading_level=None, heading_parts=[])

    def start(self, tag: str) -> None:
        level = _HEADING_TAGS.get(tag)
        if level is not None:
            self._flush_body()
            self.heading_level = level
            self.heading_parts = []
            return
        if tag == "li":
            self._newline()
            self.body_parts.append("- ")
            return
        if tag in _BLOCK_TAGS:
            self._newline()

    def end(self, tag: str) -> None:
        level = _HEADING_TAGS.get(tag)
        if level is not None and self.heading_level is not None:
            title = self._normalize_inline("".join(self.heading_parts))
            if title:
                while len(self.heading_stack) >= level:
                    self.heading_stack.pop()
                while len(self.heading_stack) < level - 1:
                    self.heading_stack.append("")
                self.heading_stack.append(title)
            self.heading_level = None
            self.heading_parts = []
            self._newline()
            return
        if tag in _BLOCK_TAGS:
            self._newline()

    def data(self, data: str) -> None:
        text = self._normalize_inline(data)
        if not text:
            return
        if self.heading_level is not None:
            _append_inline_text(self.heading_parts, text)
            return
        _append_inline_text(self.body_parts, text)

    def finish(self) -> List[Tuple[str, str]]:
        self._flush_body()
        return self.sections

    def add_block(self, block: str) -> None:
        clean = block.strip()
        if not clean:
            return
        if self.body_parts and not self.body_parts[-1].endswith("\n"):
            self.body_parts.append("\n")
        if self.body_parts and not self.body_parts[-1].endswith("\n\n"):
            self.body_parts.append("\n")
        self.body_parts.append(clean)
        self.body_parts.append("\n\n")

    def _flush_body(self) -> None:
        if not self.body_parts:
            return
        body = _normalize_text_preserving_code("".join(self.body_parts))
        self.body_parts = []
        if not body:
            return
        title = " > ".join([h for h in self.heading_stack if h])
        self.sections.append((title, body))

    def _newline(self) -> None:
        if self.body_parts and not self.body_parts[-1].endswith("\n"):
            self.body_parts.append("\n")

    @staticmethod
    def _normalize_inline(text: str) -> str:
        return re.sub(r"\s+", " ", unescape(text)).strip()


@dataclass
class _NodeState:
    ignored: bool
    in_main: bool
    in_table: bool
    code_language: str = ""


@dataclass
class _CodeContext:
    in_main: bool
    language: str
    lines: List[str]
    plain_parts: List[str]
    current_line_parts: List[str] | None
    line_span_depth: int


@dataclass
class _TableContext:
    rows: List[List[str]]
    row_header_flags: List[bool]
    current_row: List[str] | None
    current_row_has_th: bool
    current_cell_parts: List[str] | None
    in_main: bool

    @classmethod
    def create(cls, in_main: bool) -> "_TableContext":
        return cls(
            rows=[],
            row_header_flags=[],
            current_row=None,
            current_row_has_th=False,
            current_cell_parts=None,
            in_main=in_main,
        )


class _HtmlSectionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.all_collector = _SectionCollector.create()
        self.main_collector = _SectionCollector.create()
        self._stack: List[_NodeState] = []
        self.seen_main = False
        self._table_stack: List[_TableContext] = []
        self._code_context: _CodeContext | None = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str | None]]) -> None:
        attr_map = {str(k).lower(): (v or "") for k, v in attrs}
        parent = self._stack[-1] if self._stack else _NodeState(ignored=False, in_main=False, in_table=False)
        ignored = parent.ignored or self._is_noise(tag, attr_map)
        main_marker = self._is_main(tag, attr_map)
        in_main = (not ignored) and (parent.in_main or main_marker)
        in_table = (not ignored) and (parent.in_table or tag == "table")
        code_language = self._language_from_attrs(attr_map) or parent.code_language
        self._stack.append(
            _NodeState(ignored=ignored, in_main=in_main, in_table=in_table, code_language=code_language)
        )

        if ignored:
            return
        if self._code_context is not None:
            self._handle_code_start(tag, attr_map)
            return
        if tag == "pre":
            self._start_code_block(in_main=in_main, language=code_language)
            return
        if self._handle_table_start(tag, in_main):
            return
        self.all_collector.start(tag)
        if in_main:
            self.main_collector.start(tag)
            self.seen_main = True

    def handle_endtag(self, tag: str) -> None:
        if not self._stack:
            return
        state = self._stack.pop()
        if state.ignored:
            return
        if self._code_context is not None:
            self._handle_code_end(tag)
            return
        if self._handle_table_end(tag):
            return
        self.all_collector.end(tag)
        if state.in_main:
            self.main_collector.end(tag)

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        state = self._stack[-1] if self._stack else _NodeState(ignored=False, in_main=False, in_table=False)
        if state.ignored:
            return
        if self._code_context is not None:
            self._handle_code_data(data)
            return
        if self._handle_table_data(data):
            return
        self.all_collector.data(data)
        if state.in_main:
            self.main_collector.data(data)

    def finish(self) -> List[Tuple[str, str]]:
        all_sections = self.all_collector.finish()
        main_sections = self.main_collector.finish()
        if self.seen_main and self._total_text_len(main_sections) >= 120:
            return main_sections
        return main_sections or all_sections

    @staticmethod
    def _attrs_blob(attr_map: Dict[str, str]) -> str:
        return " ".join(
            [
                attr_map.get("id", ""),
                attr_map.get("class", ""),
                attr_map.get("role", ""),
                attr_map.get("aria-label", ""),
            ]
        ).lower()

    @classmethod
    def _is_noise(cls, tag: str, attr_map: Dict[str, str]) -> bool:
        if tag in _NOISE_TAGS:
            return True
        blob = cls._attrs_blob(attr_map)
        return any(marker in blob for marker in _NOISE_MARKERS)

    @classmethod
    def _is_main(cls, tag: str, attr_map: Dict[str, str]) -> bool:
        if tag in {"main", "article"}:
            return True
        blob = cls._attrs_blob(attr_map)
        return any(marker in blob for marker in _MAIN_MARKERS)

    @staticmethod
    def _total_text_len(sections: List[Tuple[str, str]]) -> int:
        return sum(len(body) for _, body in sections)

    @staticmethod
    def _language_from_attrs(attr_map: Dict[str, str]) -> str:
        for key in ("data-ext", "data-title"):
            value = attr_map.get(key, "").strip().lower()
            if value and re.fullmatch(r"[a-z0-9_+-]+", value):
                return value

        blob = " ".join(
            [
                attr_map.get("class", ""),
                attr_map.get("data-highlighter", ""),
            ]
        )
        match = re.search(r"(?:^|\s)language-([A-Za-z0-9_+-]+)", blob)
        if match:
            return match.group(1).lower()
        return ""

    @staticmethod
    def _is_code_line_span(tag: str, attr_map: Dict[str, str]) -> bool:
        return tag == "span" and "line" in attr_map.get("class", "").split()

    def _start_code_block(self, in_main: bool, language: str) -> None:
        self._code_context = _CodeContext(
            in_main=in_main,
            language=language,
            lines=[],
            plain_parts=[],
            current_line_parts=None,
            line_span_depth=0,
        )

    def _handle_code_start(self, tag: str, attr_map: Dict[str, str]) -> None:
        context = self._code_context
        if context is None:
            return
        language = self._language_from_attrs(attr_map)
        if language and not context.language:
            context.language = language
        if tag == "br":
            if context.current_line_parts is not None:
                context.current_line_parts.append("\n")
            else:
                context.plain_parts.append("\n")
            return
        if self._is_code_line_span(tag, attr_map):
            if context.current_line_parts is not None:
                context.lines.append("".join(context.current_line_parts))
            context.current_line_parts = []
            context.line_span_depth = 1
            return
        if tag == "span" and context.current_line_parts is not None:
            context.line_span_depth += 1

    def _handle_code_end(self, tag: str) -> None:
        context = self._code_context
        if context is None:
            return
        if tag == "span" and context.current_line_parts is not None and context.line_span_depth > 0:
            context.line_span_depth -= 1
            if context.line_span_depth == 0:
                context.lines.append("".join(context.current_line_parts))
                context.current_line_parts = None
            return
        if tag == "pre":
            self._finish_code_block()

    def _handle_code_data(self, data: str) -> None:
        context = self._code_context
        if context is None:
            return
        if context.current_line_parts is not None:
            context.current_line_parts.append(unescape(data))
        else:
            context.plain_parts.append(unescape(data))

    def _finish_code_block(self) -> None:
        context = self._code_context
        if context is None:
            return
        if context.current_line_parts is not None:
            context.lines.append("".join(context.current_line_parts))
        if context.lines:
            code = "\n".join(context.lines)
        else:
            code = "".join(context.plain_parts)
        code = code.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        self._code_context = None
        if not code.strip():
            return
        opening = f"```{context.language}" if context.language else "```"
        block = f"{opening}\n{code}\n```"
        self.all_collector.add_block(block)
        if context.in_main:
            self.main_collector.add_block(block)

    def _handle_table_start(self, tag: str, in_main: bool) -> bool:
        if tag == "table":
            self._table_stack.append(_TableContext.create(in_main=in_main))
            return True
        if not self._table_stack:
            return False
        table = self._table_stack[-1]
        if tag == "tr":
            table.current_row = []
            table.current_row_has_th = False
            return True
        if tag in {"td", "th"}:
            table.current_cell_parts = []
            if tag == "th":
                table.current_row_has_th = True
            return True
        if tag == "br" and table.current_cell_parts is not None:
            table.current_cell_parts.append("\n")
            return True
        if tag in {"p", "div", "li"} and table.current_cell_parts is not None and table.current_cell_parts:
            table.current_cell_parts.append("\n")
            return True
        return False

    def _handle_table_end(self, tag: str) -> bool:
        if not self._table_stack:
            return False
        table = self._table_stack[-1]
        if tag in {"td", "th"}:
            if table.current_row is None:
                table.current_row = []
            cell = self._normalize_table_cell("".join(table.current_cell_parts or []))
            table.current_row.append(cell)
            table.current_cell_parts = None
            return True
        if tag == "tr":
            if table.current_cell_parts is not None:
                if table.current_row is None:
                    table.current_row = []
                cell = self._normalize_table_cell("".join(table.current_cell_parts))
                table.current_row.append(cell)
                table.current_cell_parts = None
            if table.current_row is not None:
                if table.current_row:
                    table.rows.append(table.current_row)
                    table.row_header_flags.append(table.current_row_has_th)
                table.current_row = None
            table.current_row_has_th = False
            return True
        if tag == "table":
            if table.current_cell_parts is not None:
                if table.current_row is None:
                    table.current_row = []
                cell = self._normalize_table_cell("".join(table.current_cell_parts))
                table.current_row.append(cell)
                table.current_cell_parts = None
            if table.current_row:
                table.rows.append(table.current_row)
                table.row_header_flags.append(table.current_row_has_th)
            block = self._table_to_markdown(table.rows, table.row_header_flags)
            self._table_stack.pop()
            if block:
                self.all_collector.add_block(block)
                if table.in_main:
                    self.main_collector.add_block(block)
            return True
        return True

    def _handle_table_data(self, data: str) -> bool:
        if not self._table_stack:
            return False
        table = self._table_stack[-1]
        if table.current_cell_parts is None:
            return True
        text = self._normalize_table_cell(data)
        if not text:
            return True
        _append_inline_text(table.current_cell_parts, text)
        return True

    @staticmethod
    def _normalize_table_cell(text: str) -> str:
        normalized = unescape(text)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
        normalized = re.sub(r"\n[ \t]+", "\n", normalized)
        normalized = re.sub(r"[ \t]+\n", "\n", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    @classmethod
    def _table_to_markdown(cls, rows: List[List[str]], row_header_flags: List[bool]) -> str:
        clean_rows: List[List[str]] = []
        clean_flags: List[bool] = []
        for row, has_header in zip(rows, row_header_flags):
            normalized = [cls._normalize_table_cell(cell) for cell in row]
            if any(cell for cell in normalized):
                clean_rows.append(normalized)
                clean_flags.append(has_header)
        if not clean_rows:
            return ""

        max_cols = max(len(row) for row in clean_rows)
        normalized_rows = [row + [""] * (max_cols - len(row)) for row in clean_rows]
        header = normalized_rows[0]
        data_rows = normalized_rows[1:]

        lines: List[str] = []
        lines.append(cls._table_row_to_markdown(header))
        lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        for row in data_rows:
            lines.append(cls._table_row_to_markdown(row))
        if not data_rows and not any(clean_flags):
            lines.append(cls._table_row_to_markdown([""] * max_cols))
        return "\n".join(lines).strip()

    @staticmethod
    def _table_row_to_markdown(cells: List[str]) -> str:
        escaped = [cell.replace("|", r"\|").replace("\n", "<br>") for cell in cells]
        return "| " + " | ".join(escaped) + " |"


class ResourceIndexer:
    def __init__(self, resource_dir: Path) -> None:
        self.resource_dir = resource_dir

    def index(self) -> List[Chunk]:
        chunks: List[Chunk] = []
        for path in self.list_docs():
            raw = path.read_text(encoding='utf-8', errors='ignore')
            sections = self._extract_sections(path, raw)
            for idx, part in enumerate(self._chunk_sections(sections, source_key=str(path))):
                text = part.text.strip()
                if len(text) < 50:
                    continue
                cid = hashlib.md5(f'{path}:{idx}:{part.block_id or ""}:{part.part_index}'.encode('utf-8')).hexdigest()[:12]
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        source=str(path),
                        text=text,
                        block_id=part.block_id,
                        block_type=part.block_type,
                        part_index=part.part_index,
                        part_count=part.part_count,
                    )
                )
        return chunks

    def list_docs(self) -> List[Path]:
        return list(self._iter_docs(self.resource_dir))

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(_INDEXER_SCHEMA_VERSION.encode("utf-8"))
        for p in sorted(self.list_docs()):
            st = p.stat()
            h.update(str(p).encode('utf-8'))
            h.update(str(st.st_size).encode('utf-8'))
            h.update(str(st.st_mtime_ns).encode('utf-8'))
        return h.hexdigest()

    def _iter_docs(self, root: Path) -> Iterable[Path]:
        if not root.exists():
            return []
        patterns = ('*.md', '*.markdown', '*.html', '*.htm', '*.txt')
        files: List[Path] = []
        for p in patterns:
            files.extend(root.rglob(p))
        return files

    def _extract_sections(self, path: Path, raw: str) -> List[Tuple[str, str]]:
        if path.suffix.lower() in {".html", ".htm"}:
            html_sections = self._sections_from_html(raw)
            if html_sections:
                return html_sections
        return self._sections_from_text(raw)

    def _sections_from_html(self, raw: str) -> List[Tuple[str, str]]:
        parser = _HtmlSectionParser()
        parser.feed(raw)
        parser.close()
        return parser.finish()

    def _sections_from_text(self, raw: str) -> List[Tuple[str, str]]:
        text = self._normalize_text(raw)
        if not text:
            return []

        sections: List[Tuple[str, str]] = []
        heading_stack: List[str] = []
        body_lines: List[str] = []

        def flush_body() -> None:
            if not body_lines:
                return
            body = self._normalize_text("\n".join(body_lines))
            body_lines.clear()
            if not body:
                return
            title = " > ".join([h for h in heading_stack if h])
            sections.append((title, body))

        in_fence = False
        for line in text.split("\n"):
            if line.strip().startswith("```"):
                body_lines.append(line)
                in_fence = not in_fence
                continue
            if in_fence:
                body_lines.append(line)
                continue
            m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if not m:
                body_lines.append(line)
                continue

            flush_body()
            level = len(m.group(1))
            heading = self._normalize_inline(m.group(2))
            while len(heading_stack) >= level:
                heading_stack.pop()
            while len(heading_stack) < level - 1:
                heading_stack.append("")
            heading_stack.append(heading)

        flush_body()
        return sections or [("", text)]

    def _chunk_sections(
        self,
        sections: List[Tuple[str, str]],
        size: int = 1200,
        overlap: int = 200,
        source_key: str = "",
    ) -> List[_ChunkDraft]:
        out: List[_ChunkDraft] = []
        for section_idx, (title, body) in enumerate(sections):
            clean_body = self._normalize_text(body)
            if not clean_body:
                continue
            prefix = f"Section: {title}\n" if title else ""
            part_size = max(200, size - len(prefix))
            block_namespace = f"{source_key}:{section_idx}:{title}"
            for part in self._split_section(
                clean_body,
                size=part_size,
                overlap=overlap,
                block_namespace=block_namespace,
            ):
                content = f"{prefix}{part.text.strip()}".strip()
                if content:
                    out.append(
                        _ChunkDraft(
                            text=content,
                            block_id=part.block_id,
                            block_type=part.block_type,
                            part_index=part.part_index,
                            part_count=part.part_count,
                        )
                    )
        return out

    def _split_section(
        self,
        text: str,
        size: int,
        overlap: int,
        block_namespace: str = "",
    ) -> List[_ChunkDraft]:
        paragraphs = self._split_paragraphs_preserving_code(text)
        if not paragraphs:
            return [_ChunkDraft(text=part) for part in self._split(text, size=size, overlap=overlap)]

        raw_parts: List[_ChunkDraft] = []
        current: List[str] = []
        code_block_idx = 0

        def flush_current() -> None:
            if not current:
                return
            raw_parts.append(_ChunkDraft(text="\n\n".join(current).strip()))
            current.clear()

        for paragraph in paragraphs:
            if self._is_fenced_code_block(paragraph):
                flush_current()
                code_parts = self._split_fenced_code_block(paragraph, size=size)
                block_key = f"{block_namespace}:code:{code_block_idx}:{hashlib.md5(paragraph.encode('utf-8')).hexdigest()[:12]}"
                block_id = hashlib.md5(block_key.encode("utf-8")).hexdigest()[:16]
                part_count = len(code_parts)
                for part_index, code_part in enumerate(code_parts):
                    raw_parts.append(
                        _ChunkDraft(
                            text=code_part,
                            block_id=block_id,
                            block_type="code",
                            part_index=part_index,
                            part_count=part_count,
                        )
                    )
                code_block_idx += 1
                continue
            if len(paragraph) > size:
                flush_current()
                raw_parts.extend(
                    _ChunkDraft(text=part)
                    for part in self._split_long_paragraph(paragraph, size=size, overlap=overlap)
                )
                continue

            candidate = "\n\n".join(current + [paragraph]).strip()
            if not current or len(candidate) <= size:
                current.append(paragraph)
            else:
                flush_current()
                current.append(paragraph)
        flush_current()

        if not raw_parts:
            raw_parts = [_ChunkDraft(text=part) for part in self._split(text, size=size, overlap=overlap)]

        if overlap <= 0:
            return [part for part in raw_parts if part]

        # Keep overlap while preserving paragraph-first chunks.
        stitched: List[_ChunkDraft] = []
        prev_tail = ""
        prev_was_code = False
        for part in raw_parts:
            chunk = part.text.strip()
            if not chunk:
                continue
            current_is_code = part.block_type == "code" or self._is_fenced_code_block(chunk)
            if prev_tail and not current_is_code and not prev_was_code and not chunk.startswith("- "):
                allowed = max(0, size - len(chunk) - 1)
                if allowed > 0:
                    overlap_text = prev_tail[-allowed:]
                    chunk = f"{overlap_text}\n{chunk}".strip()
            stitched.append(
                _ChunkDraft(
                    text=chunk,
                    block_id=part.block_id,
                    block_type=part.block_type,
                    part_index=part.part_index,
                    part_count=part.part_count,
                )
            )
            prev_tail = "" if current_is_code or overlap <= 0 else chunk[-overlap:]
            prev_was_code = current_is_code
        return stitched

    @staticmethod
    def _split_paragraphs_preserving_code(text: str) -> List[str]:
        paragraphs: List[str] = []
        current: List[str] = []
        in_fence = False

        def flush() -> None:
            nonlocal current
            block = "\n".join(current).strip("\n")
            if block.strip():
                paragraphs.append(block)
            current = []

        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            stripped = line.strip()
            if stripped.startswith("```"):
                if not in_fence and current:
                    flush()
                current.append(line)
                if in_fence:
                    in_fence = False
                    flush()
                else:
                    in_fence = True
                continue
            if in_fence:
                current.append(line)
                continue
            if not stripped:
                flush()
                continue
            current.append(line)
        flush()
        return paragraphs

    @staticmethod
    def _is_fenced_code_block(text: str) -> bool:
        stripped = text.strip()
        return stripped.startswith("```") and stripped.endswith("```") and "\n" in stripped

    @classmethod
    def _split_fenced_code_block(cls, block: str, size: int) -> List[str]:
        stripped = block.strip("\n")
        if len(stripped) <= size:
            return [stripped]
        lines = stripped.split("\n")
        if len(lines) <= 2:
            return [stripped]

        opening = lines[0]
        closing = lines[-1] if lines[-1].strip().startswith("```") else "```"
        code_lines = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
        parts: List[str] = []
        current: List[str] = []

        def emit() -> None:
            nonlocal current
            code = "\n".join(current)
            fenced = f"{opening}\n{code}\n{closing}".strip("\n")
            if fenced.strip():
                parts.append(fenced)
            current = []

        for line in code_lines:
            candidate = f"{opening}\n{chr(10).join(current + [line])}\n{closing}"
            if current and len(candidate) > size:
                emit()
            current.append(line)
        if current or not parts:
            emit()
        return parts

    def _split_long_paragraph(self, paragraph: str, size: int, overlap: int) -> List[str]:
        if self._is_list_block(paragraph):
            return self._split_line_block(paragraph, size=size)

        sentence_units = [s.strip() for s in self._split_sentences(paragraph) if s.strip()]
        if len(sentence_units) <= 1:
            return self._split(paragraph, size=size, overlap=overlap)

        parts: List[str] = []
        current = ""
        for sentence in sentence_units:
            if len(sentence) > size:
                if current:
                    parts.append(current.strip())
                    current = ""
                parts.extend(self._split(sentence, size=size, overlap=overlap))
                continue

            candidate = sentence if not current else f"{current} {sentence}"
            if len(candidate) <= size:
                current = candidate
            else:
                parts.append(current.strip())
                current = sentence
        if current:
            parts.append(current.strip())
        return parts or self._split(paragraph, size=size, overlap=overlap)

    @staticmethod
    def _is_list_block(paragraph: str) -> bool:
        lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
        return sum(1 for line in lines if line.startswith("- ")) >= 2

    def _split_line_block(self, paragraph: str, size: int) -> List[str]:
        parts: List[str] = []
        current: List[str] = []

        def emit() -> None:
            nonlocal current
            block = "\n".join(current).strip()
            if block:
                parts.append(block)
            current = []

        for line in [line.rstrip() for line in paragraph.split("\n") if line.strip()]:
            if len(line) > size:
                emit()
                parts.extend(self._split(line, size=size, overlap=0))
                continue
            candidate = "\n".join(current + [line]).strip()
            if current and len(candidate) > size:
                emit()
            current.append(line)
        emit()
        return parts or self._split(paragraph, size=size, overlap=0)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[。！？!?；;])\s+|(?<=\.)\s+(?=[A-Z0-9\u4e00-\u9fff])", text.strip())
        return [p for p in parts if p.strip()]

    @staticmethod
    def _normalize_inline(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return _normalize_text_preserving_code(text)

    @staticmethod
    def _split(text: str, size: int = 1200, overlap: int = 200) -> List[str]:
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = max(0, end - overlap)
        return chunks
