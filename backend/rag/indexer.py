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
    "code",
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
        if tag in _BLOCK_TAGS:
            self._newline()

    def end(self, tag: str) -> None:
        level = _HEADING_TAGS.get(tag)
        if level is not None and self.heading_level is not None:
            title = self._normalize_inline(" ".join(self.heading_parts))
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
            self.heading_parts.append(text)
            return
        if self.body_parts and not self.body_parts[-1].endswith(("\n", " ")):
            self.body_parts.append(" ")
        self.body_parts.append(text)

    def finish(self) -> List[Tuple[str, str]]:
        self._flush_body()
        return self.sections

    def _flush_body(self) -> None:
        if not self.body_parts:
            return
        body = "".join(self.body_parts)
        body = re.sub(r"[ \t]+\n", "\n", body)
        body = re.sub(r"\n[ \t]+", "\n", body)
        body = re.sub(r"\n{3,}", "\n\n", body)
        body = re.sub(r"[ \t]{2,}", " ", body).strip()
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


class _HtmlSectionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.all_collector = _SectionCollector.create()
        self.main_collector = _SectionCollector.create()
        self._stack: List[_NodeState] = []
        self.seen_main = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str | None]]) -> None:
        attr_map = {str(k).lower(): (v or "") for k, v in attrs}
        parent = self._stack[-1] if self._stack else _NodeState(ignored=False, in_main=False)
        ignored = parent.ignored or self._is_noise(tag, attr_map)
        main_marker = self._is_main(tag, attr_map)
        in_main = (not ignored) and (parent.in_main or main_marker)
        self._stack.append(_NodeState(ignored=ignored, in_main=in_main))

        if ignored:
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
        self.all_collector.end(tag)
        if state.in_main:
            self.main_collector.end(tag)

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        state = self._stack[-1] if self._stack else _NodeState(ignored=False, in_main=False)
        if state.ignored:
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


class ResourceIndexer:
    def __init__(self, resource_dir: Path) -> None:
        self.resource_dir = resource_dir

    def index(self) -> List[Chunk]:
        chunks: List[Chunk] = []
        for path in self.list_docs():
            raw = path.read_text(encoding='utf-8', errors='ignore')
            sections = self._extract_sections(path, raw)
            for idx, part in enumerate(self._chunk_sections(sections)):
                if len(part.strip()) < 50:
                    continue
                cid = hashlib.md5(f'{path}:{idx}'.encode('utf-8')).hexdigest()[:12]
                chunks.append(Chunk(chunk_id=cid, source=str(path), text=part.strip()))
        return chunks

    def list_docs(self) -> List[Path]:
        return list(self._iter_docs(self.resource_dir))

    def fingerprint(self) -> str:
        h = hashlib.sha256()
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

        for line in text.split("\n"):
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

    def _chunk_sections(self, sections: List[Tuple[str, str]], size: int = 1200, overlap: int = 200) -> List[str]:
        out: List[str] = []
        for title, body in sections:
            clean_body = self._normalize_text(body)
            if not clean_body:
                continue
            prefix = f"Section: {title}\n" if title else ""
            part_size = max(200, size - len(prefix))
            for part in self._split(clean_body, size=part_size, overlap=overlap):
                content = f"{prefix}{part.strip()}".strip()
                if content:
                    out.append(content)
        return out

    @staticmethod
    def _normalize_inline(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

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
