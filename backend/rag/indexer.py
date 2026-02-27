from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
import hashlib
import re


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str


class ResourceIndexer:
    def __init__(self, resource_dir: Path) -> None:
        self.resource_dir = resource_dir

    def index(self) -> List[Chunk]:
        chunks: List[Chunk] = []
        for path in self.list_docs():
            raw = path.read_text(encoding='utf-8', errors='ignore')
            text = self._normalize(raw)
            for idx, part in enumerate(self._split(text)):
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

    @staticmethod
    def _normalize(raw: str) -> str:
        no_script = re.sub(r'<script[\\s\\S]*?</script>', ' ', raw, flags=re.IGNORECASE)
        no_style = re.sub(r'<style[\\s\\S]*?</style>', ' ', no_script, flags=re.IGNORECASE)
        no_tags = re.sub(r'<[^>]+>', ' ', no_style)
        squashed = re.sub(r'\s+', ' ', no_tags)
        return squashed

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
