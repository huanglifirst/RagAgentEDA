from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

from backend.rag.indexer import Chunk
from backend.rag.retriever import ScoredChunk


_CODE_FENCE_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_+-]*)[ \t]*\n(?P<code>[\s\S]*?)```")


def expand_code_block_evidence(evidence: Iterable[ScoredChunk], chunks: Iterable[Chunk]) -> List[ScoredChunk]:
    """Expand hits from split code blocks back to their full fenced block."""

    block_index = _build_code_block_index(chunks)
    expanded: List[ScoredChunk] = []
    seen_blocks: set[Tuple[str, str]] = set()

    for item in evidence:
        chunk = item.chunk
        key = _code_block_key(chunk)
        if key is None or chunk.part_count <= 1:
            expanded.append(item)
            continue

        if key in seen_blocks:
            continue
        seen_blocks.add(key)

        block_parts = block_index.get(key)
        if not block_parts:
            expanded.append(item)
            continue

        merged = _merge_code_block_parts(block_parts)
        if merged is None:
            expanded.append(item)
            continue

        expanded.append(ScoredChunk(chunk=merged, score=item.score))

    return expanded


def _build_code_block_index(chunks: Iterable[Chunk]) -> Dict[Tuple[str, str], List[Chunk]]:
    index: Dict[Tuple[str, str], List[Chunk]] = {}
    for chunk in chunks:
        key = _code_block_key(chunk)
        if key is None:
            continue
        index.setdefault(key, []).append(chunk)
    for parts in index.values():
        parts.sort(key=lambda c: (c.part_index, c.chunk_id))
    return index


def _code_block_key(chunk: Chunk) -> Tuple[str, str] | None:
    if chunk.block_type != "code" or not chunk.block_id:
        return None
    return (chunk.source, chunk.block_id)


def _merge_code_block_parts(parts: List[Chunk]) -> Chunk | None:
    if not parts:
        return None

    prefix = ""
    language = ""
    code_segments: List[str] = []
    fallback_text: List[str] = []

    for part in parts:
        text = part.text.strip()
        match = _CODE_FENCE_RE.search(text)
        if not match:
            fallback_text.append(text)
            continue
        if not prefix:
            prefix = text[: match.start()].strip()
        if not language:
            language = match.group("lang").strip()
        code_segments.append(match.group("code").strip("\n"))

    if not code_segments:
        merged_text = "\n\n".join(fallback_text).strip()
    else:
        code = "\n".join(segment for segment in code_segments if segment).strip("\n")
        opening = f"```{language}" if language else "```"
        merged_text = f"{opening}\n{code}\n```"
        if prefix:
            merged_text = f"{prefix}\n{merged_text}"

    if not merged_text:
        return None

    first = parts[0]
    return Chunk(
        chunk_id=f"{first.block_id}:expanded",
        source=first.source,
        text=merged_text,
        block_id=first.block_id,
        block_type="code",
        part_index=0,
        part_count=max(part.part_count for part in parts),
    )
