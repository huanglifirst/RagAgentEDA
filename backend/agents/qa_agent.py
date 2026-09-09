from __future__ import annotations

from dataclasses import dataclass
import re
from threading import Lock
from time import perf_counter
from typing import List, Set, Tuple

from backend.config import settings
from backend.llm.client import OpenAICompatClient
from backend.rag.evidence import expand_code_block_evidence
from backend.rag.indexer import Chunk, ResourceIndexer
from backend.rag.retriever import HybridRetriever, ScoredChunk
from backend.rag.vector_store import EmbeddingRetriever, PersistentEmbeddingIndex
from backend.schemas.api import EvidenceItem, RagAskResponse

_ABSTAIN_TEXT = "\u672a\u68c0\u7d22\u5230\u76f8\u5173\u5185\u5bb9"
_INSUFFICIENT_EXAMPLE_TEXT = "\u8bc1\u636e\u4e0d\u8db3\u4ee5\u751f\u6210\u53ef\u9760\u793a\u4f8b"
_REPLACE_NOTE_TEXT = "\u3010\u9700\u81ea\u884c\u66ff\u6362\u3011"
_TOP_K = 6
_PROMPT_EVIDENCE_LIMIT = 6
_CODE_SEGMENT_CHAR_LIMIT = 6000
_TEXT_SEGMENT_CHAR_LIMIT = 500

_EN_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_CODE_FENCE_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_+-]*)[ \t]*\n(?P<code>[\s\S]*?)```")
_USAGE_QUESTION_RE = re.compile(
    "(?:"
    "\\u5982\\u4f55|\\u600e\\u4e48|\\u600e\\u6837|\\u7528\\u6cd5|\\u4f7f\\u7528|\\u8c03\\u7528|"
    "\\u793a\\u4f8b|\\u6837\\u4f8b|\\u4f8b\\u5b50|\\u4ee3\\u7801|\\u811a\\u672c|"
    "\\u6700\\u5c0f\\u53ef\\u8fd0\\u884c|\\u53c2\\u6570\\u7ec4\\u5408|"
    "example|examples|sample|snippet|usage|how\\s+to|call|use\\b|invoke"
    ")",
    flags=re.IGNORECASE,
)
_PLACEHOLDER_PATTERNS = (
    re.compile(r"<[^>\n]{1,80}>"),
    re.compile(r"\bYOUR_[A-Z0-9_]+\b"),
    re.compile(r"\byour_[a-z0-9_]+\b"),
    re.compile(r"\b(?:xxx|xxx_[A-Za-z0-9_]+|module_name|net_name|file_name|class_name|function_name)\b"),
    re.compile(r"\bpath/to/[A-Za-z0-9_./-]*\b", flags=re.IGNORECASE),
    re.compile(r"\b(?:your|custom|example|sample)[-_](?:path|module|net|node|port|name|file)\b", flags=re.IGNORECASE),
)


@dataclass
class _RetrievalResult:
    evidence: List[ScoredChunk]
    retrieval_mode: str
    warning: str
    top1_score: float
    qa_overlap: float
    vector_search_ms: float
    lexical_broad_ms: float
    rerank_ms: float


@dataclass
class _RuntimeAssets:
    chunks: List[Chunk]
    fingerprint: str
    emb: EmbeddingRetriever
    lexical_retriever: HybridRetriever


class RagQaAgent:
    def __init__(
        self,
        indexer: ResourceIndexer,
        chat_client: OpenAICompatClient,
        embedding_client: OpenAICompatClient,
        rerank_client: OpenAICompatClient,
        vector_store: PersistentEmbeddingIndex,
    ) -> None:
        self.indexer = indexer
        self.chat_client = chat_client
        self.embedding_client = embedding_client
        self.rerank_client = rerank_client
        self.vector_store = vector_store
        self._asset_lock = Lock()
        self._runtime_assets: _RuntimeAssets | None = None

    def invalidate_cache(self) -> None:
        with self._asset_lock:
            self._runtime_assets = None

    def warmup(self) -> None:
        self._get_runtime_assets()

    def _get_runtime_assets(self) -> Tuple[_RuntimeAssets, bool]:
        with self._asset_lock:
            if self._runtime_assets is not None:
                return self._runtime_assets, True

            chunks = self.indexer.index()
            if not chunks:
                raise RuntimeError(
                    f"resource indexing failed: no eligible documents found in {self.indexer.resource_dir}"
                )

            fingerprint = self.indexer.fingerprint()
            emb = EmbeddingRetriever.from_chunks(
                self.embedding_client,
                settings.embedding_model_text,
                chunks,
                self.vector_store,
                fingerprint,
                force_rebuild=False,
            )
            lexical_retriever = HybridRetriever(chunks)
            self._runtime_assets = _RuntimeAssets(
                chunks=chunks,
                fingerprint=fingerprint,
                emb=emb,
                lexical_retriever=lexical_retriever,
            )
            return self._runtime_assets, False

    def ask(self, question: str) -> RagAskResponse:
        ask_started = perf_counter()
        status = "error"
        retrieval_mode = "unknown"
        cache_hit: bool | None = None
        assets_prepare_ms = 0.0
        vector_search_ms = 0.0
        lexical_broad_ms = 0.0
        rerank_ms = 0.0
        answer_llm_ms = 0.0
        query = question.strip()
        if not query:
            raise ValueError("question is empty")
        try:
            prepare_started = perf_counter()
            assets, cache_hit = self._get_runtime_assets()
            assets_prepare_ms = (perf_counter() - prepare_started) * 1000.0

            rerank_top_n = max(20, _TOP_K * max(1, settings.rerank_topn_factor))
            retrieval = self._retrieve_with_fallback(
                query,
                assets.chunks,
                assets.emb,
                assets.lexical_retriever,
                rerank_top_n,
            )
            retrieval_mode = retrieval.retrieval_mode
            vector_search_ms = retrieval.vector_search_ms
            lexical_broad_ms = retrieval.lexical_broad_ms
            rerank_ms = retrieval.rerank_ms
            expanded_evidence = expand_code_block_evidence(retrieval.evidence, assets.chunks)
            evidence_items = [
                EvidenceItem(source=item.chunk.source, score=round(item.score, 6), snippet=item.chunk.text)
                for item in expanded_evidence[:_TOP_K]
            ]

            warning = retrieval.warning
            if self._should_abstain(retrieval.top1_score, retrieval.qa_overlap):
                reason = "pre_abstain_low_retrieval"
                warning = f"{warning}; {reason}" if warning else reason
                status = "not_found"
                return RagAskResponse(status=status, answer=_ABSTAIN_TEXT, evidence=evidence_items, warning=warning)

            answer_started = perf_counter()
            answer = self._build_answer(query, expanded_evidence)
            answer_llm_ms = (perf_counter() - answer_started) * 1000.0
            model_abstained = (not answer) or (_ABSTAIN_TEXT in answer)
            if model_abstained:
                reason = "model_abstain"
                warning = f"{warning}; {reason}" if warning else reason
                status = "not_found"
                return RagAskResponse(status=status, answer=_ABSTAIN_TEXT, evidence=evidence_items, warning=warning)

            status = "answered"
            return RagAskResponse(status=status, answer=answer, evidence=evidence_items, warning=warning)
        finally:
            if settings.rag_qa_timing_log:
                total_ms = (perf_counter() - ask_started) * 1000.0
                print(
                    "[RAG-QA-Timing] "
                    f"status={status} "
                    f"mode={retrieval_mode} "
                    f"cache_hit={'na' if cache_hit is None else ('1' if cache_hit else '0')} "
                    f"assets_prepare_ms={assets_prepare_ms:.2f} "
                    f"vector_search_ms={vector_search_ms:.2f} "
                    f"lexical_broad_ms={lexical_broad_ms:.2f} "
                    f"rerank_ms={rerank_ms:.2f} "
                    f"answer_llm_ms={answer_llm_ms:.2f} "
                    f"total_ms={total_ms:.2f}"
                )

    def _retrieve_with_fallback(
        self,
        query: str,
        chunks: List[Chunk],
        emb: EmbeddingRetriever,
        lexical_retriever: HybridRetriever,
        rerank_top_n: int,
    ) -> _RetrievalResult:
        warning_parts: List[str] = []
        retrieval_mode = "vector_lexical_rerank"
        vector_candidates: List[Chunk] = []
        lexical_candidates: List[Chunk] = []
        vector_search_ms = 0.0
        lexical_broad_ms = 0.0
        rerank_ms = 0.0

        try:
            vector_started = perf_counter()
            emb_hits = emb.search(query, top_k=rerank_top_n)
            vector_search_ms = (perf_counter() - vector_started) * 1000.0
            vector_candidates = [h.chunk for h in emb_hits]
            if not vector_candidates:
                warning_parts.append("embedding retrieval returned no hits; using lexical supplement")
        except Exception as exc:  # noqa: BLE001
            warning_parts.append(f"embedding retrieval unavailable: {exc}; using lexical supplement")
            vector_search_ms = (perf_counter() - vector_started) * 1000.0

        lexical_started = perf_counter()
        lexical_candidates = [item.chunk for item in lexical_retriever.retrieve_broad(query, top_n=rerank_top_n)]
        lexical_broad_ms = (perf_counter() - lexical_started) * 1000.0
        candidates = self._merge_candidates(
            vector_candidates,
            lexical_candidates,
            limit=max(1, rerank_top_n) * 2,
        )
        if not candidates:
            candidates = chunks
            warning_parts.append("candidate merge produced no hits; using full corpus")

        warning = "; ".join(part for part in warning_parts if part)
        reranked: List[ScoredChunk]
        rerank_started = perf_counter()
        if settings.rerank_enabled and candidates:
            try:
                docs = [chunk.text for chunk in candidates]
                ranked = self.rerank_client.rerank(
                    model=settings.rerank_model_text,
                    query=query,
                    documents=docs,
                    top_n=min(len(docs), rerank_top_n),
                )
                reranked = []
                for idx, score in ranked:
                    if 0 <= idx < len(candidates):
                        reranked.append(ScoredChunk(chunk=candidates[idx], score=score))
                if not reranked:
                    raise RuntimeError("rerank returned no usable entries")
                reranked = reranked[:_TOP_K]
            except Exception as exc:  # noqa: BLE001
                warning = f"{warning}; rerank unavailable: {exc}; using lexical rerank fallback" if warning else (
                    f"rerank unavailable: {exc}; using lexical rerank fallback"
                )
                retrieval_mode = "lexical_fallback"
                hybrid = HybridRetriever(candidates)
                reranked = hybrid.retrieve(query, top_k=_TOP_K)
        else:
            retrieval_mode = "lexical_fallback"
            hybrid = HybridRetriever(candidates)
            reranked = hybrid.retrieve(query, top_k=_TOP_K)
        rerank_ms = (perf_counter() - rerank_started) * 1000.0

        qa_overlap = self._overlap_ratio(query, reranked, top_n=3)
        top1_score = reranked[0].score if reranked else 0.0
        return _RetrievalResult(
            evidence=reranked,
            retrieval_mode=retrieval_mode,
            warning=warning,
            top1_score=top1_score,
            qa_overlap=qa_overlap,
            vector_search_ms=vector_search_ms,
            lexical_broad_ms=lexical_broad_ms,
            rerank_ms=rerank_ms,
        )

    @staticmethod
    def _merge_candidates(
        vector_candidates: List[Chunk],
        lexical_candidates: List[Chunk],
        limit: int,
    ) -> List[Chunk]:
        merged: List[Chunk] = []
        seen: set[str] = set()
        for chunk in vector_candidates + lexical_candidates:
            cid = chunk.chunk_id
            if cid in seen:
                continue
            seen.add(cid)
            merged.append(chunk)
            if len(merged) >= max(1, limit):
                break
        return merged

    def _build_answer(self, question: str, evidence: List[ScoredChunk]) -> str:
        usage_question = self._is_usage_question(question)
        code_evidence, text_evidence = self._prepare_evidence_sections(evidence[:_PROMPT_EVIDENCE_LIMIT])
        has_code_evidence = bool(code_evidence.strip())
        evidence_summary = "code-first evidence available" if has_code_evidence else "text-only / parameter-style evidence"
        primary_language = self._preferred_language(code_evidence)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a TED documentation assistant.\n"
                    "Answer strictly from the provided evidence. Do not invent any API, class, function, "
                    "parameter, default value, return value, file path, or execution step.\n"
                    "Rules:\n"
                    "1. Read code evidence first, then text evidence. If code evidence is enough, explain around it.\n"
                    "2. For usage/example/call-pattern questions, if evidence is sufficient, include at least one "
                    "proper fenced code block.\n"
                    "3. Example code should reuse API and parameter names that already appear in evidence.\n"
                    f"4. Any placeholder that must be edited manually must be marked at the end of the same line with {_REPLACE_NOTE_TEXT}. "
                    "Use # for Python/Shell/YAML, // for JS/Java/C-like languages, and -- for SQL.\n"
                    "5. Prefer python fenced blocks unless evidence clearly points to bash/sql/javascript.\n"
                    f"6. If the user asks for an example but evidence is insufficient, explicitly say {_INSUFFICIENT_EXAMPLE_TEXT} "
                    "and explain what is missing. Do not fabricate.\n"
                    f"7. Only reply with {_ABSTAIN_TEXT} when the evidence cannot support any answer at all."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Usage/example question: {'yes' if usage_question else 'no'}\n"
                    f"Evidence summary: {evidence_summary}\n"
                    f"Preferred code language: {primary_language}\n\n"
                    f"Code evidence:\n{code_evidence or '(none)'}\n\n"
                    f"Text evidence:\n{text_evidence or '(none)'}\n\n"
                    "Produce the final answer in Chinese. If you include example code, placeholder items must be "
                    "marked at the end of the same line."
                ),
            },
        ]
        answer = self.chat_client.chat(
            model=settings.model_name,
            messages=messages,
            temperature=0.1,
        ).strip()
        answer = self._ensure_placeholder_markers(answer)

        if usage_question and not self._has_fenced_code(answer):
            answer = self._repair_answer_format(
                question=question,
                answer=answer,
                code_evidence=code_evidence,
                text_evidence=text_evidence,
                preferred_language=primary_language,
            )
            answer = self._ensure_placeholder_markers(answer)

        if usage_question and not self._has_fenced_code(answer):
            if _INSUFFICIENT_EXAMPLE_TEXT not in answer:
                answer = (
                    f"{answer}\n\n{_INSUFFICIENT_EXAMPLE_TEXT}。"
                    if answer
                    else f"{_INSUFFICIENT_EXAMPLE_TEXT}。"
                )
        return answer.strip()

    def _repair_answer_format(
        self,
        question: str,
        answer: str,
        code_evidence: str,
        text_evidence: str,
        preferred_language: str,
    ) -> str:
        repaired = self.chat_client.chat(
            model=settings.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are repairing answer format only. Do not change the question and do not add unsupported facts.\n"
                        "If evidence is sufficient, rewrite the answer so it includes a fenced code block.\n"
                        f"If evidence is insufficient, explicitly output {_INSUFFICIENT_EXAMPLE_TEXT} and keep the reason brief.\n"
                        f"Placeholder values inside code must be marked with {_REPLACE_NOTE_TEXT} at end of line."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Preferred code language: {preferred_language}\n\n"
                        f"Current answer:\n{answer or '(empty)'}\n\n"
                        f"Code evidence:\n{code_evidence or '(none)'}\n\n"
                        f"Text evidence:\n{text_evidence or '(none)'}"
                    ),
                },
            ],
            temperature=0.0,
        ).strip()
        return repaired or answer

    @staticmethod
    def _is_usage_question(question: str) -> bool:
        return bool(_USAGE_QUESTION_RE.search(question or ""))

    def _prepare_evidence_sections(self, evidence: List[ScoredChunk]) -> Tuple[str, str]:
        code_parts: List[str] = []
        text_parts: List[str] = []
        seen_code: set[str] = set()

        for item in evidence:
            source = item.chunk.source
            code_segments = self._extract_code_segments(item.chunk.text)
            for segment in code_segments[:2]:
                normalized = segment.strip()
                if not normalized or normalized in seen_code:
                    continue
                seen_code.add(normalized)
                language = self._infer_code_language(normalized)
                code_parts.append(
                    f"- source: {source}\n```{language}\n{self._trim_text(normalized, _CODE_SEGMENT_CHAR_LIMIT)}\n```"
                )
            text_parts.append(
                f"- source: {source}\n  snippet: {self._trim_text(self._collapse_whitespace(item.chunk.text), _TEXT_SEGMENT_CHAR_LIMIT)}"
            )
        return "\n\n".join(code_parts), "\n".join(text_parts)

    @staticmethod
    def _trim_text(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _normalize_newlines(text: str) -> str:
        return (text or "").replace("\r\n", "\n").replace("\r", "\n")

    @classmethod
    def _collapse_whitespace(cls, text: str) -> str:
        lines = [line.strip() for line in cls._normalize_newlines(text).split("\n") if line.strip()]
        return " ".join(lines)

    @classmethod
    def _extract_code_segments(cls, text: str) -> List[str]:
        raw = cls._normalize_newlines(text)
        segments: List[str] = []
        for match in _CODE_FENCE_RE.finditer(raw):
            code = match.group("code").strip()
            if code:
                segments.append(code)
        for paragraph in cls._split_paragraphs(raw):
            if cls._is_code_paragraph(paragraph):
                segments.append("\n".join(paragraph).strip())

        deduped: List[str] = []
        seen: set[str] = set()
        for segment in segments:
            normalized = segment.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    @classmethod
    def _split_paragraphs(cls, text: str) -> List[List[str]]:
        paragraphs: List[List[str]] = []
        for chunk in re.split(r"\n\s*\n+", cls._normalize_newlines(text)):
            lines = [line.rstrip() for line in chunk.split("\n") if line.strip()]
            if lines:
                paragraphs.append(lines)
        return paragraphs

    @classmethod
    def _is_code_paragraph(cls, lines: List[str]) -> bool:
        code_scores = [cls._line_code_score(line) for line in lines]
        code_like_count = sum(score >= 2 for score in code_scores)
        strong_count = sum(score >= 3 for score in code_scores)
        return (
            code_like_count >= 2
            and (
                strong_count >= 1
                or code_like_count == len(lines)
                or code_like_count / max(1, len(lines)) >= 0.6
            )
        )

    @staticmethod
    def _line_code_score(line: str) -> int:
        stripped = line.strip()
        if not stripped:
            return 0

        score = 0
        lowered = stripped.lower()
        if line.startswith(("    ", "\t")):
            score += 1
        if stripped.startswith(("```", ">>>", "$ ")):
            score += 3
        if re.match(r"^(def|class|import|from|return|if|elif|else:|for|while|try:|except|finally:|with|@)\b", lowered):
            score += 3
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*.+$", stripped):
            score += 2
        if re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(", stripped):
            score += 2
        if re.search(r"[{}[\];]|=>|->|::", stripped):
            score += 1
        if re.search(r"\b(curl|python|pytest|pip|bash)\b", lowered):
            score += 1
        if stripped.startswith(("#", "//")):
            score += 1
        return score

    @classmethod
    def _preferred_language(cls, code_evidence: str) -> str:
        if not code_evidence.strip():
            return "python"
        return cls._infer_code_language(code_evidence)

    @staticmethod
    def _infer_code_language(text: str) -> str:
        lowered = text.lower()
        if re.search(r"^\s*(curl|export|python3?|pip|bash)\b", lowered, flags=re.MULTILINE) or "$ " in text:
            return "bash"
        if re.search(r"\b(select|insert|update|delete|from|where)\b", lowered):
            return "sql"
        if re.search(r"\b(const|let|function|console\.log|=>)\b", lowered):
            return "javascript"
        return "python"

    @staticmethod
    def _has_fenced_code(text: str) -> bool:
        return bool(_CODE_FENCE_RE.search(text or ""))

    @classmethod
    def _ensure_placeholder_markers(cls, answer: str) -> str:
        if not answer.strip():
            return answer

        def replace_block(match: re.Match[str]) -> str:
            language = (match.group("lang") or "").strip() or cls._infer_code_language(match.group("code"))
            code = cls._annotate_code_placeholders(match.group("code"), language)
            return f"```{language}\n{code}\n```"

        return _CODE_FENCE_RE.sub(replace_block, answer)

    @classmethod
    def _annotate_code_placeholders(cls, code: str, language: str) -> str:
        comment_marker = cls._comment_marker_for_language(language)
        lines = code.split("\n")
        updated: List[str] = []
        for line in lines:
            if cls._line_has_placeholder(line) and _REPLACE_NOTE_TEXT not in line:
                suffix = f" {comment_marker} {_REPLACE_NOTE_TEXT}"
                updated.append(f"{line.rstrip()}{suffix}")
            else:
                updated.append(line.rstrip())
        return "\n".join(updated).rstrip()

    @staticmethod
    def _comment_marker_for_language(language: str) -> str:
        lowered = (language or "").strip().lower()
        if lowered in {"js", "javascript", "ts", "typescript", "java", "c", "cpp", "c++", "go", "rust"}:
            return "//"
        if lowered == "sql":
            return "--"
        return "#"

    @staticmethod
    def _line_has_placeholder(line: str) -> bool:
        content = line.strip()
        if not content or content.startswith(("#", "//", "--")):
            return False
        return any(pattern.search(content) for pattern in _PLACEHOLDER_PATTERNS)

    @staticmethod
    def _tokens(text: str) -> List[str]:
        lowered = text.lower()
        tokens: List[str] = []
        tokens.extend(token for token in _EN_TOKEN_RE.findall(lowered) if len(token) >= 2)
        zh_chars = [ch for ch in lowered if _ZH_CHAR_RE.fullmatch(ch)]
        if len(zh_chars) >= 2:
            tokens.extend(
                "".join(zh_chars[idx: idx + 2])
                for idx in range(len(zh_chars) - 1)
            )
        return tokens

    @classmethod
    def _overlap_ratio(cls, query: str, evidence: List[ScoredChunk], top_n: int = 3) -> float:
        q_tokens = set(cls._tokens(query))
        if not q_tokens:
            return 0.0
        e_tokens: Set[str] = set()
        for item in evidence[:top_n]:
            e_tokens.update(cls._tokens(item.chunk.text))
        hit = len(q_tokens.intersection(e_tokens))
        return hit / max(1, len(q_tokens))

    @staticmethod
    def _should_abstain(top1_score: float, qa_overlap: float) -> bool:
        return qa_overlap < 0.15 or (top1_score < 0.70 and qa_overlap < 0.30)
