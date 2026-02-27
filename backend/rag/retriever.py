from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List
import math
import re

from .indexer import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class HybridRetriever:
    """Two-stage retriever: broad retrieval + focused rerank."""

    def __init__(self, chunks: Iterable[Chunk]):
        self.chunks = list(chunks)
        self._tokenized = [self._tokens(c.text) for c in self.chunks]
        self._idf = self._build_idf(self._tokenized)

    def retrieve(self, query: str, top_k: int = 6) -> List[ScoredChunk]:
        q_tokens = self._tokens(query)
        broad = self._broad_retrieval(q_tokens, top_n=max(30, top_k * 5))
        reranked = self._focused_rerank(query, broad)
        return reranked[:top_k]

    def _broad_retrieval(self, q_tokens: List[str], top_n: int) -> List[ScoredChunk]:
        results: List[ScoredChunk] = []
        q_set = set(q_tokens)
        for chunk, tokens in zip(self.chunks, self._tokenized):
            bm25_like = self._bm25_like(q_tokens, tokens)
            overlap = len(q_set.intersection(tokens)) / max(1, len(q_set))
            score = bm25_like * 0.7 + overlap * 0.3
            results.append(ScoredChunk(chunk=chunk, score=score))
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_n]

    def _focused_rerank(self, query: str, candidates: List[ScoredChunk]) -> List[ScoredChunk]:
        query_lower = query.lower()
        weighted: List[ScoredChunk] = []
        for item in candidates:
            bonus = 0.0
            text_l = item.chunk.text.lower()
            for kw in ("bandwidth", "sfdr", "loop gain", "ac", "fft", "pyted", "api"):
                if kw in query_lower and kw in text_l:
                    bonus += 0.08
            weighted.append(ScoredChunk(chunk=item.chunk, score=item.score + bonus))
        weighted.sort(key=lambda x: x.score, reverse=True)
        return weighted

    def _bm25_like(self, q_tokens: List[str], d_tokens: List[str]) -> float:
        if not d_tokens:
            return 0.0
        tf: Dict[str, int] = {}
        for t in d_tokens:
            tf[t] = tf.get(t, 0) + 1

        score = 0.0
        doc_len = len(d_tokens)
        avgdl = sum(len(t) for t in self._tokenized) / max(1, len(self._tokenized))
        k1 = 1.2
        b = 0.75
        for t in q_tokens:
            idf = self._idf.get(t, 0.0)
            f = tf.get(t, 0)
            denom = f + k1 * (1 - b + b * doc_len / max(1.0, avgdl))
            if denom == 0:
                continue
            score += idf * (f * (k1 + 1)) / denom
        return score

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z_]{2,}|[\u4e00-\u9fff]{1,}", text.lower())

    @staticmethod
    def _build_idf(tokenized_docs: List[List[str]]) -> Dict[str, float]:
        n_docs = len(tokenized_docs)
        df: Dict[str, int] = {}
        for tokens in tokenized_docs:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        return {t: math.log((n_docs - c + 0.5) / (c + 0.5) + 1) for t, c in df.items()}
