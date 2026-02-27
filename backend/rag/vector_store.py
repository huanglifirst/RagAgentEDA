from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import json
import math

from backend.llm.client import OpenAICompatClient
from backend.rag.indexer import Chunk


@dataclass
class VectorHit:
    chunk: Chunk
    score: float


class PersistentEmbeddingIndex:
    """Persist embeddings to disk so Resource is vectorized once and reused."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def load(self, fingerprint: str) -> Tuple[List[Chunk], List[List[float]]] | None:
        path = self.index_dir / f'{fingerprint}.json'
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding='utf-8'))
        chunks = [Chunk(**c) for c in data['chunks']]
        vectors = data['vectors']
        return chunks, vectors

    def save(self, fingerprint: str, chunks: List[Chunk], vectors: List[List[float]]) -> Path:
        path = self.index_dir / f'{fingerprint}.json'
        payload = {
            'fingerprint': fingerprint,
            'chunk_count': len(chunks),
            'vector_dim': len(vectors[0]) if vectors else 0,
            'chunks': [c.__dict__ for c in chunks],
            'vectors': vectors,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        latest = self.index_dir / 'LATEST'
        latest.write_text(path.name, encoding='utf-8')
        return path


class EmbeddingRetriever:
    def __init__(self, client: OpenAICompatClient, model: str, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        self.client = client
        self.model = model
        self.chunks = chunks
        self._vectors = vectors

    @classmethod
    def from_chunks(
        cls,
        client: OpenAICompatClient,
        model: str,
        chunks: List[Chunk],
        store: PersistentEmbeddingIndex,
        fingerprint: str,
    ) -> 'EmbeddingRetriever':
        loaded = store.load(fingerprint)
        if loaded is not None:
            cached_chunks, cached_vectors = loaded
            return cls(client, model, cached_chunks, cached_vectors)

        vectors = client.embed(model, [c.text[:3000] for c in chunks]) if chunks else []
        store.save(fingerprint, chunks, vectors)
        return cls(client, model, chunks, vectors)

    def search(self, query: str, top_k: int = 20) -> List[VectorHit]:
        if not self.chunks:
            return []
        qv = self.client.embed(self.model, [query])[0]
        scored: List[VectorHit] = []
        for c, v in zip(self.chunks, self._vectors):
            scored.append(VectorHit(chunk=c, score=self._cosine(qv, v)))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
