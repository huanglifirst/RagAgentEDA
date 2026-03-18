from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple
import json
import math
import re
import time

from backend.config import settings
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

    def load(
        self,
        fingerprint: str,
        expected_model: str | None = None,
        expected_api_base: str | None = None,
    ) -> Tuple[List[Chunk], List[List[float]]] | None:
        path = self.index_dir / f'{fingerprint}.json'
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding='utf-8'))
        if not self._is_payload_valid(data):
            return None
        if expected_model and data.get('embedding_model') != expected_model:
            return None
        if expected_api_base and data.get('embedding_api_base') != expected_api_base:
            return None

        chunks = [Chunk(**c) for c in data['chunks']]
        vectors = data['vectors']
        return chunks, vectors

    def save(
        self,
        fingerprint: str,
        chunks: List[Chunk],
        vectors: List[List[float]],
        embedding_model: str,
        embedding_api_base: str,
    ) -> Path:
        self._validate_vectors(chunks, vectors)
        path = self.index_dir / f'{fingerprint}.json'
        payload = {
            'fingerprint': fingerprint,
            'chunk_count': len(chunks),
            'vector_dim': len(vectors[0]) if vectors else 0,
            'chunks': [c.__dict__ for c in chunks],
            'vectors': vectors,
            'embedding_model': embedding_model,
            'embedding_api_base': embedding_api_base,
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        latest = self.index_dir / 'LATEST'
        latest.write_text(path.name, encoding='utf-8')
        return path

    @staticmethod
    def _is_payload_valid(payload: dict[str, Any]) -> bool:
        chunks_raw = payload.get('chunks')
        vectors_raw = payload.get('vectors')
        if not isinstance(chunks_raw, list) or not isinstance(vectors_raw, list):
            return False
        try:
            chunks = [Chunk(**c) for c in chunks_raw]
        except Exception:
            return False
        try:
            PersistentEmbeddingIndex._validate_vectors(chunks, vectors_raw)
        except Exception:
            return False
        return True

    @staticmethod
    def _validate_vectors(chunks: List[Chunk], vectors: List[List[float]]) -> None:
        if not vectors:
            raise RuntimeError('embedding vectors are empty')
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f'embedding vector count mismatch: vectors={len(vectors)} chunks={len(chunks)}'
            )

        expected_dim: int | None = None
        for idx, vector in enumerate(vectors):
            if not isinstance(vector, list) or not vector:
                raise RuntimeError(f'embedding vector[{idx}] is empty or invalid')
            for value in vector:
                if not isinstance(value, (int, float)):
                    raise RuntimeError(f'embedding vector[{idx}] contains non-numeric value')
            dim = len(vector)
            if expected_dim is None:
                expected_dim = dim
                if expected_dim <= 0:
                    raise RuntimeError('embedding vector dimension must be > 0')
            elif dim != expected_dim:
                raise RuntimeError(
                    f'inconsistent embedding vector dimensions: expected={expected_dim}, got={dim}'
                )


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
        force_rebuild: bool = False,
    ) -> 'EmbeddingRetriever':
        loaded = None if force_rebuild else store.load(
            fingerprint,
            expected_model=model,
            expected_api_base=client.base_url,
        )
        if loaded is not None:
            cached_chunks, cached_vectors = loaded
            return cls(client, model, cached_chunks, cached_vectors)

        vectors = cls._embed_in_batches(client, model, chunks) if chunks else []
        store.save(
            fingerprint,
            chunks,
            vectors,
            embedding_model=model,
            embedding_api_base=client.base_url,
        )
        return cls(client, model, chunks, vectors)

    def search(self, query: str, top_k: int = 20) -> List[VectorHit]:
        if not self.chunks:
            return []
        if not self._vectors:
            raise RuntimeError('embedding index contains no vectors')
        if len(self._vectors) != len(self.chunks):
            raise RuntimeError(
                f'embedding index mismatch: vectors={len(self._vectors)} chunks={len(self.chunks)}'
            )
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

    @classmethod
    def _embed_in_batches(
        cls,
        client: OpenAICompatClient,
        model: str,
        chunks: List[Chunk],
    ) -> List[List[float]]:
        texts = [c.text[:3000] for c in chunks]
        batch_size = max(1, settings.embedding_batch_size)
        max_retries = max(1, settings.embedding_retry_count)
        base_delay = max(0.1, settings.embedding_retry_delay_sec)

        all_vectors: List[List[float]] = []
        start = 0
        while start < len(texts):
            current_batch_size = min(batch_size, len(texts) - start)
            batch = texts[start:start + current_batch_size]
            last_exc: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    vectors = client.embed(model, batch)
                    if len(vectors) != len(batch):
                        raise RuntimeError(
                            f'embedding batch size mismatch: input={len(batch)} output={len(vectors)} '
                            f'(batch_start={start})'
                        )
                    all_vectors.extend(vectors)
                    last_exc = None
                    start += len(batch)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    limited = cls._extract_batch_limit(str(exc))
                    if limited is not None and limited > 0 and limited < len(batch):
                        batch_size = limited
                        current_batch_size = min(batch_size, len(texts) - start)
                        batch = texts[start:start + current_batch_size]
                        continue
                    if not cls._is_retryable_embed_error(str(exc)) or attempt >= max_retries:
                        raise RuntimeError(
                            f'embedding batch failed after {attempt} attempt(s), '
                            f'batch_start={start}, batch_size={len(batch)}: {exc}'
                        ) from exc
                    time.sleep(base_delay * (2 ** (attempt - 1)))

            if last_exc is not None:
                raise RuntimeError(
                    f'embedding batch failed permanently, batch_start={start}, batch_size={len(batch)}: {last_exc}'
                )
        return all_vectors

    @staticmethod
    def _is_retryable_embed_error(message: str) -> bool:
        lowered = message.lower()
        retry_markers = (
            'service load is too high',
            'badrequest.toolarge',
            'too many requests',
            'rate limit',
            'timed out',
            'timeout',
            'temporarily unavailable',
            'api urlerror',
        )
        return any(marker in lowered for marker in retry_markers)

    @staticmethod
    def _extract_batch_limit(message: str) -> int | None:
        # Example: "... batch size is invalid, it should not be larger than 10 ..."
        match = re.search(r'not be larger than\s+(\d+)', message, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None
