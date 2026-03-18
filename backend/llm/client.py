from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class OpenAICompatClient:
    """OpenAI-compatible client for Volcano Ark (`/api/v3`) endpoints."""

    base_url: str
    api_key: str

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        payload = {'model': model, 'input': texts}
        data = self._post('/embeddings', payload)
        items = data.get('data')
        if not isinstance(items, list) or not items:
            raise RuntimeError(
                f'Embedding API returned empty or invalid data for model={model}, '
                f'base={self.base_url}: {self._brief(data)}'
            )

        vectors: List[List[float]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise RuntimeError(f'Embedding API data[{idx}] is not an object: {self._brief(item)}')
            embedding = item.get('embedding')
            if not isinstance(embedding, list) or not embedding:
                raise RuntimeError(f'Embedding API data[{idx}].embedding is empty or invalid')

            vec: List[float] = []
            for jdx, value in enumerate(embedding):
                if not isinstance(value, (int, float)):
                    raise RuntimeError(
                        f'Embedding API data[{idx}].embedding[{jdx}] is non-numeric: {self._brief(value)}'
                    )
                vec.append(float(value))
            vectors.append(vec)
        return vectors

    def chat(self, model: str, messages: list[dict[str, str]], temperature: float = 0.1) -> str:
        payload = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'stream': False,
        }
        data = self._post('/chat/completions', payload)
        return data['choices'][0]['message']['content']

    def rerank(self, model: str, query: str, documents: List[str], top_n: int) -> List[Tuple[int, float]]:
        payload = {
            'model': model,
            'query': query,
            'documents': documents,
            'top_n': top_n,
        }
        data = self._post('/rerank', payload)
        items = data.get('results')
        if not isinstance(items, list) or not items:
            items = data.get('data')
        if not isinstance(items, list) or not items:
            raise RuntimeError(
                f'Rerank API returned empty or invalid data for model={model}, '
                f'base={self.base_url}: {self._brief(data)}'
            )

        ranked: List[Tuple[int, float]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise RuntimeError(f'Rerank API item[{idx}] is not an object: {self._brief(item)}')
            doc_index = item.get('index', item.get('document_index'))
            score = item.get('relevance_score', item.get('score'))
            if not isinstance(doc_index, int):
                raise RuntimeError(f'Rerank API item[{idx}] missing integer index: {self._brief(item)}')
            if not isinstance(score, (int, float)):
                raise RuntimeError(f'Rerank API item[{idx}] missing numeric score: {self._brief(item)}')
            ranked.append((doc_index, float(score)))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:max(1, top_n)]

    def list_models(self) -> dict[str, Any]:
        return self._get('/models')

    def _get(self, path: str) -> dict[str, Any]:
        url = self.base_url.rstrip('/') + path
        req = Request(
            url,
            method='GET',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
            },
        )
        return self._request(req)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url.rstrip('/') + path
        req = Request(
            url,
            method='POST',
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
            },
        )
        return self._request(req)

    @staticmethod
    def _request(req: Request) -> dict[str, Any]:
        try:
            with urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except HTTPError as exc:
            body = exc.read().decode('utf-8', errors='ignore')
            raise RuntimeError(f'API HTTPError {exc.code}: {body}') from exc
        except URLError as exc:
            raise RuntimeError(f'API URLError: {exc.reason}') from exc

    @staticmethod
    def _brief(value: Any, max_len: int = 280) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except Exception:
            text = str(value)
        if len(text) <= max_len:
            return text
        return text[:max_len] + '...'
