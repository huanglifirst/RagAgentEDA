from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, List
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
        return [item['embedding'] for item in data.get('data', [])]

    def chat(self, model: str, messages: list[dict[str, str]], temperature: float = 0.1) -> str:
        payload = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'stream': False,
        }
        data = self._post('/chat/completions', payload)
        return data['choices'][0]['message']['content']

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
