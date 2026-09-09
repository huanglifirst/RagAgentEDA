from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class RunTaskRequest(BaseModel):
    query: str = Field(..., description="Natural language test requirement")
    circuit_description: Optional[str] = Field(default=None)
    top_k: int = Field(default=6, ge=1, le=20)
    execute: bool = Field(default=True, description="Whether to execute the generated script")


class EvidenceItem(BaseModel):
    source: str
    score: float
    snippet: str


class RagAskRequest(BaseModel):
    question: str = Field(..., description="Natural language question for RAG QA")


class QueryRewriteRequest(BaseModel):
    query: str = Field(..., description="Natural language query to rewrite for TED retrieval")
    scene: Literal["qa", "task"] = Field(default="qa", description="Rewrite scene")
    mode: str = Field(default="conservative", description="Rewrite mode: conservative or aggressive")


class RagAskResponse(BaseModel):
    status: str
    answer: str
    evidence: List[EvidenceItem] = Field(default_factory=list)
    warning: str = ""


class QueryRewriteResponse(BaseModel):
    original_query: str
    rewritten_query: str
    changed: bool
    strategy: str
    warning: str = ""


class RunTaskResponse(BaseModel):
    task_id: str
    status: str
    generated_code: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    logs: Dict[str, str] = Field(default_factory=dict)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    error: Optional[str] = None
