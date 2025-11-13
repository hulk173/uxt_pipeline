from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

class Chunk(BaseModel):
    id: str
    doc_id: str
    chunk_id: int
    type: str
    text: str
    meta: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SearchResult(BaseModel):
    score: float
    chunk: Chunk

class IngestJob(BaseModel):
    id: str
    status: str
    input_paths: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
