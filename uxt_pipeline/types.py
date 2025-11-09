from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class Element(BaseModel):
    type: str
    text: str
    page_number: Optional[int] = None
    level: Optional[int] = None
    section_path: Optional[List[str]] = None

class Chunk(BaseModel):
    id: str
    text: str
    section_path: Optional[List[str]] = None
    meta: Dict = Field(default_factory=dict)
