"""ContextOS Schema - CoALA Memory Types"""

import time
import uuid
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class MemoryType(str, Enum):
    """CoALA memory classification types."""
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    WORKING = "working"


class ContextNode(BaseModel):
    """A single memory node in the context graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str
    type: MemoryType
    created_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)
    access_count: int = 1
    decay_factor: float = 0.99
    importance_score: float = 0.0
    source_id: Optional[str] = None


class ContextEdge(BaseModel):
    """A relationship between two memory nodes."""
    source: str
    target: str
    relation: str
    weight: float = 1.0