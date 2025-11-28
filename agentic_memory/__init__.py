"""
Agentic Memory - A Graph-Theoretic Memory Kernel for AI Agents

Framework for building AI agents with persistent, structured memory
using hybrid graph topology and vector similarity retrieval.
"""

from .client import ContextClient
from .core.schema import ContextNode, ContextEdge, MemoryType
from .core.graph import ContextGraph
from .memory.ingestor import Ingestor
from .memory.compiler import ContextCompiler

__version__ = "0.1.2"
__all__ = [
    "ContextClient",
    "ContextNode",
    "ContextEdge",
    "MemoryType",
    "ContextGraph",
    "Ingestor",
    "ContextCompiler",
]