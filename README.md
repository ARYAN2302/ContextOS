# ContextOS

**A Graph-Theoretic Memory Kernel for Agentic AI Systems**

> *"Beyond RAG: Stateful memory for AI agents that actually remembers."*

[![PyPI version](https://badge.fury.io/py/agentic-memory.svg)](https://pypi.org/project/agentic-memory/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is ContextOS?

ContextOS is a **framework for building AI agents with persistent, structured memory**. Unlike standard RAG (Retrieval-Augmented Generation) which treats documents as flat vectors, ContextOS models memory as a **Knowledge Graph** where:

- **Nodes** are memories (semantic facts, episodic events, procedural rules)
- **Edges** encode relationships (temporal, causal, associative)
- **Retrieval** uses hybrid scoring: `PageRank centrality + Vector similarity`

This enables **multi-hop reasoning** that pure vector search cannot achieve.

---

## ✨ Key Features

- 🧠 **CoALA Memory Architecture** - Semantic, Episodic, and Procedural memory types
- 🔗 **Graph-Native Storage** - NetworkX topology + ChromaDB vectors
- ⚡ **Hybrid Retrieval** - `Relevance = (α × Vector) + (β × Graph)`
- 💾 **Persistent by Default** - Memory survives restarts
- 🔌 **Framework Agnostic** - Works with LangChain, LlamaIndex, or raw API calls

---

## Installation

```bash
pip install agentic-memory
```

Or for local development:

```bash
git clone https://github.com/ARYAN2302/ContextOS.git
cd ContextOS
pip install -e .
```

---

## Quick Start

### Simple API (Recommended)

```python
from agentic_memory import ContextClient, MemoryType

# Initialize (loads existing memory if available)
client = ContextClient()

# Add memories
client.add_memory("User prefers dark mode", MemoryType.SEMANTIC)
client.add_memory("User asked about Python yesterday", MemoryType.EPISODIC)

# Compile context for a query
context = client.compile("What are the user's preferences?")
print(context)
```

### Full Chat Loop

```python
from agentic_memory import ContextClient

client = ContextClient()

def my_llm(system_prompt: str, user_query: str) -> str:
    # Your LLM call here (OpenAI, Groq, Anthropic, etc.)
    return llm.invoke(system_prompt + user_query)

# Run a full RAG loop with automatic memory logging
response = client.chat("What should I work on today?", llm_callable=my_llm)
```

### Low-Level API

```python
from agentic_memory import ContextGraph, ContextNode, ContextEdge, MemoryType, ContextCompiler

# Direct graph access
kernel = ContextGraph()
node = ContextNode(content="Important fact", type=MemoryType.SEMANTIC)
kernel.add_node(node)

# Add relationships
edge = ContextEdge(source=node1.id, target=node2.id, relation="CAUSES")
kernel.add_edge(edge)

# Compile context
compiler = ContextCompiler(kernel)
context = compiler.compile("query", token_budget=500, alpha=50, beta=50)
```

---

## Benchmarks

### Needle-in-a-Haystack (NIAH)

Does the system remember a specific fact hidden among 100+ distractor memories?

- **Vector RAG**: Recall degrades as context grows (Lost-in-the-Middle problem)
- **ContextOS**: Maintains **100% Recall** via graph topology

```bash
cd experiments && python niah_benchmark.py
```

### Ablation Study (Multi-Hop Reasoning)

Can the system answer questions requiring multiple inference steps?

| Configuration | Multi-Hop Accuracy | Analysis |
|--------------|-------------------|----------|
| Vector Only (RAG) | 50% | Found first hop, missed connections |
| Graph Only | 50% | Failed to ground initial query |
| **ContextOS (Hybrid)** | **100%** | Anchored via Vector, traversed via Graph |

---

## Architecture

```
agentic_memory/
├── client.py           # ContextClient - main entry point
├── core/
│   ├── schema.py       # Pydantic models (ContextNode, ContextEdge, MemoryType)
│   └── graph.py        # Hybrid storage (NetworkX + ChromaDB)
├── memory/
│   ├── ingestor.py     # LLM-powered memory classification
│   └── compiler.py     # PageRank + Vector hybrid retrieval
└── utils/
    └── text.py         # Text processing utilities
```

### The Hybrid Scoring Formula

```
relevance(node, query) = (α × semantic_similarity) + (β × pagerank_centrality × time_decay)
```

- **α (alpha)**: Weight for semantic similarity (vector search)
- **β (beta)**: Weight for graph centrality (structural importance)
- **time_decay**: Recency factor for episodic memories

---

## Configuration

```python
client = ContextClient(
    storage_path="my_memory.json",    # Graph persistence
    chroma_path="my_vectors/",         # Vector store
    auto_persist=True                  # Save on every change
)

# Retrieval tuning
context = client.compile(
    query="...",
    token_budget=1000,    # Max tokens in context
    alpha=50.0,           # Vector weight
    beta=50.0             # Graph weight
)
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Inspired by the [CoALA](https://arxiv.org/abs/2309.02427) architecture for cognitive agents.
