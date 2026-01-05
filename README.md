# ContextOS

**A Graph-Theoretic Memory Kernel for Agentic AI Systems**

> *"Beyond RAG: Stateful memory for AI agents that actually remembers."*

[![PyPI version](https://badge.fury.io/py/agentic-memory.svg)](https://pypi.org/project/agentic-memory/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/ContextOS_Paper.pdf)

---

## 🎯 Quick Demo

Launch the interactive dashboard and explore the memory graph:

```bash
# Clone and setup
git clone https://github.com/ARYAN2302/ContextOS.git
cd ContextOS

# Install dependencies
pip install -r requirements.txt

# Launch the demo dashboard
streamlit run examples/app.py
```

Then open **http://localhost:8501** and click **"Load Demo Brain"** to see the 3D memory visualization!

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

## 🎮 Dashboard Demo Features

The Streamlit dashboard includes:

### Interactive Memory Graph
- **3D Brain Visualization** - Drag, zoom, and explore nodes
- **Color-Coded Nodes** - Cyan (Semantic), Orange (Episodic), Green (Procedural)
- **Size by PageRank** - Important memories appear larger
- **Hover Details** - See memory content and metadata

### Real-time Metrics
- **Total Memories** - Count of stored memories
- **Connections** - Relationship edges in the graph
- **Vector Index** - Semantic search index size
- **Graph Density** - Connectivity measure

### Demo Mode
- **One-Click Brain Load** - Generate 60+ memories instantly
- **Needle in Haystack Demo** - See how important facts become graph hubs
- **Chat Interface** - Add memories and query context

### Benchmark Showcase
- **15x Memory Boost** - Llama-8B + ContextOS vs alone
- **100% vs 6.7%** - Accuracy comparison
- **Multi-Hop Reasoning** - Graph traversal advantages

---

## Installation

```bash
pip install agentic-memory
```

Or for local development:

```bash
git clone https://github.com/ARYAN2302/ContextOS.git
cd ContextOS
pip install -r requirements.txt
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

### Memory Boost Benchmark (Core Thesis)

**Does memory make small LLMs useful?**

| Setting | Accuracy |
|---------|----------|
| **Llama-8B + ContextOS** | **100%** |
| Llama-8B alone (stateless) | 6.7% |

> **15x improvement** - Proves that SLM + structured memory >> SLM alone.

```bash
cd experiments && python memory_boost_benchmark.py
```

### HotpotQA (Multi-Hop Reasoning)

Real HotpotQA dev set with 10 paragraphs per question (2 relevant + 8 distractors).

| Method | Exact Match | F1 Score |
|--------|-------------|----------|
| **ContextOS** | **54.0%** | **67.7%** |
| Vector-only RAG | 48.0% | 64.3% |
| No retrieval (stateless) | 0.0% | 10.8% |

> **+3.4% F1** over pure vector RAG. Graph structure helps filter noisy distractors.

```bash
cd experiments && python hotpotqa_real_benchmark.py
```

### Ablation Study

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

## Running the Dashboard

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Streamlit dashboard
streamlit run examples/app.py
```

**Dashboard Features:**
- 🧠 Interactive 3D memory graph visualization
- 📊 Real-time statistics and metrics
- 🚀 One-click demo brain generation
- 💬 Chat interface with memory persistence
- 📈 Benchmark comparison charts

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

## Roadmap & Future Work v0.2 (Planned):

Multi-Session Persistence: Long-term user profiles and cross-session entity resolution.

Memory Consolidation (Sleep Cycles): Background merging of duplicate nodes and pruning of stale memories to optimize graph density.

v0.3 (Planned):

Causal Reasoning Engine: New edge types (CAUSES, PREVENTS) for deeper logic.

Document Ingestion: Parsing full PDFs into knowledge sub-graphs.

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Inspired by the [CoALA](https://arxiv.org/abs/2309.02427) architecture for cognitive agents.

---

## 👤 Author

**Aryan** - [GitHub](https://github.com/ARYAN2302)

---

**Star ⭐ the repo if you find it useful!**
