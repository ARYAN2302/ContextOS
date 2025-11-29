#!/usr/bin/env python3
"""
ContextOS Brain Visualizer
==========================

Generates an interactive 3D visualization of the memory graph.
Demonstrates the CoALA architecture with:
- Semantic Nodes (Cyan) - Stable facts, high PageRank
- Episodic Nodes (Orange) - Transient events, decaying
- Procedural Nodes (Green) - Action rules

This is the "Needle in a Haystack" demo - can the system find
the important memories among 50+ distractors?

Usage:
    python examples/visualize_brain.py
    
Then open brain.html in your browser.
"""

import sys
import os
import random
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_memory import ContextClient, MemoryType
from agentic_memory.core.schema import ContextEdge


def generate_demo_brain():
    """Creates a rich memory graph for visualization."""
    
    print("=" * 60)
    print("🧠 CONTEXTOS BRAIN GENERATOR")
    print("=" * 60)
    
    # Fresh client for demo (clean slate)
    client = ContextClient(
        storage_path="demo_brain.json",
        chroma_path="demo_brain_vectors"
    )
    
    # =========================================
    # CORE SEMANTIC NODES (The "Needles")
    # These are the important facts - they should
    # become hubs with high PageRank
    # =========================================
    
    print("\n📌 Injecting SEMANTIC nodes (stable facts)...")
    
    semantic_facts = [
        "User's name is Aryan Thakur",
        "Aryan is building ContextOS, a graph-theoretic memory system",
        "ContextOS uses hybrid retrieval: PageRank + Vector similarity",
        "The system is inspired by CoALA cognitive architecture",
        "User prefers Python over JavaScript",
        "User is applying for AI/ML internships",
        "ContextOS is published on PyPI as 'agentic-memory'",
        "The memory equation is S(t) = S₀·e^(-λt) + β·C(n)",
    ]
    
    semantic_ids = []
    for fact in semantic_facts:
        node_id = client.add_memory(fact, MemoryType.SEMANTIC)
        semantic_ids.append(node_id)
        print(f"   ✅ {fact[:50]}...")
    
    # =========================================
    # EPISODIC NODES (The "Haystack")
    # These are transient chat events - they should
    # cluster around the semantic anchors
    # =========================================
    
    print("\n💬 Injecting EPISODIC nodes (chat events)...")
    
    # Chat history simulation - 50 messages
    episodic_events = [
        # Related to ContextOS
        "User asked: How does PageRank work in memory systems?",
        "User said: I want to visualize the memory graph",
        "User asked: What's the difference between RAG and ContextOS?",
        "User said: The benchmark shows 100% vs 6.7% accuracy",
        "User asked: Can we add a decay function to memories?",
        "User said: I need to prepare for my internship interview",
        "User asked: How do I publish to PyPI?",
        "User said: Let's run the HotpotQA benchmark",
        
        # Random distractors (noise)
        "User said: What's the weather like?",
        "User asked: Can you help me with my homework?",
        "User said: I'm feeling tired today",
        "User asked: What time is it?",
        "User said: Let's take a break",
        "User asked: Do you know any good movies?",
        "User said: I had coffee this morning",
        "User asked: What's 2 + 2?",
        "User said: Hello, how are you?",
        "User asked: Can you tell me a joke?",
        "User said: I like pizza",
        "User asked: What's your favorite color?",
        "User said: The sky is blue",
        "User asked: Is it going to rain?",
        "User said: I went for a walk yesterday",
        "User asked: What's the capital of France?",
        "User said: I'm learning machine learning",
        "User asked: How do neural networks work?",
        "User said: Transformers are cool",
        "User asked: What is attention mechanism?",
        "User said: GPT-4 is impressive",
        "User asked: Can AI be conscious?",
        "User said: I read a paper on LLMs",
        "User asked: What's the difference between AI and ML?",
        "User said: I'm building a chatbot",
        "User asked: How do embeddings work?",
        "User said: Vector databases are useful",
        "User asked: What is ChromaDB?",
        "User said: NetworkX is great for graphs",
        "User asked: How does graph traversal work?",
        "User said: BFS vs DFS comparison",
        "User asked: What is Dijkstra's algorithm?",
        "User said: I love algorithms",
        "User asked: What's Big O notation?",
        "User said: Time complexity matters",
        "User asked: How to optimize code?",
        "User said: Clean code is important",
        "User asked: What's test-driven development?",
        "User said: I should write more tests",
        "User asked: How to debug effectively?",
        "User said: Print statements are my friends",
    ]
    
    episodic_ids = []
    for event in episodic_events:
        node_id = client.add_memory(event, MemoryType.EPISODIC)
        episodic_ids.append(node_id)
    
    print(f"   ✅ Added {len(episodic_events)} episodic events")
    
    # =========================================
    # PROCEDURAL NODES (Action Rules)
    # =========================================
    
    print("\n⚙️  Injecting PROCEDURAL nodes (rules)...")
    
    procedural_rules = [
        "When asked about preferences, recall semantic facts first",
        "When context is long, prioritize high PageRank nodes",
        "When user mentions 'ContextOS', retrieve related memories",
        "Apply decay to old episodic memories",
    ]
    
    procedural_ids = []
    for rule in procedural_rules:
        node_id = client.add_memory(rule, MemoryType.PROCEDURAL)
        procedural_ids.append(node_id)
        print(f"   ✅ {rule[:50]}...")
    
    # =========================================
    # CREATE EDGES (Relationships)
    # This is where the graph magic happens
    # =========================================
    
    print("\n🔗 Creating relationship edges...")
    
    # Connect episodic events to their semantic anchors
    # (This creates the hub-spoke pattern)
    
    contextos_semantic = semantic_ids[1]  # "Aryan is building ContextOS..."
    pagerank_semantic = semantic_ids[2]   # "ContextOS uses hybrid retrieval..."
    user_semantic = semantic_ids[0]       # "User's name is Aryan..."
    
    # Link ContextOS-related episodic to ContextOS semantic
    contextos_related = [0, 1, 2, 3, 4, 6, 7]  # indices of related events
    for idx in contextos_related:
        if idx < len(episodic_ids):
            edge = ContextEdge(
                source=episodic_ids[idx],
                target=contextos_semantic,
                relation="DERIVED_FROM",
                weight=0.8
            )
            client.kernel.add_edge(edge)
    
    # Link user-related events to user semantic
    user_related = [5]  # internship related
    for idx in user_related:
        if idx < len(episodic_ids):
            edge = ContextEdge(
                source=episodic_ids[idx],
                target=user_semantic,
                relation="ABOUT",
                weight=0.9
            )
            client.kernel.add_edge(edge)
    
    # Link semantic nodes to each other (knowledge network)
    semantic_links = [
        (0, 1),  # User -> Building ContextOS
        (1, 2),  # ContextOS -> Hybrid retrieval
        (2, 3),  # Hybrid retrieval -> CoALA
        (1, 6),  # ContextOS -> PyPI
        (0, 5),  # User -> Internships
        (3, 7),  # CoALA -> Memory equation
    ]
    
    for src_idx, tgt_idx in semantic_links:
        edge = ContextEdge(
            source=semantic_ids[src_idx],
            target=semantic_ids[tgt_idx],
            relation="CAUSAL",
            weight=1.0
        )
        client.kernel.add_edge(edge)
    
    # Link procedural to semantic (rules depend on facts)
    for proc_id in procedural_ids:
        edge = ContextEdge(
            source=proc_id,
            target=random.choice(semantic_ids),
            relation="REQUIRES",
            weight=0.7
        )
        client.kernel.add_edge(edge)
    
    # Add some temporal chains in episodic (conversation flow)
    for i in range(len(episodic_ids) - 1):
        if random.random() > 0.7:  # 30% chance of temporal link
            edge = ContextEdge(
                source=episodic_ids[i],
                target=episodic_ids[i + 1],
                relation="TEMPORAL",
                weight=0.5
            )
            client.kernel.add_edge(edge)
    
    print(f"   ✅ Created {client.kernel.graph.number_of_edges()} edges")
    
    # =========================================
    # GENERATE VISUALIZATION
    # =========================================
    
    print("\n" + "=" * 60)
    print("🎨 GENERATING VISUALIZATION...")
    print("=" * 60)
    
    output_file = client.kernel.visualize_brain("brain.html")
    
    # Stats
    stats = client.kernel.stats()
    print(f"\n📊 MEMORY STATISTICS:")
    print(f"   Total Nodes: {stats['total_nodes']}")
    print(f"   Total Edges: {stats['total_edges']}")
    print(f"   Vector Index: {stats['vectors']}")
    
    print("\n" + "=" * 60)
    print("✨ DONE! Open brain.html in your browser")
    print("=" * 60)
    print("""
    LEGEND:
    🔵 CYAN nodes   = Semantic (Facts) - Should be HUBS
    🟠 ORANGE nodes = Episodic (Events) - Should cluster around hubs  
    🟢 GREEN nodes  = Procedural (Rules)
    
    The SIZE of each node = its PageRank (importance)
    Watch how semantic facts become central anchors!
    """)
    
    return client


if __name__ == "__main__":
    generate_demo_brain()
