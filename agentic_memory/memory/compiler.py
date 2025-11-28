"""ContextOS Compiler - PageRank + Semantic Vector retrieval"""

import networkx as nx
import time
from typing import List, Dict
from ..core.graph import ContextGraph
from ..core.schema import ContextNode, MemoryType


class ContextCompiler:
    """Compiles relevant context from the memory graph for a given query."""
    
    def __init__(self, graph_kernel: ContextGraph):
        self.kernel = graph_kernel
        self._semantic_cache = {}

    def _calculate_node_weight(self, node: Dict) -> int:
        """Estimates token cost of a node."""
        return int(len(node['content'].split()) * 1.3)

    def _calculate_relevance(self, query: str, node_id: str) -> float:
        """Calculates semantic similarity using vector embeddings."""
        cache_key = f"{query}:{node_id}"
        if cache_key in self._semantic_cache:
            return self._semantic_cache[cache_key]
        
        try:
            results = self.kernel.vector_collection.query(
                query_texts=[query],
                n_results=self.kernel.graph.number_of_nodes(),
            )
            
            if node_id in results['ids'][0]:
                idx = results['ids'][0].index(node_id)
                distance = results['distances'][0][idx]
                similarity = max(0, 1.0 - (distance / 2.0))
                self._semantic_cache[cache_key] = similarity
                return similarity
            
            return 0.0
            
        except Exception as e:
            print(f"    [Compiler] Semantic error: {e}")
            return 0.0

    def compile(self, query: str, token_budget: int = 500, alpha: float = 50.0, beta: float = 10.0, verbose: bool = True) -> str:
        """
        Compiles context using hybrid scoring.
        
        Args:
            query: The user query to compile context for.
            token_budget: Maximum tokens to include.
            alpha: Weight for semantic relevance (vector).
            beta: Weight for graph centrality (PageRank).
            verbose: Print debug information.
        """
        if verbose:
            print(f"    [Compiler] Compiling context for: '{query}'...")
        
        self._semantic_cache = {}
        
        try:
            pagerank_scores = nx.pagerank(self.kernel.graph)
        except:
            pagerank_scores = {n: 1.0 for n in self.kernel.graph.nodes()}

        candidates = []
        current_time = time.time()
        
        for node_id in self.kernel.graph.nodes():
            node = self.kernel.graph.nodes[node_id]
            
            relevance = self._calculate_relevance(query, node_id)
            centrality = pagerank_scores.get(node_id, 0)
            
            age_hours = (current_time - node['last_accessed']) / 3600
            time_decay = node['decay_factor'] ** age_hours
            
            vector_score = relevance * alpha
            graph_score = centrality * beta * time_decay
            total_value = vector_score + graph_score
            
            cost = self._calculate_node_weight(node)
            
            candidates.append({
                "id": node_id,
                "content": node['content'],
                "type": node['type'],
                "value": total_value,
                "relevance": relevance,
                "centrality": centrality,
                "weight": cost
            })

        candidates.sort(key=lambda x: x['value'], reverse=True)
        
        compiled_context = []
        current_tokens = 0
        
        for item in candidates:
            if item['value'] < 0.01:
                continue
            if current_tokens + item['weight'] <= token_budget:
                compiled_context.append(item)
                current_tokens += item['weight']
        
        compiled_context.sort(key=lambda x: x['type'])
        
        final_prompt = "--- CONTEXTOS MEMORY BLOCK ---\n"
        for item in compiled_context:
            final_prompt += f"[{item['type'].upper()}] {item['content']}\n"
        
        if verbose:
            print(f"    [Compiler] Selected {len(compiled_context)} memories ({current_tokens}/{token_budget} tokens).")
            print("    [Compiler] Relevance scores:")
            for c in sorted(candidates, key=lambda x: x['relevance'], reverse=True)[:5]:
                print(f"       - {c['relevance']:.3f} | {c['content'][:50]}...")
        
        return final_prompt