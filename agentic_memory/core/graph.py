"""ContextOS Graph Kernel - Hybrid Storage with NetworkX + ChromaDB"""

import networkx as nx
import json
import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Tuple
from .schema import ContextNode, ContextEdge


class ContextGraph:
    """Hybrid graph storage combining topology (NetworkX) with semantic vectors (ChromaDB)."""
    
    def __init__(self, storage_path: str = "context_os_db.json", chroma_path: str = "context_os_chroma"):
        self.storage_path = storage_path
        self.chroma_path = chroma_path
        self.graph = nx.DiGraph()
        
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.vector_collection = self.chroma_client.get_or_create_collection(
            name="context_memory",
            embedding_function=self.embedding_fn
        )
        
        if os.path.exists(storage_path):
            self.load_graph()

    def add_node(self, node: ContextNode) -> str:
        """Saves node to both Graph and Vector DB."""
        attributes = node.model_dump()
        self.graph.add_node(node.id, **attributes)
        self.vector_collection.upsert(
            documents=[node.content],
            metadatas=[{"type": str(node.type.value), "content": node.content}],
            ids=[node.id]
        )
        self._save_graph()
        return node.id

    def add_edge(self, edge: ContextEdge):
        """Adds a relationship between two nodes."""
        if self.graph.has_node(edge.source) and self.graph.has_node(edge.target):
            self.graph.add_edge(
                edge.source, 
                edge.target, 
                relation=edge.relation, 
                weight=edge.weight
            )
            self._save_graph()

    def get_node(self, node_id: str) -> Dict:
        """Retrieves a node by ID."""
        return self.graph.nodes[node_id]

    def stats(self) -> Dict:
        """Returns memory statistics."""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "vectors": self.vector_collection.count()
        }

    def semantic_search(self, query: str, n_results: int = 5) -> List[Tuple[str, float]]:
        """Finds nodes by semantic similarity to query."""
        results = self.vector_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        if results['documents'] and results['distances']:
            return list(zip(results['documents'][0], results['distances'][0]))
        return []

    def get_semantic_neighbors(self, query: str, k: int = 5) -> Tuple[List[str], List[float]]:
        """Finds node IDs by semantic similarity to query."""
        results = self.vector_collection.query(
            query_texts=[query],
            n_results=k
        )
        return results['ids'][0], results['distances'][0]

    def _save_graph(self):
        """Persists the graph to disk."""
        data = nx.node_link_data(self.graph)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_graph(self):
        """Loads the graph from disk."""
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data)
            print(f">>> [ContextOS] Database Loaded ({self.graph.number_of_nodes()} nodes).")
        except Exception as e:
            print(f"⚠️ Database Error: {e}")
            self.graph = nx.DiGraph()