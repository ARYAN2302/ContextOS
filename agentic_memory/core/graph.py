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

    def visualize_brain(self, output_file: str = "brain.html", physics: bool = True):
        """
        Renders the memory graph as an interactive 3D network visualization.
        
        Node colors based on Memory Type (CoALA architecture):
        - Semantic (Facts): Cyan - stable, high centrality
        - Episodic (Events): Orange - transient, decaying
        - Procedural (Rules): Green - action patterns
        
        Args:
            output_file: Path to save the HTML visualization
            physics: Enable physics simulation for organic layout
        """
        from pyvis.network import Network
        import math
        from datetime import datetime
        
        # Dark theme network
        net = Network(
            height="800px", 
            width="100%", 
            bgcolor="#0a0a0a", 
            font_color="#ffffff",
            directed=True
        )
        
        # Calculate PageRank for node sizing
        if self.graph.number_of_nodes() > 0:
            pagerank = nx.pagerank(self.graph) if self.graph.number_of_edges() > 0 else {n: 0.1 for n in self.graph.nodes()}
        else:
            pagerank = {}
        
        # Color scheme based on paper's Memory Types
        TYPE_COLORS = {
            "semantic": "#00d4ff",    # Cyan - Facts (stable)
            "episodic": "#ff6b35",    # Orange - Events (transient)
            "procedural": "#00ff88",  # Green - Rules
            "core": "#ffffff",        # White - Identity
            "MemoryType.SEMANTIC": "#00d4ff",
            "MemoryType.EPISODIC": "#ff6b35", 
            "MemoryType.PROCEDURAL": "#00ff88",
        }
        
        # Add nodes with visual encoding
        for node_id, data in self.graph.nodes(data=True):
            content = data.get('content', 'Unknown')[:100]  # Truncate for tooltip
            node_type = str(data.get('type', 'episodic')).lower()
            timestamp = data.get('timestamp', '')
            
            # Color by type
            color = TYPE_COLORS.get(node_type, TYPE_COLORS.get(data.get('type', ''), "#888888"))
            
            # Size by PageRank (importance in graph)
            pr_score = pagerank.get(node_id, 0.1)
            size = 10 + (pr_score * 200)  # Scale PageRank to visible size
            
            # Calculate decay for episodic nodes (from paper: S(t) = S₀·e^(-λt) + β·C(n))
            if 'episodic' in node_type.lower():
                # Simulate decay based on age (if timestamp available)
                decay = 0.7  # Default decay factor
                opacity = max(0.3, decay)
                border_color = f"rgba(255, 107, 53, {opacity})"
            else:
                border_color = color
            
            # Rich tooltip with memory details
            tooltip = f"""
            <b>Type:</b> {node_type}<br>
            <b>Content:</b> {content}<br>
            <b>PageRank:</b> {pr_score:.4f}<br>
            <b>ID:</b> {node_id[:8]}...
            """
            
            net.add_node(
                node_id, 
                label=content[:20] + "..." if len(content) > 20 else content,
                title=tooltip,
                color=color,
                size=size,
                borderWidth=2,
                borderWidthSelected=4,
                font={'size': 10, 'color': '#ffffff'}
            )
        
        # Add edges with relationship labels
        EDGE_COLORS = {
            "TEMPORAL": "#444444",
            "CAUSAL": "#ff0000", 
            "ASSOCIATIVE": "#00ff00",
            "DERIVED_FROM": "#0088ff",
        }
        
        for source, target, edge_data in self.graph.edges(data=True):
            relation = edge_data.get('relation', 'ASSOCIATIVE')
            weight = edge_data.get('weight', 1.0)
            
            edge_color = EDGE_COLORS.get(relation, "#333333")
            
            net.add_edge(
                source, 
                target, 
                color=edge_color,
                width=weight * 2,
                title=relation,
                arrows={'to': {'enabled': True, 'scaleFactor': 0.5}}
            )
        
        # Physics configuration for organic brain-like layout
        if physics:
            net.set_options("""
            {
                "physics": {
                    "enabled": true,
                    "barnesHut": {
                        "gravitationalConstant": -3000,
                        "centralGravity": 0.3,
                        "springLength": 150,
                        "springConstant": 0.04,
                        "damping": 0.09
                    },
                    "stabilization": {
                        "enabled": true,
                        "iterations": 200
                    }
                },
                "interaction": {
                    "hover": true,
                    "tooltipDelay": 100,
                    "zoomView": true,
                    "dragView": true
                },
                "nodes": {
                    "shadow": {
                        "enabled": true,
                        "color": "rgba(0,0,0,0.5)",
                        "size": 10
                    }
                },
                "edges": {
                    "smooth": {
                        "enabled": true,
                        "type": "continuous"
                    }
                }
            }
            """)
        
        # Generate HTML
        net.save_graph(output_file)
        
        # Stats
        print(f"\n🧠 BRAIN VISUALIZATION GENERATED")
        print(f"   📍 File: {output_file}")
        print(f"   🔵 Semantic Nodes (Facts): {sum(1 for _, d in self.graph.nodes(data=True) if 'semantic' in str(d.get('type', '')).lower())}")
        print(f"   🟠 Episodic Nodes (Events): {sum(1 for _, d in self.graph.nodes(data=True) if 'episodic' in str(d.get('type', '')).lower())}")
        print(f"   🔗 Connections: {self.graph.number_of_edges()}")
        print(f"\n   Open in browser: file://{os.path.abspath(output_file)}")
        
        return output_file