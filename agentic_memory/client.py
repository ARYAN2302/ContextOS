"""
ContextClient - The main entry point for the Agentic Memory Framework.

This class wraps all complexity of graph traversal, vector search,
and centrality calculation into a simple API.
"""

from typing import Callable, Optional

from .core.graph import ContextGraph
from .core.schema import ContextNode, MemoryType
from .memory.ingestor import Ingestor
from .memory.compiler import ContextCompiler


class ContextClient:
    """
    The main entry point for the Agentic Memory Framework.
    
    Example:
        >>> from agentic_memory import ContextClient, MemoryType
        >>> client = ContextClient()
        >>> client.add_memory("User prefers dark mode", MemoryType.SEMANTIC)
        >>> context = client.compile("What are the user's preferences?")
    """
    
    def __init__(
        self, 
        storage_path: str = "context_os_db.json",
        chroma_path: str = "context_os_chroma",
        auto_persist: bool = True
    ):
        """
        Initialize ContextOS.
        
        Args:
            storage_path: Path for graph persistence (JSON)
            chroma_path: Path for vector store (ChromaDB)
            auto_persist: Whether to auto-save on changes
        """
        self.kernel = ContextGraph(
            storage_path=storage_path,
            chroma_path=chroma_path
        )
        self.ingestor = Ingestor()
        self.compiler = ContextCompiler(self.kernel)
        
        stats = self.kernel.stats()
        print(f"🟢 ContextOS Initialized. Memory: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
    
    def add_memory(
        self, 
        text: str, 
        memory_type: Optional[MemoryType] = None,
        auto_classify: bool = True
    ) -> str:
        """
        Add a memory to the context graph.
        
        Args:
            text: The content to remember
            memory_type: Optional type override (SEMANTIC, EPISODIC, PROCEDURAL)
            auto_classify: Use LLM to classify type if memory_type not provided
            
        Returns:
            The ID of the created memory node
        """
        if auto_classify and memory_type is None:
            node = self.ingestor.parse_input(text)
        else:
            node = ContextNode(
                content=text,
                type=memory_type or MemoryType.EPISODIC
            )
        
        self.kernel.add_node(node)
        return node.id
    
    def compile(
        self, 
        query: str, 
        token_budget: int = 500,
        alpha: float = 50.0,
        beta: float = 50.0
    ) -> str:
        """
        Compile relevant context for a query.
        
        Args:
            query: The query to compile context for
            token_budget: Max tokens in the context window
            alpha: Weight for semantic similarity (vector)
            beta: Weight for graph centrality (PageRank)
            
        Returns:
            Formatted context string for LLM prompt injection
        """
        return self.compiler.compile(
            query=query,
            token_budget=token_budget,
            alpha=alpha,
            beta=beta
        )
    
    def chat(
        self, 
        query: str, 
        llm_callable: Callable[[str, str], str],
        token_budget: int = 500
    ) -> str:
        """
        Run a full RAG loop: Ingest -> Compile -> Generate -> Log.
        
        Args:
            query: User's input message
            llm_callable: Function(system_prompt, user_query) -> response
            token_budget: Max tokens for context window
            
        Returns:
            The LLM's response string
        """
        user_node = self.ingestor.parse_input(query)
        self.kernel.add_node(user_node)
        
        context_window = self.compiler.compile(query, token_budget=token_budget)
        
        system_prompt = f"""You are an AI Assistant powered by ContextOS.
Your memory is dynamic and persistent. Use the Context Block below to inform your responses.

{context_window}"""
        
        response = llm_callable(system_prompt, query)
        
        ai_node = self.ingestor.parse_input(f"AI replied: {response}")
        self.kernel.add_node(ai_node)
        
        return response
    
    def stats(self) -> dict:
        """Get current memory statistics."""
        return self.kernel.stats()
    
    def clear(self) -> None:
        """Clear all memory (use with caution)."""
        self.kernel = ContextGraph(
            storage_path=self.kernel.storage_path,
            chroma_path=self.kernel.chroma_path
        )
        self.compiler = ContextCompiler(self.kernel)
        print("🗑️ Memory cleared.")
