"""Basic tests for ContextOS framework."""

import os
import shutil
import uuid
import pytest


def get_test_paths():
    """Generate unique test paths per test."""
    uid = uuid.uuid4().hex[:8]
    return f"test_db_{uid}.json", f"test_chroma_{uid}"


def cleanup_path(db_path, chroma_path):
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)


class TestSchema:
    """Test data models."""
    
    def test_memory_type_enum(self):
        from agentic_memory import MemoryType
        
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.PROCEDURAL.value == "procedural"
    
    def test_context_node_creation(self):
        from agentic_memory import ContextNode, MemoryType
        
        node = ContextNode(content="Test memory", type=MemoryType.SEMANTIC)
        
        assert node.content == "Test memory"
        assert node.type == MemoryType.SEMANTIC
        assert node.id is not None
        assert node.created_at is not None


class TestContextGraph:
    """Test graph kernel."""
    
    def test_graph_initialization(self):
        from agentic_memory import ContextGraph
        
        db, chroma = get_test_paths()
        try:
            graph = ContextGraph(storage_path=db, chroma_path=chroma)
            stats = graph.stats()
            assert stats['total_nodes'] == 0
            assert stats['total_edges'] == 0
        finally:
            cleanup_path(db, chroma)
    
    def test_add_node(self):
        from agentic_memory import ContextGraph, ContextNode, MemoryType
        
        db, chroma = get_test_paths()
        try:
            graph = ContextGraph(storage_path=db, chroma_path=chroma)
            node = ContextNode(content="Test node", type=MemoryType.EPISODIC)
            graph.add_node(node)
            stats = graph.stats()
            assert stats['total_nodes'] == 1
        finally:
            cleanup_path(db, chroma)
    
    def test_semantic_search(self):
        from agentic_memory import ContextGraph, ContextNode, MemoryType
        
        db, chroma = get_test_paths()
        try:
            graph = ContextGraph(storage_path=db, chroma_path=chroma)
            graph.add_node(ContextNode(content="I love sushi", type=MemoryType.SEMANTIC))
            graph.add_node(ContextNode(content="Python is great", type=MemoryType.SEMANTIC))
            results = graph.semantic_search("favorite food", n_results=2)
            assert len(results) > 0
            assert any("sushi" in r[0].lower() for r in results)
        finally:
            cleanup_path(db, chroma)


class TestCompiler:
    """Test context compilation."""
    
    def test_compile_empty_graph(self):
        from agentic_memory import ContextGraph, ContextCompiler
        
        db, chroma = get_test_paths()
        try:
            graph = ContextGraph(storage_path=db, chroma_path=chroma)
            compiler = ContextCompiler(graph)
            result = compiler.compile("test query", token_budget=100)
            assert "CONTEXTOS MEMORY BLOCK" in result
        finally:
            cleanup_path(db, chroma)
    
    def test_compile_with_memories(self):
        from agentic_memory import ContextGraph, ContextCompiler, ContextNode, MemoryType
        
        db, chroma = get_test_paths()
        try:
            graph = ContextGraph(storage_path=db, chroma_path=chroma)
            graph.add_node(ContextNode(content="The secret code is 1234", type=MemoryType.SEMANTIC))
            compiler = ContextCompiler(graph)
            result = compiler.compile("what is the code?", token_budget=200)
            assert "1234" in result
        finally:
            cleanup_path(db, chroma)


class TestContextClient:
    """Test the main client API."""
    
    def test_client_initialization(self):
        from agentic_memory import ContextClient
        
        db, chroma = get_test_paths()
        try:
            client = ContextClient(storage_path=db, chroma_path=chroma)
            stats = client.stats()
            assert stats['total_nodes'] == 0
        finally:
            cleanup_path(db, chroma)
    
    def test_add_memory(self):
        from agentic_memory import ContextClient, MemoryType
        
        db, chroma = get_test_paths()
        try:
            client = ContextClient(storage_path=db, chroma_path=chroma)
            node_id = client.add_memory("User likes dark mode", MemoryType.SEMANTIC, auto_classify=False)
            assert node_id is not None
            assert client.stats()['total_nodes'] == 1
        finally:
            cleanup_path(db, chroma)
    
    def test_compile(self):
        from agentic_memory import ContextClient, MemoryType
        
        db, chroma = get_test_paths()
        try:
            client = ContextClient(storage_path=db, chroma_path=chroma)
            client.add_memory("My favorite color is blue", MemoryType.SEMANTIC, auto_classify=False)
            context = client.compile("What is my favorite color?")
            assert "blue" in context.lower()
        finally:
            cleanup_path(db, chroma)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
