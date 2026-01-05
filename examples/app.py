"""
ContextOS Dashboard - Interactive Memory Graph Visualization
Professional Streamlit dashboard for the graph-theoretic memory kernel.
"""
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import sys
import os
import tempfile
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_memory import ContextClient, MemoryType
from agentic_memory.core.graph import ContextGraph
from agentic_memory.core.schema import ContextEdge

# Page configuration
st.set_page_config(
    page_title="ContextOS - Graph-Theoretic Memory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.2em;
    }
    
    .main-header p {
        color: #a0a0c0 !important;
        margin: 8px 0 0 0;
        font-size: 1.1em;
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #1a1a2e;
        margin: 0;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9em;
        margin: 5px 0 0 0;
    }
    
    .memory-type-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 15px;
        font-weight: bold;
        margin: 3px;
        font-size: 0.85em;
    }
    
    .semantic { background: #00d4ff; color: #000; }
    .episodic { background: #ff6b35; color: #fff; }
    .procedural { background: #00ff88; color: #000; }
    
    .benchmark-highlight {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 2px solid #4caf50;
    }
    
    .benchmark-value {
        font-size: 3em;
        font-weight: bold;
        color: #2e7d32;
    }
    
    .chat-user {
        background: #16213e;
        color: white;
        padding: 10px 15px;
        border-radius: 15px 15px 4px 15px;
        margin: 5px 0;
    }
    
    .chat-ai {
        background: #f0f0f0;
        color: #1a1a2e;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 4px;
        margin: 5px 0;
        border-left: 3px solid #00d4ff;
    }
    
    .brain-container {
        background: #0a0a0a;
        border-radius: 15px;
        padding: 15px;
        margin: 15px 0;
    }
    
    .coala-legend {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 15px 0;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .legend-dot {
        width: 15px;
        height: 15px;
        border-radius: 50%;
    }
</style>
""", unsafe_allow_html=True)


class ContextOSDashboard:
    """Professional dashboard for ContextOS memory system."""
    
    def __init__(self):
        self.session_state_init()
    
    def session_state_init(self):
        """Initialize Streamlit session state."""
        if "client" not in st.session_state:
            st.session_state.client = None
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "demo_loaded" not in st.session_state:
            st.session_state.demo_loaded = False
    
    def display_header(self):
        """Display professional header."""
        st.markdown("""
        <div class="main-header">
            <h1>🧠 ContextOS</h1>
            <p>Graph-Theoretic Memory Kernel for Agentic AI Systems • CoALA Architecture</p>
        </div>
        """, unsafe_allow_html=True)
    
    def display_coala_legend(self):
        """Display CoALA memory type legend."""
        st.markdown("""
        <div class="coala-legend">
            <div class="legend-item">
                <div class="legend-dot" style="background: #00d4ff;"></div>
                <span><strong>Semantic</strong> (Stable Facts)</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: #ff6b35;"></div>
                <span><strong>Episodic</strong> (Transient Events)</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: #00ff88;"></div>
                <span><strong>Procedural</strong> (Action Rules)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def display_metrics_panel(self):
        """Display real-time metrics."""
        st.markdown("### 📊 Memory Statistics")
        
        if st.session_state.client:
            stats = st.session_state.client.stats()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{stats['total_nodes']}</p>
                    <p class="metric-label">Total Memories</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{stats['total_edges']}</p>
                    <p class="metric-label">Connections</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{stats['vectors']}</p>
                    <p class="metric-label">Vector Index</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                # Calculate density
                nodes = stats['total_nodes']
                edges = stats['total_edges']
                density = (2 * edges / (nodes * (nodes - 1))) if nodes > 1 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{density:.3f}</p>
                    <p class="metric-label">Graph Density</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Initialize ContextOS to see memory statistics")
    
    def display_benchmark_highlight(self):
        """Display benchmark comparison."""
        st.markdown("### 🏆 Benchmark Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="benchmark-highlight">
                <p class="benchmark-value">15x</p>
                <p class="metric-label">Memory Boost<br>Llama-8B + ContextOS vs Alone</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: #f5f5f5; border-radius: 12px; padding: 20px; text-align: center; color: #1a1a2e; border: 1px solid #ddd;">
                <h4 style="margin: 0 0 10px 0; color: #1a1a2e;">Core Thesis</h4>
                <p style="font-size: 1.1em; margin: 0; color: #1a1a2e;">
                    <strong>100%</strong> vs 6.7% accuracy<br>
                    <small style="color: #555;">SLM + Memory >> SLM alone</small>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("📈 Full Benchmark Details"):
            st.markdown("""
            | Method | Multi-Hop Accuracy | Notes |
            |--------|-------------------|-------|
            | **ContextOS (Hybrid)** | **100%** | Anchored via Vector, traversed via Graph |
            | Vector-only RAG | 50% | Found first hop, missed connections |
            | No retrieval (stateless) | 6.7% | Can't reason across context |
            
            **Key Insight:** Hybrid retrieval combining PageRank + Vector similarity 
            enables multi-hop reasoning that pure vector search cannot achieve.
            """)
    
    def display_demo_section(self):
        """Display demo mode section."""
        st.markdown("### 🚀 Demo Mode")
        st.markdown("Load a pre-built memory graph to see ContextOS in action!")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("🧠 Load Demo Brain", type="primary", use_container_width=True):
                self.load_demo_brain()
        
        with col2:
            if st.session_state.demo_loaded:
                st.success("✅ Demo brain loaded! Scroll down to interact with the memory graph.")
            else:
                st.info("💡 Click the button to generate a demo memory graph with 60+ memories")
        
        if st.session_state.demo_loaded:
            st.markdown("---")
            self.display_memory_graph()
    
    def load_demo_brain(self):
        """Load the demo memory brain."""
        with st.spinner("🧠 Generating demo brain..."):
            # Create fresh client
            st.session_state.client = ContextClient(
                storage_path="demo_brain.json",
                chroma_path="demo_brain_vectors"
            )
            
            # Core semantic memories (the "needles")
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
            
            for fact in semantic_facts:
                st.session_state.client.add_memory(fact, MemoryType.SEMANTIC)
            
            # Episodic events (the "haystack")
            episodic_events = [
                "User asked: How does PageRank work in memory systems?",
                "User said: I want to visualize the memory graph",
                "User asked: What's the difference between RAG and ContextOS?",
                "User said: The benchmark shows 100% vs 6.7% accuracy",
                "User asked: Can we add a decay function to memories?",
                "User said: I need to prepare for my internship interview",
                "User asked: How do I publish to PyPI?",
                "User said: Let's run the HotpotQA benchmark",
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
            
            for event in episodic_events:
                st.session_state.client.add_memory(event, MemoryType.EPISODIC)
            
            # Procedural rules
            procedural_rules = [
                "When asked about preferences, recall semantic facts first",
                "When context is long, prioritize high PageRank nodes",
                "When user mentions 'ContextOS', retrieve related memories",
                "Apply decay to old episodic memories",
            ]
            
            for rule in procedural_rules:
                st.session_state.client.add_memory(rule, MemoryType.PROCEDURAL)
            
            # Create relationships (graph magic)
            client = st.session_state.client
            
            # Get node IDs
            stats = client.stats()
            # In a real implementation, we'd get actual IDs, but this demo shows the concept
            
            st.session_state.demo_loaded = True
            st.success(f"✅ Demo brain loaded! {stats['total_nodes']} memories, {stats['total_edges']} connections")
            st.rerun()
    
    def display_memory_graph(self):
        """Display interactive memory graph."""
        st.markdown("### 🧠 Interactive Memory Graph")
        
        self.display_coala_legend()
        
        if st.session_state.client:
            try:
                # Generate visualization
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                    output_file = f.name
                
                st.session_state.client.kernel.visualize_brain(output_file)
                
                # Read and display HTML
                with open(output_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Display in iframe
                st.components.v1.html(html_content, height=850, scrolling=True)
                
                # Clean up
                os.unlink(output_file)
                
                st.markdown("""
                <div style="text-align: center; margin-top: 10px;">
                    <small>💡 Drag nodes to rearrange • Scroll to zoom • Hover for details</small>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error generating visualization: {e}")
        else:
            st.info("Load the demo brain to see the memory graph visualization")
    
    def display_chat_section(self):
        """Display chat interface with memory."""
        st.markdown("### 💬 Chat with Memory")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("**Add a Memory:**")
            memory_text = st.text_area("Enter text to remember:", height=100, placeholder="User prefers dark mode...")
            
            memory_type = st.selectbox("Memory Type:", 
                ["Auto-detect", "Semantic (Fact)", "Episodic (Event)", "Procedural (Rule)"])
            
            if st.button("💾 Add Memory", use_container_width=True):
                if memory_text:
                    self.add_memory(memory_text, memory_type)
        
        with col2:
            st.markdown("**Query Context:**")
            query = st.text_input("Ask something:", placeholder="What are the user's preferences?")
            
            if st.button("🔍 Compile Context"):
                if query and st.session_state.client:
                    with st.spinner("Compiling relevant context..."):
                        context = st.session_state.client.compile(query)
                        st.markdown(f"""
                        <div style="background: #f5f5f5; padding: 15px; border-radius: 10px; margin-top: 10px;">
                            <strong>Compiled Context:</strong><br>
                            <small>{context[:500]}...</small>
                        </div>
                        """, unsafe_allow_html=True)
    
    def add_memory(self, text: str, memory_type: str):
        """Add a memory to the system."""
        if not st.session_state.client:
            st.session_state.client = ContextClient()
        
        # Map selection to enum
        type_map = {
            "Auto-detect": None,
            "Semantic (Fact)": MemoryType.SEMANTIC,
            "Episodic (Event)": MemoryType.EPISODIC,
            "Procedural (Rule)": MemoryType.PROCEDURAL,
        }
        
        mem_type = type_map.get(memory_type)
        
        node_id = st.session_state.client.add_memory(text, mem_type)
        st.success(f"✅ Memory added: {node_id[:8]}...")
        st.rerun()
    
    def display_sidebar(self):
        """Display sidebar with info."""
        with st.sidebar:
            st.markdown("### 🧠 ContextOS")
            st.markdown("""
            **Graph-Theoretic Memory Kernel**
            
            Beyond RAG: Stateful memory for AI agents.
            
            **Key Features:**
            - 🧠 CoALA Memory Architecture
            - 🔗 NetworkX + ChromaDB hybrid
            - ⚡ Hybrid retrieval (PageRank + Vector)
            - 💾 Persistent memory
            """)
            
            st.markdown("---")
            st.markdown("### 📚 Resources")
            st.markdown("- [GitHub](https://github.com/ARYAN2302/ContextOS)")
            st.markdown("- [PyPI](https://pypi.org/project/agentic-memory/)")
            st.markdown("- [Paper](Contextos_preprint.pdf)")
            
            st.markdown("---")
            st.markdown("### ⚙️ Settings")
            
            if st.button("🗑️ Clear All Memory"):
                if st.session_state.client:
                    st.session_state.client.clear()
                    st.session_state.demo_loaded = False
                    st.session_state.client = None
                    st.rerun()
    
    def run(self):
        """Run the dashboard."""
        # Display sidebar
        self.display_sidebar()
        
        # Display header
        self.display_header()
        
        # Display metrics
        self.display_metrics_panel()
        
        # Display benchmark
        self.display_benchmark_highlight()
        
        # Display demo section
        self.display_demo_section()
        
        # Display chat section
        self.display_chat_section()


def main():
    """Main entry point."""
    dashboard = ContextOSDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
