"""Ablation Study: Vector vs. Graph vs. Hybrid retrieval comparison"""

import sys
import os
import shutil
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agentic_memory.core.graph import ContextGraph
from agentic_memory.core.schema import ContextNode, MemoryType, ContextEdge
from agentic_memory.memory.compiler import ContextCompiler


def run_ablation():
    print("🔬 STARTING ABLATION STUDY: Vector vs. Graph vs. Hybrid")
    print("=" * 60)
    
    for path in ["context_os_db.json", "context_os_chroma"]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    
    kernel = ContextGraph()
    
    n1 = ContextNode(
        content="Project Apollo uses the Rust programming language.", 
        type=MemoryType.SEMANTIC
    )
    n2 = ContextNode(
        content="Rust provides memory safety without garbage collection.", 
        type=MemoryType.SEMANTIC
    )
    n3 = ContextNode(
        content="The Apollo space program landed on the moon in 1969.", 
        type=MemoryType.SEMANTIC
    )
    n4 = ContextNode(
        content="Airplane safety regulations require regular inspections.", 
        type=MemoryType.SEMANTIC
    )
    n5 = ContextNode(
        content="Human memory can be improved with regular exercise.", 
        type=MemoryType.SEMANTIC
    )
    
    kernel.add_node(n1)
    kernel.add_node(n2)
    kernel.add_node(n3)
    kernel.add_node(n4)
    kernel.add_node(n5)
    
    kernel.add_edge(ContextEdge(source=n1.id, target=n2.id, relation="USES"))
    
    print(f"\n📊 Graph Setup:")
    print(f"   Node 1: '{n1.content}' [TARGET]")
    print(f"   Node 2: '{n2.content}' [TARGET - linked via graph]")
    print(f"   Node 3: '{n3.content}' [distractor]")
    print(f"   Node 4: '{n4.content}' [distractor]")
    print(f"   Node 5: '{n5.content}' [distractor]")
    print(f"   Edge: Node1 --USES--> Node2")
    
    compiler = ContextCompiler(kernel)
    query = "Why is Project Apollo memory safe?"
    
    print(f"\n🔍 Query: '{query}'")
    print("   Expected: Need 'Project Apollo' AND 'Rust memory safety'")
    print("   Challenge: Node 2 doesn't mention 'Apollo' - requires graph hop!")
    
    configs = [
        {"name": "Vector Only (RAG)", "alpha": 50.0, "beta": 0.0},
        {"name": "Graph Only (Topology)", "alpha": 0.0, "beta": 50.0},
        {"name": "ContextOS (Hybrid)", "alpha": 50.0, "beta": 50.0}
    ]
    
    scores = []
    results = []
    
    for config in configs:
        print(f"\n{'=' * 60}")
        print(f"Testing: {config['name']} (α={config['alpha']}, β={config['beta']})")
        print("-" * 60)
        
        context = compiler.compile(
            query, 
            token_budget=30,
            alpha=config['alpha'], 
            beta=config['beta'],
            verbose=True
        )
        
        has_apollo_rust = "Project Apollo" in context and "Rust" in context
        has_rust_safety = "memory safety" in context or "without garbage" in context
        has_distractor_moon = "moon" in context or "1969" in context
        has_distractor_airplane = "Airplane" in context
        has_distractor_human = "Human memory" in context
        
        score = 0
        if has_apollo_rust: 
            score += 50
            print("   ✅ Found Project Apollo + Rust connection")
        else:
            print("   ❌ Missing Project Apollo + Rust connection")
            
        if has_rust_safety: 
            score += 50
            print("   ✅ Found Rust memory safety info (multi-hop success!)")
        else:
            print("   ❌ Missing Rust memory safety info (multi-hop failed)")
            
        if has_distractor_moon:
            print("   ⚠️  Retrieved distractor: Apollo space program")
        if has_distractor_airplane:
            print("   ⚠️  Retrieved distractor: Airplane safety")
        if has_distractor_human:
            print("   ⚠️  Retrieved distractor: Human memory")
        
        scores.append(score)
        results.append({
            "config": config['name'],
            "context": context,
            "score": score,
            "has_apollo": has_apollo_rust,
            "has_memory_safety": has_rust_safety
        })
        
        print(f"\n   📝 Retrieved Context:")
        for line in context.split('\n'):
            if line.strip():
                print(f"      {line}")
        print(f"\n   🎯 Score: {score}/100")

    print("\n" + "=" * 60)
    print("📊 ABLATION STUDY RESULTS")
    print("=" * 60)
    for r in results:
        status = "✅" if r['score'] == 100 else "⚠️" if r['score'] >= 50 else "❌"
        print(f"   {status} {r['config']}: {r['score']}%")
    
    plot_ablation(configs, scores)
    return results


def plot_ablation(configs, scores):
    """Generate ablation study visualization."""
    names = [c['name'] for c in configs]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, scores, color=colors, edgecolor='black', linewidth=1.2)
    
    plt.ylabel('Retrieval Accuracy (%)', fontsize=12)
    plt.xlabel('Configuration', fontsize=12)
    plt.title('Ablation Study: Multi-Hop Reasoning Retrieval\n"Why is Project Apollo memory safe?"', fontsize=14)
    plt.ylim(0, 110)
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + 2,
            f'{score}%', 
            ha='center', 
            va='bottom', 
            fontsize=14, 
            fontweight='bold'
        )
    
    plt.figtext(0.5, 0.02, 
                'Score = 50% if found "Project Apollo" + 50% if found "memory safety" (requires graph traversal)',
                ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    os.makedirs('paper/figures', exist_ok=True)
    plt.savefig('paper/figures/ablation_chart.png', dpi=300, bbox_inches='tight')
    print("\n📊 Chart saved to 'paper/figures/ablation_chart.png'")


if __name__ == "__main__":
    run_ablation()
