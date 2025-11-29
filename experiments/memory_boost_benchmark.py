"""
Memory Boost Benchmark: SLM + ContextOS vs SLM Alone

Proves the core thesis: A small language model with ContextOS memory
can match or exceed its baseline on memory-dependent tasks.

Test Design:
- Multi-turn conversation with facts scattered across turns
- Questions require recalling previously stated information
- Compare: SLM alone (stateless) vs SLM + ContextOS (stateful)
"""

import os
import json
import tempfile
import shutil
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ContextOS imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentic_memory import ContextClient, MemoryType

MODEL = "llama-3.1-8b-instant"

# Test conversations: (user_input, expected_answer_keywords, is_question)
# None for expected means it's a statement to remember, not a question
TEST_SCENARIOS = [
    {
        "name": "Personal Facts",
        "turns": [
            ("My name is Aryan and I'm a software engineer.", None, False),
            ("I work at a startup called NeuralForge.", None, False),
            ("My favorite programming language is Rust.", None, False),
            ("What's my name?", ["aryan"], True),
            ("Where do I work?", ["neuralforge", "neural forge"], True),
            ("What's my favorite language?", ["rust"], True),
        ]
    },
    {
        "name": "Project Context",
        "turns": [
            ("I'm building a project called Apollo.", None, False),
            ("Apollo is a distributed database written in Go.", None, False),
            ("The main challenge is achieving consensus under network partitions.", None, False),
            ("What project am I working on?", ["apollo"], True),
            ("What language is Apollo written in?", ["go"], True),
            ("What's the main technical challenge?", ["consensus", "partition", "network"], True),
        ]
    },
    {
        "name": "Multi-Hop Reasoning",
        "turns": [
            ("Project Titan uses the Kafka framework.", None, False),
            ("Kafka is written in Scala and Java.", None, False),
            ("Scala runs on the JVM.", None, False),
            ("What framework does Titan use?", ["kafka"], True),
            ("What languages is that framework written in?", ["scala", "java"], True),
            ("What runtime does Scala use?", ["jvm", "java virtual machine"], True),
        ]
    },
    {
        "name": "Preference Retention",
        "turns": [
            ("I prefer dark mode in all my applications.", None, False),
            ("I use vim keybindings everywhere.", None, False),
            ("My timezone is PST.", None, False),
            ("What theme do I prefer?", ["dark"], True),
            ("What keybindings do I use?", ["vim"], True),
            ("What's my timezone?", ["pst", "pacific"], True),
        ]
    },
    {
        "name": "Temporal Context",
        "turns": [
            ("Yesterday I fixed a critical bug in the auth module.", None, False),
            ("Last week I deployed version 2.0 to production.", None, False),
            ("Tomorrow I have a code review with the team.", None, False),
            ("What did I do yesterday?", ["bug", "auth", "fixed"], True),
            ("What happened last week?", ["deployed", "2.0", "production"], True),
            ("What's scheduled for tomorrow?", ["code review", "review", "team"], True),
        ]
    },
]


def create_llm():
    return ChatGroq(model=MODEL, temperature=0)


def query_llm_stateless(llm, conversation_history: list[str], question: str) -> str:
    """Query LLM without any memory - just the current question."""
    messages = [
        SystemMessage(content="You are a helpful assistant. Answer concisely."),
        HumanMessage(content=question)
    ]
    response = llm.invoke(messages)
    return response.content


def query_llm_with_history(llm, conversation_history: list[str], question: str) -> str:
    """Query LLM with full conversation history in context."""
    history_text = "\n".join(conversation_history)
    messages = [
        SystemMessage(content="You are a helpful assistant. Answer concisely based on the conversation."),
        HumanMessage(content=f"Previous conversation:\n{history_text}\n\nQuestion: {question}")
    ]
    response = llm.invoke(messages)
    return response.content


def query_llm_with_contextos(llm, client: ContextClient, question: str) -> str:
    """Query LLM with ContextOS memory retrieval."""
    context = client.compile(question, token_budget=500)
    messages = [
        SystemMessage(content=f"You are a helpful assistant. Answer concisely.\n\n{context}"),
        HumanMessage(content=question)
    ]
    response = llm.invoke(messages)
    return response.content


def check_answer(response: str, expected_keywords: list[str]) -> bool:
    """Check if response contains any expected keyword."""
    response_lower = response.lower()
    return any(kw.lower() in response_lower for kw in expected_keywords)


def run_benchmark():
    print("=" * 60)
    print("MEMORY BOOST BENCHMARK")
    print("SLM + ContextOS vs SLM Alone")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print()

    llm = create_llm()
    
    results = {
        "stateless": {"correct": 0, "total": 0},
        "with_history": {"correct": 0, "total": 0},
        "contextos": {"correct": 0, "total": 0},
    }
    
    detailed_results = []

    for scenario in TEST_SCENARIOS:
        print(f"\n--- Scenario: {scenario['name']} ---")
        
        # Create fresh ContextOS client for each scenario
        temp_dir = tempfile.mkdtemp()
        client = ContextClient(
            storage_path=os.path.join(temp_dir, "memory.json"),
            chroma_path=os.path.join(temp_dir, "chroma")
        )
        
        conversation_history = []
        
        for user_input, expected, is_question in scenario["turns"]:
            if not is_question:
                # Statement - add to memory
                conversation_history.append(f"User: {user_input}")
                client.add_memory(user_input, MemoryType.EPISODIC)
                print(f"  [STORED] {user_input[:50]}...")
            else:
                # Question - test all three methods
                print(f"\n  Q: {user_input}")
                print(f"  Expected: {expected}")
                
                # 1. Stateless (no memory)
                resp_stateless = query_llm_stateless(llm, conversation_history, user_input)
                correct_stateless = check_answer(resp_stateless, expected)
                results["stateless"]["total"] += 1
                if correct_stateless:
                    results["stateless"]["correct"] += 1
                
                # 2. With full history (baseline RAG-like)
                resp_history = query_llm_with_history(llm, conversation_history, user_input)
                correct_history = check_answer(resp_history, expected)
                results["with_history"]["total"] += 1
                if correct_history:
                    results["with_history"]["correct"] += 1
                
                # 3. With ContextOS
                resp_contextos = query_llm_with_contextos(llm, client, user_input)
                correct_contextos = check_answer(resp_contextos, expected)
                results["contextos"]["total"] += 1
                if correct_contextos:
                    results["contextos"]["correct"] += 1
                
                print(f"  Stateless:    {'✓' if correct_stateless else '✗'} - {resp_stateless[:60]}...")
                print(f"  With History: {'✓' if correct_history else '✗'} - {resp_history[:60]}...")
                print(f"  ContextOS:    {'✓' if correct_contextos else '✗'} - {resp_contextos[:60]}...")
                
                detailed_results.append({
                    "scenario": scenario["name"],
                    "question": user_input,
                    "expected": expected,
                    "stateless": {"response": resp_stateless, "correct": correct_stateless},
                    "with_history": {"response": resp_history, "correct": correct_history},
                    "contextos": {"response": resp_contextos, "correct": correct_contextos},
                })
                
                conversation_history.append(f"User: {user_input}")
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for method, data in results.items():
        accuracy = (data["correct"] / data["total"] * 100) if data["total"] > 0 else 0
        print(f"{method:15} : {data['correct']}/{data['total']} ({accuracy:.1f}%)")
    
    # Calculate improvement
    stateless_acc = results["stateless"]["correct"] / results["stateless"]["total"] * 100
    contextos_acc = results["contextos"]["correct"] / results["contextos"]["total"] * 100
    improvement = contextos_acc - stateless_acc
    
    print()
    print(f">>> ContextOS Improvement over Stateless: +{improvement:.1f}%")
    print()
    
    # Save detailed results
    output_path = os.path.join(os.path.dirname(__file__), "memory_boost_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "model": MODEL,
            "summary": results,
            "detailed": detailed_results
        }, f, indent=2)
    print(f"Detailed results saved to: {output_path}")
    
    return results


def generate_chart():
    """Generate comparison chart for paper."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping chart generation")
        return
    
    results_path = os.path.join(os.path.dirname(__file__), "memory_boost_results.json")
    if not os.path.exists(results_path):
        print("Run benchmark first to generate results")
        return
    
    with open(results_path) as f:
        data = json.load(f)
    
    methods = ["Stateless\n(No Memory)", "Full History\n(In-Context)", "ContextOS\n(Graph Memory)"]
    accuracies = [
        data["summary"]["stateless"]["correct"] / data["summary"]["stateless"]["total"] * 100,
        data["summary"]["with_history"]["correct"] / data["summary"]["with_history"]["total"] * 100,
        data["summary"]["contextos"]["correct"] / data["summary"]["contextos"]["total"] * 100,
    ]
    
    colors = ["#ff6b6b", "#ffd93d", "#6bcb77"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, accuracies, color=colors, edgecolor="black", linewidth=1.2)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Memory Boost Benchmark\n{data['model']}", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    chart_path = os.path.join(os.path.dirname(__file__), "..", "paper", "figures", "memory_boost.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to: {chart_path}")
    plt.close()


if __name__ == "__main__":
    results = run_benchmark()
    generate_chart()
