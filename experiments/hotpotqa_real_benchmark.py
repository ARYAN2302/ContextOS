"""
HotpotQA Benchmark for ContextOS (Real Dataset)

Uses the official HotpotQA dev set (distractor setting).
Each question has 10 context paragraphs - only 2 are relevant.

Metrics: Exact Match (EM) and F1 Score
"""

import os
import json
import re
import string
import tempfile
import shutil
from collections import Counter
from typing import List, Dict, Any

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ContextOS imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentic_memory import ContextClient, ContextGraph, MemoryType
from agentic_memory.memory.compiler import ContextCompiler

MODEL = "llama-3.1-8b-instant"
SAMPLE_SIZE = 50  # Number of questions to evaluate (full dataset has 7405)
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hotpot_dev_distractor_v1.json")


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return float(prediction_tokens == ground_truth_tokens)
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def create_llm():
    return ChatGroq(model=MODEL, temperature=0)


def load_hotpotqa(path: str, n_samples: int = 50) -> List[Dict]:
    """Load HotpotQA dataset."""
    with open(path) as f:
        data = json.load(f)
    
    # Take first n samples (or random sample for more diversity)
    return data[:n_samples]


def format_context(context: List) -> List[str]:
    """Format HotpotQA context into paragraphs."""
    paragraphs = []
    for title, sentences in context:
        text = f"{title}: " + " ".join(sentences)
        paragraphs.append(text)
    return paragraphs


def answer_with_contextos(llm, client: ContextClient, question: str) -> str:
    """Answer question using ContextOS retrieval."""
    context = client.compile(question, token_budget=800, alpha=50, beta=50)
    
    messages = [
        SystemMessage(content=f"""Answer the question based on the context provided. 
Give a short, direct answer (just the answer, no explanation).

{context}"""),
        HumanMessage(content=question)
    ]
    
    response = llm.invoke(messages)
    return response.content.strip()


def answer_with_vector_only(llm, graph: ContextGraph, question: str) -> str:
    """Answer question using vector-only retrieval (no graph)."""
    results = graph.semantic_search(question, n_results=5)
    
    context = "Context:\n"
    for doc, score in results:
        context += f"- {doc}\n"
    
    messages = [
        SystemMessage(content=f"""Answer the question based on the context provided.
Give a short, direct answer (just the answer, no explanation).

{context}"""),
        HumanMessage(content=question)
    ]
    
    response = llm.invoke(messages)
    return response.content.strip()


def answer_no_retrieval(llm, question: str) -> str:
    """Answer question without any retrieval (pure LLM)."""
    messages = [
        SystemMessage(content="Answer the question directly and concisely. Give a short, direct answer."),
        HumanMessage(content=question)
    ]
    
    response = llm.invoke(messages)
    return response.content.strip()


def run_hotpotqa_benchmark():
    print("=" * 70)
    print("HOTPOTQA BENCHMARK (Real Dataset)")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Dataset: HotpotQA Dev (Distractor Setting)")
    
    # Load data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        print("Run: python -c \"import urllib.request; urllib.request.urlretrieve('http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json', 'data/hotpot_dev_distractor_v1.json')\"")
        return
    
    data = load_hotpotqa(DATA_PATH, SAMPLE_SIZE)
    print(f"Samples: {len(data)}")
    print(f"Context paragraphs per question: 10 (2 relevant, 8 distractors)")
    print()
    
    llm = create_llm()
    
    results = {
        "contextos": {"em": [], "f1": []},
        "vector_only": {"em": [], "f1": []},
        "no_retrieval": {"em": [], "f1": []},
    }
    
    detailed_results = []
    
    for idx, sample in enumerate(data):
        question = sample["question"]
        ground_truth = sample["answer"]
        context_paragraphs = format_context(sample["context"])
        
        print(f"\n[{idx+1}/{len(data)}] {question[:60]}...")
        print(f"  Paragraphs: {len(context_paragraphs)}, Answer: {ground_truth}")
        
        # Create fresh ContextOS for each question
        temp_dir = tempfile.mkdtemp()
        try:
            client = ContextClient(
                storage_path=os.path.join(temp_dir, "memory.json"),
                chroma_path=os.path.join(temp_dir, "chroma")
            )
            
            # Ingest all 10 context paragraphs as memories
            for para in context_paragraphs:
                client.add_memory(para, MemoryType.SEMANTIC)
            
            # 1. ContextOS (hybrid)
            try:
                pred_contextos = answer_with_contextos(llm, client, question)
            except Exception as e:
                pred_contextos = f"Error: {e}"
            
            em_contextos = exact_match_score(pred_contextos, ground_truth)
            f1_contextos = f1_score(pred_contextos, ground_truth)
            results["contextos"]["em"].append(em_contextos)
            results["contextos"]["f1"].append(f1_contextos)
            
            # 2. Vector-only
            try:
                pred_vector = answer_with_vector_only(llm, client.kernel, question)
            except Exception as e:
                pred_vector = f"Error: {e}"
            
            em_vector = exact_match_score(pred_vector, ground_truth)
            f1_vector = f1_score(pred_vector, ground_truth)
            results["vector_only"]["em"].append(em_vector)
            results["vector_only"]["f1"].append(f1_vector)
            
            # 3. No retrieval
            try:
                pred_no_ret = answer_no_retrieval(llm, question)
            except Exception as e:
                pred_no_ret = f"Error: {e}"
            
            em_no_ret = exact_match_score(pred_no_ret, ground_truth)
            f1_no_ret = f1_score(pred_no_ret, ground_truth)
            results["no_retrieval"]["em"].append(em_no_ret)
            results["no_retrieval"]["f1"].append(f1_no_ret)
            
            print(f"  ContextOS:    '{pred_contextos[:30]}...' (EM={em_contextos:.0f}, F1={f1_contextos:.2f})")
            print(f"  Vector-only:  '{pred_vector[:30]}...' (EM={em_vector:.0f}, F1={f1_vector:.2f})")
            print(f"  No-retrieval: '{pred_no_ret[:30]}...' (EM={em_no_ret:.0f}, F1={f1_no_ret:.2f})")
            
            detailed_results.append({
                "question": question,
                "ground_truth": ground_truth,
                "num_paragraphs": len(context_paragraphs),
                "contextos": {"prediction": pred_contextos, "em": em_contextos, "f1": f1_contextos},
                "vector_only": {"prediction": pred_vector, "em": em_vector, "f1": f1_vector},
                "no_retrieval": {"prediction": pred_no_ret, "em": em_no_ret, "f1": f1_no_ret},
            })
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'Exact Match':<15} {'F1 Score':<15}")
    print("-" * 50)
    
    for method, scores in results.items():
        avg_em = sum(scores["em"]) / len(scores["em"]) * 100
        avg_f1 = sum(scores["f1"]) / len(scores["f1"]) * 100
        print(f"{method:<20} {avg_em:>10.1f}%     {avg_f1:>10.1f}%")
    
    # Calculate improvement
    contextos_f1 = sum(results["contextos"]["f1"]) / len(results["contextos"]["f1"]) * 100
    vector_f1 = sum(results["vector_only"]["f1"]) / len(results["vector_only"]["f1"]) * 100
    no_ret_f1 = sum(results["no_retrieval"]["f1"]) / len(results["no_retrieval"]["f1"]) * 100
    
    print()
    print(f">>> ContextOS vs Vector-only: {contextos_f1 - vector_f1:+.1f}% F1")
    print(f">>> ContextOS vs No-retrieval: {contextos_f1 - no_ret_f1:+.1f}% F1")
    print()
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "hotpotqa_real_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "model": MODEL,
            "dataset": "HotpotQA Dev (Distractor)",
            "num_samples": len(data),
            "paragraphs_per_question": 10,
            "summary": {
                method: {
                    "exact_match": sum(scores["em"]) / len(scores["em"]) * 100,
                    "f1": sum(scores["f1"]) / len(scores["f1"]) * 100,
                }
                for method, scores in results.items()
            },
            "detailed": detailed_results
        }, f, indent=2)
    print(f"Results saved to: {output_path}")
    
    return results


def generate_chart():
    """Generate comparison chart for paper."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping chart generation")
        return
    
    results_path = os.path.join(os.path.dirname(__file__), "hotpotqa_real_results.json")
    if not os.path.exists(results_path):
        print("Run benchmark first to generate results")
        return
    
    with open(results_path) as f:
        data = json.load(f)
    
    methods = ["No Retrieval\n(LLM Only)", "Vector Only\n(Standard RAG)", "ContextOS\n(Hybrid)"]
    f1_scores = [
        data["summary"]["no_retrieval"]["f1"],
        data["summary"]["vector_only"]["f1"],
        data["summary"]["contextos"]["f1"],
    ]
    em_scores = [
        data["summary"]["no_retrieval"]["exact_match"],
        data["summary"]["vector_only"]["exact_match"],
        data["summary"]["contextos"]["exact_match"],
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(methods))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], em_scores, width, label='Exact Match', color='#ff6b6b', edgecolor='black')
    bars2 = ax.bar([i + width/2 for i in x], f1_scores, width, label='F1 Score', color='#6bcb77', edgecolor='black')
    
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(f'HotpotQA Multi-Hop QA Benchmark\n{data["model"]} (n={data["num_samples"]}, 10 paragraphs/question)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    chart_path = os.path.join(os.path.dirname(__file__), "..", "paper", "figures", "hotpotqa_real_results.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {chart_path}")
    plt.close()


if __name__ == "__main__":
    results = run_hotpotqa_benchmark()
    generate_chart()
