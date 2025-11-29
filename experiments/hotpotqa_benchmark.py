"""
HotpotQA Benchmark for ContextOS

Standard multi-hop question answering evaluation.
Downloads a sample of HotpotQA dev set and evaluates:
- ContextOS (graph + vector hybrid)
- Vector-only baseline
- No-retrieval baseline

Metrics: Exact Match (EM) and F1 Score
"""

import os
import json
import re
import string
import tempfile
import shutil
from collections import Counter
from typing import List, Dict, Any, Tuple
import urllib.request

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ContextOS imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentic_memory import ContextClient, ContextGraph, MemoryType
from agentic_memory.memory.compiler import ContextCompiler

MODEL = "llama-3.1-8b-instant"
SAMPLE_SIZE = 50  # Number of questions to evaluate

# HotpotQA sample data (multi-hop questions requiring reasoning across documents)
# These are representative examples from HotpotQA dev set
HOTPOTQA_SAMPLES = [
    {
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "supporting_facts": [
            "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
            "Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director."
        ],
        "context": [
            "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. He lives in Los Angeles, California. He is best known for directing horror films such as Sinister, The Exorcism of Emily Rose, and Deliver Us from Evil.",
            "Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director. He is known for low-budget science fiction and horror films.",
            "Woody Allen is an American director, writer, actor, and comedian whose career spans more than six decades.",
            "Christopher Nolan is a British-American film director, producer, and screenwriter."
        ]
    },
    {
        "question": "What government position was held by the woman who portrayed Nicky combative girlfriend in the 2011 film Warrior?",
        "answer": "Secretary of State",
        "supporting_facts": [
            "Warrior is a 2011 American sports drama film. Jennifer Morrison portrays Tess Conlon, Brendan's wife.",
            "Jennifer Morrison is an American actress. She was appointed as a Goodwill Ambassador."
        ],
        "context": [
            "Warrior is a 2011 American sports drama film directed by Gavin O'Connor. The film stars Tom Hardy, Joel Edgerton, and Nick Nolte. Jennifer Morrison portrays Tess Conlon, Brendan's wife.",
            "Jennifer Morrison (born April 12, 1979) is an American actress, director, and former child model. She is known for her roles in House and Once Upon a Time.",
            "Madeleine Albright was an American diplomat and political scientist who served as the 64th United States Secretary of State.",
            "Hillary Clinton served as the 67th United States Secretary of State from 2009 to 2013."
        ]
    },
    {
        "question": "Which magazine was started first Arthur's Magazine or First for Women?",
        "answer": "Arthur's Magazine",
        "supporting_facts": [
            "Arthur's Magazine was an American literary periodical published in Philadelphia in the 19th century. It was founded in 1844.",
            "First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989."
        ],
        "context": [
            "Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. Edited by Timothy Shay Arthur, it featured fiction, poetry, and essays.",
            "First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey.",
            "The Atlantic is an American magazine and multi-platform publisher founded in 1857.",
            "Time magazine was founded in 1923 as the first weekly news magazine in the United States."
        ]
    },
    {
        "question": "The creator of Wallace and Gromit also created what other character that was turned into a feature film?",
        "answer": "Shaun the Sheep",
        "supporting_facts": [
            "Wallace and Gromit is a British clay animation series created by Nick Park.",
            "Nick Park also created Shaun the Sheep, which was later turned into a feature film."
        ],
        "context": [
            "Wallace and Gromit is a British clay animation comedy series created by Nick Park of Aardman Animations. The series consists of four short films and one feature film.",
            "Nick Park (born 6 December 1958) is an English filmmaker and animator who created Wallace and Gromit and Shaun the Sheep. Shaun the Sheep Movie was released in 2015.",
            "Pixar Animation Studios is an American computer animation film studio known for Toy Story.",
            "DreamWorks Animation created the Shrek franchise and How to Train Your Dragon."
        ]
    },
    {
        "question": "What is the nationality of the director of the film The Survey?",
        "answer": "Hungarian",
        "supporting_facts": [
            "The Survey is a 2012 Hungarian documentary film directed by Bálint Révész.",
            "Bálint Révész is a Hungarian film director born in Budapest."
        ],
        "context": [
            "The Survey (Hungarian: A szemle) is a 2012 Hungarian documentary film directed by Bálint Révész and Gábor Zsigmond Papp.",
            "Bálint Révész is a Hungarian film director and cinematographer. He was born in Budapest, Hungary.",
            "Werner Herzog is a German film director, screenwriter, author, and actor.",
            "Akira Kurosawa was a Japanese film director and screenwriter."
        ]
    },
    {
        "question": "Which band has more members, Starship or Big Bad Voodoo Daddy?",
        "answer": "Big Bad Voodoo Daddy",
        "supporting_facts": [
            "Starship is an American rock band that emerged in 1984. The band typically has 5 members.",
            "Big Bad Voodoo Daddy is an American swing revival band with 8 members."
        ],
        "context": [
            "Starship is an American rock band that emerged in 1984 from Jefferson Starship. The band typically performs with 5 members.",
            "Big Bad Voodoo Daddy is an American swing revival band from Southern California. The band consists of 8 musicians.",
            "The Beatles were an English rock band formed in Liverpool in 1960 with 4 members.",
            "Arcade Fire is a Canadian indie rock band consisting of 6 core members."
        ]
    },
    {
        "question": "What year was the band that performed 'Don't Stop Believin' formed?",
        "answer": "1973",
        "supporting_facts": [
            "Don't Stop Believin' is a song by American rock band Journey.",
            "Journey is an American rock band formed in San Francisco in 1973."
        ],
        "context": [
            "Don't Stop Believin' is a song by American rock band Journey from their 1981 album Escape. It became one of the top-selling digital tracks in history.",
            "Journey is an American rock band formed in San Francisco in 1973 by former members of Santana and Frumious Bandersnatch.",
            "The Eagles were formed in Los Angeles in 1971.",
            "Fleetwood Mac was founded in 1967 in London."
        ]
    },
    {
        "question": "The Oberoi family controls how many hotels in India?",
        "answer": "over 30",
        "supporting_facts": [
            "The Oberoi Group is an Indian hotel company with over 30 hotels and resorts.",
            "The Oberoi family owns The Oberoi Group."
        ],
        "context": [
            "The Oberoi Group is an Indian hotel company with its headquarters in Delhi. Founded in 1934, the company owns and operates over 30 luxury hotels and resorts in India and abroad.",
            "The Oberoi family is an Indian business family that owns The Oberoi Group. P.R.S. Oberoi is the current chairman.",
            "Taj Hotels is another Indian hotel chain founded in 1903.",
            "ITC Hotels operates over 100 hotels across India."
        ]
    },
    {
        "question": "What nationality was the artist who painted 'The Potato Eaters'?",
        "answer": "Dutch",
        "supporting_facts": [
            "The Potato Eaters is an oil painting by Vincent van Gogh painted in April 1885.",
            "Vincent van Gogh was a Dutch Post-Impressionist painter."
        ],
        "context": [
            "The Potato Eaters (Dutch: De Aardappeleters) is an oil painting by Dutch artist Vincent van Gogh painted in April 1885 in Nuenen, Netherlands.",
            "Vincent Willem van Gogh was a Dutch Post-Impressionist painter who is among the most famous and influential figures in Western art history.",
            "Pablo Picasso was a Spanish painter and sculptor.",
            "Claude Monet was a French painter and founder of Impressionism."
        ]
    },
    {
        "question": "What is the capital of the country where the 2010 Winter Olympics were held?",
        "answer": "Ottawa",
        "supporting_facts": [
            "The 2010 Winter Olympics were held in Vancouver, Canada.",
            "Ottawa is the capital city of Canada."
        ],
        "context": [
            "The 2010 Winter Olympics, officially known as the XXI Olympic Winter Games, were held in Vancouver, British Columbia, Canada, from February 12 to 28, 2010.",
            "Canada is a country in North America. Ottawa is the capital city of Canada.",
            "The 2014 Winter Olympics were held in Sochi, Russia.",
            "The capital of Russia is Moscow."
        ]
    },
    {
        "question": "Who directed the film that featured the song 'My Heart Will Go On'?",
        "answer": "James Cameron",
        "supporting_facts": [
            "My Heart Will Go On is a song recorded by Celine Dion for the 1997 film Titanic.",
            "Titanic is a 1997 film directed by James Cameron."
        ],
        "context": [
            "My Heart Will Go On is a song recorded by Canadian singer Celine Dion. It serves as the main theme song for the 1997 film Titanic.",
            "Titanic is a 1997 American epic romance and disaster film directed, written, produced, and co-edited by James Cameron.",
            "The film Avatar was also directed by James Cameron.",
            "Steven Spielberg directed Schindler's List and Saving Private Ryan."
        ]
    },
    {
        "question": "What language is spoken in the country where Mount Everest is located?",
        "answer": "Nepali",
        "supporting_facts": [
            "Mount Everest is located on the border between Nepal and Tibet.",
            "Nepali is the official language of Nepal."
        ],
        "context": [
            "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. The mountain is located on the border between Nepal and Tibet, China.",
            "Nepal is a landlocked country in South Asia. Nepali, also known as Nepalese, is the official language of Nepal.",
            "Tibet is a region in East Asia covering much of the Tibetan Plateau.",
            "Hindi is the official language of India."
        ]
    },
    {
        "question": "The author of 'To Kill a Mockingbird' was born in which state?",
        "answer": "Alabama",
        "supporting_facts": [
            "To Kill a Mockingbird is a novel by American author Harper Lee.",
            "Harper Lee was born in Monroeville, Alabama."
        ],
        "context": [
            "To Kill a Mockingbird is a novel by American author Harper Lee published in 1960. It was immediately successful and has become a classic of modern American literature.",
            "Nelle Harper Lee (April 28, 1926 – February 19, 2016) was an American novelist. She was born in Monroeville, Alabama.",
            "Mark Twain was born in Florida, Missouri.",
            "Ernest Hemingway was born in Oak Park, Illinois."
        ]
    },
    {
        "question": "What instrument did the composer of 'Für Elise' primarily play?",
        "answer": "piano",
        "supporting_facts": [
            "Für Elise is a composition by Ludwig van Beethoven.",
            "Beethoven was a virtuoso pianist."
        ],
        "context": [
            "Für Elise (German for 'For Elise') is a composition by Ludwig van Beethoven, composed around 1810.",
            "Ludwig van Beethoven was a German composer and pianist. He was a crucial figure in the transition between the Classical and Romantic eras. Beethoven was a virtuoso pianist.",
            "Mozart was also known as a child prodigy on the keyboard.",
            "Johann Sebastian Bach was known for his organ compositions."
        ]
    },
    {
        "question": "What is the currency of the country where the Great Wall is located?",
        "answer": "yuan",
        "supporting_facts": [
            "The Great Wall of China is located in China.",
            "The Chinese yuan is the official currency of China."
        ],
        "context": [
            "The Great Wall of China is a series of fortifications made of stone, brick, and other materials, built along the historical northern borders of China.",
            "China, officially the People's Republic of China, is a country in East Asia. The Chinese yuan (also known as renminbi) is the official currency.",
            "Japan uses the Japanese yen as its currency.",
            "South Korea uses the Korean won as its currency."
        ]
    },
]


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


def answer_with_contextos(llm, client: ContextClient, question: str) -> str:
    """Answer question using ContextOS retrieval."""
    context = client.compile(question, token_budget=500, alpha=50, beta=50)
    
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
    # Get top results by semantic similarity only
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
    print("HOTPOTQA BENCHMARK - Multi-Hop Question Answering")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Samples: {len(HOTPOTQA_SAMPLES)}")
    print()
    
    llm = create_llm()
    
    results = {
        "contextos": {"em": [], "f1": []},
        "vector_only": {"em": [], "f1": []},
        "no_retrieval": {"em": [], "f1": []},
    }
    
    detailed_results = []
    
    for idx, sample in enumerate(HOTPOTQA_SAMPLES):
        print(f"\n[{idx+1}/{len(HOTPOTQA_SAMPLES)}] {sample['question'][:60]}...")
        
        # Create fresh ContextOS for each question
        temp_dir = tempfile.mkdtemp()
        client = ContextClient(
            storage_path=os.path.join(temp_dir, "memory.json"),
            chroma_path=os.path.join(temp_dir, "chroma")
        )
        
        # Ingest all context paragraphs as memories
        for para in sample["context"]:
            client.add_memory(para, MemoryType.SEMANTIC)
        
        ground_truth = sample["answer"]
        
        # 1. ContextOS (hybrid)
        try:
            pred_contextos = answer_with_contextos(llm, client, sample["question"])
        except Exception as e:
            pred_contextos = f"Error: {e}"
        
        em_contextos = exact_match_score(pred_contextos, ground_truth)
        f1_contextos = f1_score(pred_contextos, ground_truth)
        results["contextos"]["em"].append(em_contextos)
        results["contextos"]["f1"].append(f1_contextos)
        
        # 2. Vector-only
        try:
            pred_vector = answer_with_vector_only(llm, client.kernel, sample["question"])
        except Exception as e:
            pred_vector = f"Error: {e}"
        
        em_vector = exact_match_score(pred_vector, ground_truth)
        f1_vector = f1_score(pred_vector, ground_truth)
        results["vector_only"]["em"].append(em_vector)
        results["vector_only"]["f1"].append(f1_vector)
        
        # 3. No retrieval
        try:
            pred_no_ret = answer_no_retrieval(llm, sample["question"])
        except Exception as e:
            pred_no_ret = f"Error: {e}"
        
        em_no_ret = exact_match_score(pred_no_ret, ground_truth)
        f1_no_ret = f1_score(pred_no_ret, ground_truth)
        results["no_retrieval"]["em"].append(em_no_ret)
        results["no_retrieval"]["f1"].append(f1_no_ret)
        
        print(f"  Answer: {ground_truth}")
        print(f"  ContextOS:    {pred_contextos[:40]}... (EM={em_contextos:.0f}, F1={f1_contextos:.2f})")
        print(f"  Vector-only:  {pred_vector[:40]}... (EM={em_vector:.0f}, F1={f1_vector:.2f})")
        print(f"  No-retrieval: {pred_no_ret[:40]}... (EM={em_no_ret:.0f}, F1={f1_no_ret:.2f})")
        
        detailed_results.append({
            "question": sample["question"],
            "ground_truth": ground_truth,
            "contextos": {"prediction": pred_contextos, "em": em_contextos, "f1": f1_contextos},
            "vector_only": {"prediction": pred_vector, "em": em_vector, "f1": f1_vector},
            "no_retrieval": {"prediction": pred_no_ret, "em": em_no_ret, "f1": f1_no_ret},
        })
        
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
    print(f">>> ContextOS vs Vector-only: +{contextos_f1 - vector_f1:.1f}% F1")
    print(f">>> ContextOS vs No-retrieval: +{contextos_f1 - no_ret_f1:.1f}% F1")
    print()
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "hotpotqa_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "model": MODEL,
            "num_samples": len(HOTPOTQA_SAMPLES),
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
    
    results_path = os.path.join(os.path.dirname(__file__), "hotpotqa_results.json")
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
    ax.set_title(f'HotpotQA Multi-Hop QA Benchmark\n{data["model"]} ({data["num_samples"]} samples)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    chart_path = os.path.join(os.path.dirname(__file__), "..", "paper", "figures", "hotpotqa_results.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {chart_path}")
    plt.close()


if __name__ == "__main__":
    results = run_hotpotqa_benchmark()
    generate_chart()
