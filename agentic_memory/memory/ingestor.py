"""ContextOS Ingestor - LLM-powered memory classification"""

import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from ..core.schema import ContextNode, MemoryType

INGEST_MODEL = "llama-3.1-8b-instant"
llm = ChatGroq(model=INGEST_MODEL, temperature=0)

INGEST_PROMPT = """You are the Memory Ingestor for ContextOS.
Analyze the user's input and extract a structured memory node.

Classify the input into one of these types:
- "semantic": Permanent facts (e.g., user preferences, world knowledge).
- "episodic": Temporary events (e.g., user actions, specific commands).
- "procedural": Instructions on how to do something.

Output strictly valid JSON with these keys:
{
  "content": "Refined, standalone statement of the memory",
  "type": "semantic" | "episodic" | "procedural",
  "decay_factor": float (0.99 for semantic, 0.9 for episodic)
}"""


class Ingestor:
    """Converts raw text into structured memory nodes using LLM."""
    
    def __init__(self):
        print(">>> [ContextOS] Ingestor Kernel Loaded.")

    def parse_input(self, user_text: str) -> ContextNode:
        """Parses user input into a structured ContextNode."""
        print(f"    [Ingest] Parsing: '{user_text}'...")
        
        try:
            response = llm.invoke([
                SystemMessage(content=INGEST_PROMPT),
                HumanMessage(content=user_text)
            ])
            
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            
            data = json.loads(content)
            node = ContextNode(
                content=data["content"],
                type=data["type"],
                decay_factor=data["decay_factor"]
            )
            
            print(f"    [Ingest] Created Node ({node.type}): {node.id}")
            return node

        except Exception as e:
            print(f"    [Ingest] Error: {e}")
            return ContextNode(content=user_text, type=MemoryType.EPISODIC)