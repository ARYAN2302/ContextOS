"""ContextOS Chat Demo - Interactive agent with persistent memory"""

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from agentic_memory import ContextClient

LLM_MODEL = "llama-3.3-70b-versatile"
llm = ChatGroq(model=LLM_MODEL, temperature=0.7)


def llm_callable(system_prompt: str, user_query: str) -> str:
    """LLM wrapper for ContextClient.chat()"""
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ])
    return response.content


def main():
    print(">>> 🟢 Booting ContextOS...")
    
    client = ContextClient()
    
    print("\n💬 ContextOS Chat Demo Ready. (Type 'exit' to quit)\n")
    
    while True:
        try:
            user_input = input("👤 You: ")
            if user_input.lower() in ["exit", "quit"]:
                print(">>> 🔴 Shutting down. Memory persisted.")
                break
            
            response = client.chat(user_input, llm_callable)
            print(f"\n🤖 AI: {response}\n")
            
        except KeyboardInterrupt:
            print("\n>>> 🔴 Shutting down. Memory persisted.")
            break


if __name__ == "__main__":
    main()
