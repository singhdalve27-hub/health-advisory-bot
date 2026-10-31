import os
import sys
import warnings

# Suppress noisy library warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.rag_chain import HealthAdvisoryRAG, FALLBACK_REFUSAL_MESSAGE

def main():
    print("=================================================================")
    print("   Philippine DOH Health-Advisory Assistant (CLI Mode)           ")
    print("   Orchestrated with LangChain, ChromaDB & Gemini 3.8 Flash      ")
    print("=================================================================")
    print("Commands:")
    print("  'exit' or 'quit'  : Exit application")
    print("  'clear'           : Reset conversational memory")
    print("  'sources'         : Toggle showing retrieved context chunks")
    print("=================================================================\n")

    try:
        rag = HealthAdvisoryRAG()
    except Exception as e:
        print(f"Error initializing RAG application: {e}")
        sys.exit(1)

    show_sources = False
    print("Assistant is ready! Ask any health question based on DOH advisories.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Stay safe and healthy!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Health-Advisory Assistant: Stay safe and healthy! Goodbye.")
            break

        if user_input.lower() == "clear":
            rag.clear_memory()
            print("[System] Conversational memory cleared.\n")
            continue

        if user_input.lower() == "sources":
            show_sources = not show_sources
            print(f"[System] Source citation display set to: {show_sources}\n")
            continue

        print("\nSearching DOH advisories & generating answer...")
        result = rag.ask(user_input)

        if result["standalone_query"] != user_input:
            print(f"[Memory / Rephrased Query]: \"{result['standalone_query']}\"")

        print(f"\nHealth-Advisory Assistant:\n{result['answer']}")

        if result.get("is_refusal"):
            print("\n[Guardrail Alert]: Query is out-of-scope or unverified in DOH manual; safe fallback triggered.")

        if show_sources and result.get("sources"):
            print("\n--- Retrieved DOH Context Sources ---")
            for idx, src in enumerate(result["sources"]):
                doc_name = src["metadata"].get("source", "DOH Manual")
                chunk_id = src["metadata"].get("chunk_id", f"chunk_{idx}")
                score = src.get("similarity_score", "N/A")
                print(f"[{idx+1}] Source: {doc_name} | {chunk_id} | Similarity: {score}")
                snippet = src["content"][:160].replace("\n", " ")
                print(f"    Excerpt: {snippet}...\n")

        print("-" * 65)

if __name__ == "__main__":
    main()