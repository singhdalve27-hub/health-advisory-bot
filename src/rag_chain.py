import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

import chromadb
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from src.embeddings import LocalMiniLMEmbeddings
except ImportError:
    from embeddings import LocalMiniLMEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
DEFAULT_COLLECTION = "doh_health_advisories"

FALLBACK_REFUSAL_MESSAGE = "I don't have enough information to answer that based on the DOH manual."

SYSTEM_INSTRUCTION = (
    "You are a helpful and accurate DOH Health-Advisory Assistant. "
    "Answer the user's question using ONLY the factual information provided in the retrieved context below. "
    "Rules you must strictly follow:\n"
    "1. Do not use outside medical knowledge or fabricate treatments/dosages.\n"
    f"2. If the answer cannot be found in the context, or if the question is unrelated to DOH communicable diseases/guidelines, respond EXACTLY with:\n"
    f"\"{FALLBACK_REFUSAL_MESSAGE}\"\n"
    "3. When answering from context, always cite the specific disease, advisory, or section mentioned in the context.\n"
    "4. State clearly any immediate actions (e.g., first aid, washing bites for 10-15 minutes, prophylactic consult) if specified in the text.\n"
    "5. Maintain a professional, empathetic, and responsible public health tone."
)

REPHRASE_SYSTEM_INSTRUCTION = (
    "Given a chat conversation and a follow-up user question, rephrase the follow-up question "
    "to be a standalone search query that contains all necessary entities and context from the chat history. "
    "If the question is already standalone, return it as-is. Do NOT answer the question, only return the rephrased query."
)

def clean_llm_output(content: Any) -> str:
    """Normalize LangChain LLM output into a clean string."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content).strip()

class HealthAdvisoryRAG:
    """
    RAG Orchestration Engine using LangChain, ChromaDB, and Google Gemini with Conversational Memory.
    """
    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = "gemini-3.8-flash",
        temperature: float = 0.0,
        top_k: int = 4
    ):
        load_dotenv()
        self.db_path = db_path
        self.collection_name = collection_name
        self.top_k = top_k
        self.model_name = model_name
        
        # Verify API key
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable is missing. Please set it in .env or environment.")

        # Embedding & ChromaDB client
        self.embedder = LocalMiniLMEmbeddings()
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)

        # Auto-initialize database if empty (ensures zero-downtime deployment on Railway/fresh clones)
        if self.collection.count() == 0:
            default_pdf = os.path.join(BASE_DIR, "data", "7323175-Doh-Communicable-Diseases.pdf")
            if os.path.exists(default_pdf):
                try:
                    from src.ingestion import ingest_document
                    print("Vector store is empty on startup. Auto-ingesting official DOH manual...")
                    ingest_document(default_pdf, filename="7323175-Doh-Communicable-Diseases.pdf", db_path=self.db_path, collection_name=self.collection_name)
                    print(f"Auto-ingestion complete. Total indexed chunks: {self.collection.count()}")
                except Exception as e:
                    print(f"Auto-ingestion warning: {e}")

        # Primary Generation LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature
        )

        # Rephraser LLM for Conversational Memory
        self.rephrase_llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0
        )

        # Conversational Memory State: list of dicts with role and content
        self.chat_history: List[Dict[str, str]] = []

    def clear_memory(self):
        """Reset conversational session memory."""
        self.chat_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Return full conversational history."""
        return list(self.chat_history)

    def contextualize_query(self, query: str) -> str:
        """
        Multi-turn Conversational Memory:
        Rephrase user query into a standalone query taking recent conversation context into account.
        """
        if not self.chat_history:
            return query

        # Build message history for rephrasing
        messages = [SystemMessage(content=REPHRASE_SYSTEM_INSTRUCTION)]
        # Take last 6 turns to avoid bloated context
        for turn in self.chat_history[-6:]:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))
        messages.append(HumanMessage(content=f"Follow-up question to rephrase: {query}"))

        try:
            response = self.rephrase_llm.invoke(messages)
            standalone = clean_llm_output(response.content)
            # Remove any unwanted quotation marks added by LLM
            standalone = standalone.strip("\"' \n")
            return standalone if standalone else query
        except Exception as e:
            print(f"Query rephrase warning (falling back to original query): {e}")
            return query

    def retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve most relevant document chunks from ChromaDB."""
        query_vector = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results and "documents" in results and results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            for doc, meta, dist in zip(docs, metas, dists):
                retrieved.append({
                    "content": doc,
                    "metadata": meta,
                    "distance": dist,
                    "similarity_score": round(1.0 - (dist if dist <= 1.0 else 1.0), 3)
                })
        return retrieved

    def ask(self, user_query: str) -> Dict[str, Any]:
        """
        Execute full RAG generation loop with conversational memory:
        1. Contextualize query with conversational history
        2. Retrieve relevant DOH context from ChromaDB
        3. Prompt generator LLM with strict grounding guardrails
        4. Update conversational memory
        5. Return structured result
        """
        user_query = user_query.strip()
        if not user_query:
            return {
                "answer": "Please provide a valid question.",
                "standalone_query": "",
                "is_refusal": False,
                "sources": []
            }

        # Step 1: Contextualize Query using Chat History
        standalone_query = self.contextualize_query(user_query)

        # Step 2: Retrieve Context Chunks
        retrieved_items = self.retrieve_context(standalone_query)
        context_text = "\n\n---\n\n".join([
            f"[Source: {item['metadata'].get('source', 'DOH Manual')}, Chunk: {item['metadata'].get('chunk_id', 'unknown')}]\n{item['content']}"
            for item in retrieved_items
        ])

        # Step 3: Construct Grounded Prompt with Memory
        messages = [SystemMessage(content=SYSTEM_INSTRUCTION)]

        # Include recent turns in conversation
        for turn in self.chat_history[-6:]:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))

        current_prompt = (
            f"<retrieved_doh_context>\n{context_text}\n</retrieved_doh_context>\n\n"
            f"User Question: {user_query}"
        )
        messages.append(HumanMessage(content=current_prompt))

        # Step 4: Generate Grounded Answer
        try:
            llm_response = self.llm.invoke(messages)
            answer = clean_llm_output(llm_response.content)
        except Exception as e:
            answer = f"Error generating response: {str(e)}"

        # Check refusal / fallback flag
        is_refusal = (
            FALLBACK_REFUSAL_MESSAGE.lower() in answer.lower() or
            "don't have enough information" in answer.lower()
        )

        # Step 5: Update Memory
        self.chat_history.append({"role": "user", "content": user_query})
        self.chat_history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "standalone_query": standalone_query,
            "is_refusal": is_refusal,
            "sources": retrieved_items
        }
