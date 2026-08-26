import os
import chromadb
import torch
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

def embed_query(text, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """Generates an embedding for the query."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask
    
    return embeddings.tolist()

def main():
    load_dotenv()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, 'data', 'chroma_db')

    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_collection(name='doh_health_advisories')
    
    print("Connecting to Gemini API...")
    gemini_client = genai.Client() 
    
    system_instruction = (
        "You are a helpful and accurate DOH Health-Advisory Assistant. "
        "Answer the user's question using ONLY the information provided in the context below. "
        "If the answer cannot be found in the context, respond exactly with: 'I don't have enough information to answer that based on the DOH manual.' "
        "Do not use any outside knowledge. Always cite the specific disease or section if available in the context."
    )

    print("\n=======================================================")
    print(" Health-Advisory Bot is ready! Type 'exit' to quit.")
    print("=======================================================")

    while True:
        user_query = input("\n👤 You: ")
        
        if user_query.lower() in ['exit', 'quit']:
            print("🤖 Health-Advisory Bot: Stay safe and healthy!")
            break
            
        if not user_query.strip():
            continue

        query_embedding = embed_query(user_query)
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=4
        )
        
        retrieved_context = "\n\n".join(results['documents'][0])

        try:
            response = gemini_client.models.generate_content(
                model='gemini-3.6-flash',
                contents=f"<context>\n{retrieved_context}\n</context>\n\nQuestion: {user_query}",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0
                )
            )
            print(f"\n🤖 Health-Advisory Bot:\n{response.text}")
        except Exception as e:
            print(f"\n🤖 Health-Advisory Bot: Oops, an error occurred: {e}")

if __name__ == "__main__":
    main()