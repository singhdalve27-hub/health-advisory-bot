import os
import re
import io
import time
import argparse
from typing import List, Dict, Any, Union
import PyPDF2
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

try:
    from src.embeddings import LocalMiniLMEmbeddings
except ImportError:
    from embeddings import LocalMiniLMEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
DEFAULT_COLLECTION = "doh_health_advisories"

def extract_text_from_pdf(source: Union[str, bytes, io.BytesIO]) -> str:
    """Extract clean text from a PDF path or file-like buffer."""
    text = ""
    try:
        if isinstance(source, (bytes, bytearray)):
            reader = PyPDF2.PdfReader(io.BytesIO(source))
            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[Page {page_idx + 1}]\n" + page_text + "\n"
        elif isinstance(source, io.BytesIO):
            reader = PyPDF2.PdfReader(source)
            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[Page {page_idx + 1}]\n" + page_text + "\n"
        else:
            with open(source, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_idx, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[Page {page_idx + 1}]\n" + page_text + "\n"
    except Exception as e:
        print(f"Error parsing PDF: {e}")
    return text

def clean_text(raw_text: str) -> str:
    """Normalize whitespace and remove distracting artifacts while preserving section structure."""
    if not raw_text:
        return ""
    # Normalize newline clusters
    text = re.sub(r"\r\n|\r", "\n", raw_text)
    # Replace multiple empty lines with double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove multiple horizontal spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def chunk_document(
    text: str,
    source_name: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Document]:
    """Chunk text using RecursiveCharacterTextSplitter with metadata preservation."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    raw_chunks = splitter.split_text(text)
    
    docs = []
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for idx, chunk in enumerate(raw_chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": source_name,
                "chunk_id": f"{source_name}_chunk_{idx}",
                "chunk_index": idx,
                "total_chunks": len(raw_chunks),
                "created_at": timestamp,
                "char_length": len(chunk)
            }
        )
        docs.append(doc)
    return docs

def get_chroma_collection(db_path: str = DEFAULT_DB_PATH, collection_name: str = DEFAULT_COLLECTION):
    """Retrieve or initialize persistent ChromaDB collection."""
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name=collection_name)

def ingest_document(
    file_source: Union[str, bytes, io.BytesIO],
    filename: str = "document",
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> Dict[str, Any]:
    """
    End-to-End Automated Ingestion Pipeline:
    1. Parse raw document (PDF or Text)
    2. Clean and structure text
    3. Chunk with metadata tracking
    4. Compute dense vector embeddings
    5. Upsert to ChromaDB vector store
    """
    start_time = time.time()
    
    # 1. Parsing
    if filename.lower().endswith(".pdf") or (isinstance(file_source, str) and file_source.lower().endswith(".pdf")):
        raw_text = extract_text_from_pdf(file_source)
    else:
        if isinstance(file_source, (bytes, bytearray)):
            raw_text = file_source.decode("utf-8", errors="replace")
        elif isinstance(file_source, io.BytesIO):
            raw_text = file_source.getvalue().decode("utf-8", errors="replace")
        elif isinstance(file_source, str) and os.path.exists(file_source):
            with open(file_source, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
        else:
            raw_text = str(file_source)
            
    # 2. Cleaning
    cleaned_content = clean_text(raw_text)
    if not cleaned_content:
        return {
            "success": False,
            "error": "Failed to extract or clean readable text from document.",
            "chunk_count": 0
        }
        
    # 3. Chunking
    chunks = chunk_document(
        cleaned_content,
        source_name=filename,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # 4. Embedding
    print(f"Generating embeddings for {len(chunks)} chunks from '{filename}'...")
    embedder = LocalMiniLMEmbeddings()
    texts = [c.page_content for c in chunks]
    embeddings = embedder.embed_documents(texts)
    
    # 5. ChromaDB Upsert
    collection = get_chroma_collection(db_path, collection_name)
    ids = [c.metadata["chunk_id"] for c in chunks]
    metadatas = [c.metadata for c in chunks]
    
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    elapsed = time.time() - start_time
    total_docs = collection.count()
    
    return {
        "success": True,
        "filename": filename,
        "chunk_count": len(chunks),
        "total_collection_count": total_docs,
        "elapsed_seconds": round(elapsed, 2),
        "db_path": db_path,
        "collection_name": collection_name
    }

def print_database_stats(db_path: str = DEFAULT_DB_PATH, collection_name: str = DEFAULT_COLLECTION):
    collection = get_chroma_collection(db_path, collection_name)
    count = collection.count()
    print("==================================================")
    print(" ChromaDB Vector Store Status")
    print("==================================================")
    print(f" Database Path    : {db_path}")
    print(f" Collection Name  : {collection_name}")
    print(f" Total Documents  : {count}")
    if count > 0:
        sample = collection.peek(1)
        print(f" Sample Metadata  : {sample['metadatas'][0] if sample['metadatas'] else 'None'}")
        snippet = sample['documents'][0][:120].replace('\n', ' ') if sample['documents'] else 'None'
        print(f" Sample Content   : {snippet}...")
    print("==================================================")

def main():
    parser = argparse.ArgumentParser(description="End-to-end automated document ingestion for DOH RAG system.")
    parser.add_argument("--file", type=str, help="Path to single PDF/TXT file to ingest.")
    parser.add_argument("--dir", type=str, help="Directory containing documents to ingest.")
    parser.add_argument("--stats", action="store_true", help="Display vector store document statistics.")
    parser.add_argument("--rebuild-doh", action="store_true", help="Re-ingest official DOH PDF manual from data/ folder.")
    
    args = parser.parse_args()
    
    if args.stats:
        print_database_stats()
        return
        
    if args.file:
        fname = os.path.basename(args.file)
        print(f"Initiating automated ingestion pipeline for: {args.file}")
        res = ingest_document(args.file, filename=fname)
        print(f"Result: {res}")
        return
        
    if args.dir:
        files = [f for f in os.listdir(args.dir) if f.lower().endswith((".pdf", ".txt", ".md"))]
        print(f"Found {len(files)} eligible files in {args.dir}")
        for f in files:
            p = os.path.join(args.dir, f)
            print(f"Ingesting: {f}")
            res = ingest_document(p, filename=f)
            print(f"  Processed {res['chunk_count']} chunks in {res['elapsed_seconds']}s")
        return
        
    if args.rebuild_doh or not any([args.file, args.dir, args.stats]):
        doh_pdf = os.path.join(BASE_DIR, "data", "7323175-Doh-Communicable-Diseases.pdf")
        if os.path.exists(doh_pdf):
            print(f"Automated ingestion pipeline starting for DOH manual: {doh_pdf}")
            res = ingest_document(doh_pdf, filename="7323175-Doh-Communicable-Diseases.pdf")
            if res.get("success"):
                print(f"Ingestion successful! Processed {res['chunk_count']} chunks. Total database items: {res['total_collection_count']}")
            else:
                print(f"Ingestion failed: {res.get('error')}")
        else:
            print(f"DOH PDF not found at {doh_pdf}")
        print_database_stats()

if __name__ == "__main__":
    main()
