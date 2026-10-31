import os
import torch
from typing import List
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings

class LocalMiniLMEmbeddings(Embeddings):
    """
    LangChain-compatible Embeddings wrapper using sentence-transformers/all-MiniLM-L6-v2
    via HuggingFace Transformers with mean pooling.
    
    Ensures 100% vector space consistency with the existing ChromaDB embeddings
    while integrating seamlessly with LangChain vectorstores and retrievers.
    """
    _instance = None

    def __new__(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Singleton pattern to prevent loading the model multiple times in memory
        if cls._instance is None:
            cls._instance = super(LocalMiniLMEmbeddings, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if getattr(self, "_initialized", False):
            return
        self.model_name = model_name
        # Set offline/cache preferences if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._initialized = True

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed a list of text documents."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embs = self._mean_pooling(outputs, inputs["attention_mask"])
                all_embeddings.extend(batch_embs.cpu().tolist())

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self.embed_documents([text])[0]

