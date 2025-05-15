from sentence_transformers import SentenceTransformer
import numpy as np

class LiveRetriever:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def chunk_text(self, text, chunk_size=500, overlap=50):
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

    def get_top_k_from_text(self, text, query, k=3):
        chunks = self.chunk_text(text)
        chunk_embeddings = self.model.encode(chunks, convert_to_numpy=True)
        query_vec = self.model.encode([query])[0]
        scores = np.dot(chunk_embeddings, query_vec)  # cosine similarity
        top_k_idx = np.argsort(scores)[::-1][:k]
        return [(i, chunks[i]) for i in top_k_idx]
