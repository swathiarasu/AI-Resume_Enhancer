from sentence_transformers import SentenceTransformer
import numpy as np

# bge-v1.5 retrieval instruction — applied to QUERIES ONLY, never to passages.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"

class LiveRetriever:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def chunk_text(self, text, chunk_size=500, overlap=50):
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

    def get_top_k_from_text(self, text, query, k=3):
        chunks = self.chunk_text(text)
        chunk_embeddings = self.model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        query_vec = self.model.encode(
            [f"{QUERY_INSTRUCTION} {query}"], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        scores = np.dot(chunk_embeddings, query_vec)  # cosine similarity (both sides normalized)
        top_k_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), chunks[i]) for i in top_k_idx]
