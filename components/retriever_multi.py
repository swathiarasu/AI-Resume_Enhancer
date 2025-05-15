import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class MultiPDFRetriever:
    def __init__(self, index_path="data/index/faiss_index.bin", metadata_path="data/index/metadata.pkl"):
        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        with open(metadata_path, "rb") as f:
            self.documents, self.metadata = pickle.load(f)

    def get_top_k(self, query, k=3):
        query_vec = self.model.encode([query])[0]
        D, I = self.index.search(np.array([query_vec]), k)
        results = []
        for idx in I[0]:
            doc_text = self.documents[idx]
            filename, chunk_id = self.metadata[idx]
            results.append((filename, chunk_id, doc_text))
        return results
