import pandas as pd
import os
import faiss
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Configuration
CSV_FILE = "data/AI_Resume_Screening.csv"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
OUTPUT_DIR = "data/index"

# Chunking logic
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# Load JSONL-formatted dataset
print("🔍 Loading dataset...")
dataset = pd.read_csv(CSV_FILE)


# Initialize embedder
print(" Embedding chunks...")
model = SentenceTransformer(EMBEDDING_MODEL)
documents = []
metadata = []

# Chunk and collect metadata with progress
# for paper in tqdm(dataset, desc="📄 Chunking abstracts"):
#     title = paper.get("title", "Untitled")
#     abstract = paper.get("abstract", "")
#     if abstract:
#         chunks = chunk_text(abstract)
#         documents.extend(chunks)
#         metadata.extend([(title, i) for i in range(len(chunks))])
for idx, row in tqdm(dataset.iterrows(), desc="📄 Chunking resumes"):
    text = f"Skills: {row['Skills']}\nExperience: {row['Experience (Years)']} years\nEducation: {row['Education']}\nCertifications: {row['Certifications']}\nJob Role: {row['Job Role']}\nRecruiter Decision: {row['Recruiter Decision']}\nSalary Expectation: ${row['Salary Expectation ($)']}\nProjects Count: {row['Projects Count']}\nAI Score: {row['AI Score (0-100)']}"
    if text:
        chunks = chunk_text(text)
        documents.extend(chunks)
        metadata.extend([(row["Name"], i) for i in range(len(chunks))])


# Generate embeddings with progress
print(f"📚 Generating embeddings for {len(documents)} chunks...")
embeddings = []
for chunk in tqdm(documents, desc="🔢 Embedding text"):
    embeddings.append(model.encode(chunk, convert_to_numpy=True))
embeddings = np.stack(embeddings)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Save index and metadata
os.makedirs(OUTPUT_DIR, exist_ok=True)
faiss.write_index(index, os.path.join(OUTPUT_DIR, "faiss_index.bin"))
with open(os.path.join(OUTPUT_DIR, "metadata.pkl"), "wb") as f:
    pickle.dump((documents, metadata), f)

print(f" Saved FAISS index and metadata to `{OUTPUT_DIR}`")
