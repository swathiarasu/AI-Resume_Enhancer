"""Manual end-to-end check — makes ONE real Gemini API call.

Not part of the automated smoke test (it needs a live key and costs tokens).
Run explicitly:

    export GEMINI_API_KEY=your-key
    python3 tests/e2e_check.py

Exercises the full RAG loop: FAISS retrieval -> prompt assembly -> Gemini.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from components.rag_model import RAGModel
from components.retriever_multi import MultiPDFRetriever


def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set. Run: export GEMINI_API_KEY=your-key")
        sys.exit(2)

    question = "How can a Python developer with 3 years of experience strengthen their resume for an ML role?"

    print("1) Retrieving context from FAISS index...")
    retriever = MultiPDFRetriever()
    results = retriever.get_top_k(question, k=3)
    context = "\n".join(chunk_text for _, _, chunk_text in results)
    print(f"   retrieved {len(results)} chunks ({len(context)} chars)\n")

    print("2) Calling Gemini...")
    answer = RAGModel().generate_answer(question, context)

    if answer.strip().startswith("Gemini API error"):
        print("   FAILED:", answer)
        sys.exit(1)

    print("   OK — model returned an answer.\n")
    print("=" * 60)
    print("Q:", question)
    print("-" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
