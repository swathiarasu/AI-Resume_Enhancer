
import streamlit as st
import fitz  # For PDF reading
import torch
from components.rag_model import RAGModel
from components.retriever_multi import MultiPDFRetriever  # For FAISS-based search
from components.retriever_live import LiveRetriever      # For uploaded document search

# Force CPU to avoid MacOS GPU issues (MPS)
torch.backends.mps.is_available = lambda: False
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reject uploads larger than this to protect memory / extraction time.
MAX_UPLOAD_MB = 10

# Extract text from uploaded file (PDF or .txt)
def extract_text(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    elif uploaded_file.name.lower().endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    else:
        return ""

# Main Streamlit App
def main():
    st.set_page_config(page_title="AI-Powered Resume Enhancer")
    st.title("PerfectPitch CV - Your AI-Driven Resume Enhancer")

    # Load retrievers and model
    with st.spinner("Loading index and model..."):
        faiss_retriever = MultiPDFRetriever(
            index_path="data/index/faiss_index.bin",
            metadata_path="data/index/metadata.pkl"
        )
        live_retriever = LiveRetriever()
        rag = RAGModel()

    # File uploader
    uploaded_file = st.file_uploader("Optionally upload a PDF or .txt file", type=["pdf", "txt"])

    # User query input
    question = st.text_input("Enter your query for Resume Enhancement")

    if question and question.strip():
        with st.spinner("Retrieving and generating answer..."):
            # Search FAISS index
            faiss_results = faiss_retriever.get_top_k(question, k=3)

            # Search uploaded file
            upload_context = ""
            live_results = []
            if uploaded_file:
                size_mb = uploaded_file.size / (1024 * 1024)
                if size_mb > MAX_UPLOAD_MB:
                    st.warning(
                        f"Uploaded file is {size_mb:.1f} MB, over the {MAX_UPLOAD_MB} MB limit. "
                        "Skipping it and answering from the resume index only."
                    )
                else:
                    doc_text = extract_text(uploaded_file)
                    if not doc_text.strip():
                        st.warning(
                            "Could not extract any text from the uploaded file. "
                            "If it's a scanned/image-only PDF, it isn't supported. "
                            "Answering from the resume index only."
                        )
                    else:
                        live_results = live_retriever.get_top_k_from_text(doc_text, question, k=3)
                        upload_context = "\n".join([x[1] for x in live_results])

            # Merge all context
            faiss_context = "\n".join([chunk[2] for chunk in faiss_results])
            full_context = faiss_context + "\n" + upload_context

            # Generate answer
            answer = rag.generate_answer(question, full_context)

            # Display output
            st.subheader("Answer")
            st.markdown(answer)

            with st.expander("📚 Sources from FAISS index"):
                for i, (title, chunk_id, chunk_text) in enumerate(faiss_results):
                    st.markdown(f"**[{i+1}] {title}** (chunk {chunk_id})\n\n{chunk_text}")

            if uploaded_file:
                with st.expander("📄 Sources from Uploaded File"):
                    for i, (chunk_id, chunk_text) in enumerate(live_results, start=len(faiss_results)+1):
                        st.markdown(f"**[{i}] Uploaded Doc (chunk {chunk_id})**\n\n{chunk_text}")

if __name__ == "__main__":
    main()
