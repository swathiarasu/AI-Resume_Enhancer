# PerfectPitch CV - Your AI-Driven Resume Enhancer

This project is an AI-powered Resume Enhancer using a Retrieval-Augmented Generation (RAG) model. It allows users to upload their resumes and receive personalized enhancement suggestions using a hybrid approach of FAISS-based retrieval and generative AI (Gemini API).
## Folder Structure

```
bash
├── app.py                         # Streamlit frontend for QA
├── build_index.py                 # Builds FAISS index from resume data
├── download_corpus.py             # Downloads dataset from Kaggle
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker build configuration
├── entrypoints/
│   └── run_app.sh                 # Script to launch the Streamlit app
│   └── build_index.sh             # Script to build FAISS index
│   └── download_corpus.sh         # Script to download the dataset
├── .kaggle/
│   └── kaggle.json                # Kaggle API credentials (place manually or mount via Docker)
├── data/
│   └── dataset.csv                # Downloaded dataset
│   └── index/
│       ├── faiss_index.bin        # FAISS vector index
│       └── metadata.pkl           # Metadata for the chunks
└── components/
    ├── rag_model.py               # RAG generation logic (Gemini API)
    ├── retriever_multi.py         # Retriever using prebuilt FAISS index
    └── retriever_live.py         # Retriever for live-uploaded PDFs
└── README.md                      # Project overview and setup instructions
```

## Steps to run this project

1. To spawn a container
   > docker build -t rag-app .

2. To downloading the corpus
   > GPU: `docker run --gpus all -v ./.kaggle:/root/.kaggle --entrypoint bash rag-app entrypoints/download_corpus.sh`
   >
   > CPU: `docker run -v ./.kaggle:/root/.kaggle --entrypoint bash rag-app entrypoints/download_corpus.sh`

3. To get the vector embeddings of the corpus
   > GPU: `docker run --gpus all --entrypoint bash rag-app entrypoints/build_index.sh`
   >
   > CPU: `docker run --entrypoint bash rag-app entrypoints/build_index.sh`

4. To run the app
   > GPU: `docker run -d --gpus all -p 8501:8501 rag-app`
   >
   > CPU: `docker run -d -p 8501:8501 rag-app`

5. To access the app, open your browser and go to: `http://<external-ip>:8501`


##  Usage
- Upload your resume text in PDF or TXT format.
- Ask for personalized suggestions for improving your resume.
- Receive actionable advice based on your profile and target job role.

