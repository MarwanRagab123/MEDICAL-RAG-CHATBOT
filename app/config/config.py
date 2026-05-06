import os

HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DATA_PATH = "data/"
DP_FAISS_PATH = "vectorstores/db_faiss"
EMBEDDING_REPO = "sentence-transformers/all-MiniLM-L6-v2"