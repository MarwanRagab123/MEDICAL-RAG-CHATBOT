# Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) medical chatbot that uses a local FAISS vector store and a Groq-powered LLM to answer medical questions from a medical encyclopedia dataset.

## Features
- PDF ingestion and chunking
- FAISS vectorstore for fast retrieval
- RAG pipeline that constructs context and queries an LLM
- Simple web UI (Flask)

## Requirements
- Python 3.10+
- (Optional) Docker

## Quickstart (local)
1. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add environment variables (do NOT commit this file):

Create a `.env` file in the project root with these keys (replace placeholders):

```
HUGGINGFACE_API_TOKEN=your_hf_token_here
GROQ_API_KEY=your_groq_api_key_here
```

4. Make sure vectorstore exists or build it from PDFs in `data/`:

- To build the vectorstore from PDFs (loads files from `data/` and saves into `vectorstores/db_faiss`):

```bash
python -c "from app.components.pdf_loader import pdf_loader, pdf_splitter; from app.components.vector_store import create_vectordb; docs = pdf_loader(); chunks = pdf_splitter(docs); create_vectordb(chunks)"
```

5. Run the Flask app (app package exposes `app`):

```bash
# from project root
python -m flask run
# or
python -m app.main
```

Open http://127.0.0.1:5000

## Docker (optional)
If you have a `Dockerfile` in the project root, build and run:

```bash
docker build -t medical-rag-chatbot .
docker run -p 5000:5000 --env-file .env medical-rag-chatbot
```

## Environment & Secrets
- Never commit `.env` or secrets to the repo. Add `.env` to `.gitignore` (already recommended).
- If GitHub blocked a push due to secrets, remove them from history or follow GitHub instructions for blocked secrets.

## Troubleshooting
- If the app says "No relevant documents found":
  - Ensure `vectorstores/db_faiss` exists and contains `index.faiss` and `index.pkl`.
  - Rebuild vectorstore from PDFs in `data/` using the build command above.
  - Increase retrieval `k` or `fetch_k` in `app/components/llm.py` if results are too narrow.
- If FAISS fails to load, check CPU support and that FAISS was built with required optimizations.
- If downloading HuggingFace models is slow or rate-limited, set `HUGGINGFACE_API_TOKEN`.
- Check logs in `logs/` for detailed errors.

## Useful Commands
- Build vectorstore: see step 4 above
- Clear flask session history: use the web UI "Clear" button or call `/clear-history`

## Project Structure
- `app/` — Flask app, components (LLM, embeddings, loaders)
- `data/` — place PDFs here to index
- `vectorstores/db_faiss/` — FAISS index files
- `logs/` — runtime logs

---