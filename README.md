# python_rag_pdf — Local, Private RAG Chat over PDFs

Chat with your PDF documents using a **fully local RAG pipeline** — no cloud APIs, no data leaving your machine. Streamlit UI, LangChain orchestration, local Mistral via Ollama.

## Why local?

Most "chat with your docs" demos ship your documents to a hosted LLM. This one doesn't: embeddings, vector search, and generation all run on your machine. Useful for anything you wouldn't paste into a cloud chatbot — contracts, internal docs, research drafts.

## Architecture

```
PDF ──► PyPDFLoader ──► RecursiveCharacterTextSplitter (1024 chars, 100 overlap)
                                   │
                                   ▼
                     FastEmbed embeddings ──► Chroma vector store
                                                     │
User question ──► similarity retrieval (k=3, score ≥ 0.5) ──► prompt template
                                                     │
                                                     ▼
                                     Mistral (local, via Ollama) ──► answer
```

See `drawio_diagram.xml` for the full diagram (open with [draw.io](https://app.diagrams.net/)).

- **`rag.py`** — the pipeline: ingestion (load → chunk → embed → store) and the LangChain runnable chain (retriever → prompt → model → parser). Retrieval is thresholded similarity search, so weak matches are dropped instead of hallucinated over.
- **`app.py`** — Streamlit chat UI: multi-PDF upload, ingestion spinner, chat history via `streamlit_chat`.

## Run it

```bash
# 1. Local model runtime
brew install ollama          # or: https://ollama.com/download
ollama pull mistral

# 2. Python deps
pip install -r requirements.txt

# 3. Go
streamlit run app.py
```

Upload one or more PDFs, wait for ingestion, ask questions. Answers are grounded in retrieved chunks; when nothing relevant is found, the model is instructed to say it doesn't know.

## Stack

Python · Streamlit · LangChain · Ollama (Mistral) · ChromaDB · FastEmbed · PyPDF

## Notes / possible extensions

- Swap `mistral` for any Ollama model (`ChatOllama(model=...)` in `rag.py`).
- Persist the Chroma store to disk for reuse across sessions.
- Add an evaluation harness (retrieval hit-rate, answer faithfulness) — the natural next step for production use.
