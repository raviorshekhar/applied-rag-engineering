# Applied RAG Engineering

A hands-on RAG (Retrieval-Augmented Generation) system built from scratch — from PDF ingestion to grounded AI-generated answers, deployed via Docker and Kubernetes.

## What This Project Does

Takes DB2 technical documentation (troubleshooting, backup, restore guides), breaks it into searchable chunks, and answers natural-language questions grounded in that documentation — with a choice between cloud (Gemini) and local (Ollama/Llama 3.2) generation, accessible through a web interface.

## Tech Stack

- **Python** — core pipeline
- **sentence-transformers** (all-MiniLM-L6-v2) — embeddings
- **ChromaDB** — vector database
- **rank_bm25** — lexical/keyword search (BM25)
- **LangChain** (RecursiveCharacterTextSplitter) — document chunking
- **Google Gemini API** — cloud-based answer generation
- **Ollama** (Llama 3.2) — local, offline answer generation
- **Streamlit** — web interface
- **Docker** — containerization
- **Kubernetes** (via kind) — orchestration and deployment

## Key Features

- **Hybrid Search** — combines semantic (embedding-based) and lexical (BM25) retrieval for improved accuracy, especially for precise technical terms and commands
- **Grounded Generation** — answers are strictly based on retrieved source documents; the system explicitly says when it doesn't have enough information, rather than guessing
- **Multi-document support** — indexes multiple PDFs into a unified knowledge base (~420 chunks across 3 DB2 guides)
- **Evaluation framework** — tests retrieval quality against expected keywords for a set of known questions
- **Model choice** — users can choose between cloud (Gemini) and local (Ollama) generation directly from the UI
- **Containerized and orchestrated** — packaged with Docker and deployed on a local Kubernetes cluster (kind), with secrets management for API keys

## Key Finding: Cloud vs. Local Model Reliability

Running the exact same retrieved context through both Gemini and a local model (Llama 3.2, via Ollama) revealed a significant reliability gap. The local model produced confidently-worded but factually incorrect answers on multiple occasions — including inverting a core DB2 concept about how online vs. offline backups affect database availability, and misinterpreting a Hindi-language query. Gemini, given identical context, consistently acknowledged when information was insufficient rather than guessing.

**Takeaway:** Retrieval quality alone doesn't guarantee a reliable RAG system — the generation model's own reliability and tendency toward hallucination matter just as much, particularly for domain-critical applications where an incorrect answer could lead to a real operational mistake.

## Architecture

**Indexing (runs once):**
PDF documents → Chunking (LangChain) → Embeddings (sentence-transformers) → ChromaDB vector store

**Query time (runs per question):**
User question → Semantic search (ChromaDB) + Keyword search (BM25) → Combined, deduplicated context → LLM generation (Gemini or Ollama, user's choice) → Answer displayed in Streamlit

## How to Run

### Locally (Streamlit)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Requires a `.env` file with `GEMINI_API_KEY` set, and PDF files in the project directory.

### Via Docker
```bash
docker build -t db2-assistant .
docker run -p 8501:8501 --env-file .env db2-assistant
```

### Via Kubernetes (using kind)
```bash
kind create cluster --name db2-cluster
docker build -t db2-assistant .
kind load docker-image db2-assistant --name db2-cluster
kubectl create secret generic gemini-secret --from-env-file=.env
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl port-forward service/db2-assistant-service 8501:8501
```
Then open `http://localhost:8501`.

## Background

Experienced Db2 DBA trying my hands AI Engineering, as part of an ongoing M.Tech in AI & ML (BITS Pilani, WILP). Combines enterprise database and Kubernetes (CKA-certified) experience with hands-on GenAI engineering.

## What's Next

- Larger local models (7B+ parameters) to evaluate whether model size closes the reliability gap found in testing
- Document structure-based and semantic chunking strategies
- Expanding the knowledge base with additional DB2 documentation
- Exploring agentic capabilities — connecting the assistant to live DB2 diagnostic commands