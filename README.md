# Mini RAG System — Movie Plots 🎬

A lightweight, production-grade Retrieval-Augmented Generation (RAG) system built with **LangChain**, **FAISS**, **BM25**, and **Google Gemini** that answers questions over movie plot summaries and returns validated, grounded JSON outputs.

---

## 🌟 System Architecture

### 1. Ingestion & Indexing Pipeline

```mermaid
flowchart LR
    A["📁 1. Data Source<br/>Movie Dataset CSV"] --> B["🧹 2. Data Preprocessing<br/>Load, clean & construct searchable text"]
    B --> C["✂️ 3. Document Chunking<br/>Recursive text splitting (~300 words + metadata)"]
    
    C --> H["🔢 4a. Dense Vector Embedding<br/>gemini-embedding-001"]
    C --> I["🔤 4b. Sparse Keyword Indexing<br/>BM25 text tokenization"]
    
    H --> J[("🗄️ Dense Index<br/>FAISS Vector Store")]
    I --> K[("🗄️ Sparse Index<br/>BM25 Retriever")]

    classDef stage fill:#eef2ff,stroke:#4f46e5,stroke-width:1.5px
    classDef store fill:#ecfdf5,stroke:#059669,stroke-width:2px

    class A,B,C,H,I stage
    class J,K store
```

---

### 2. RAG Query & Generation Pipeline

```mermaid
flowchart LR
    Q["👤 User Question"] --> QA["🤖 1. Query Analysis<br/>Scope check & metadata filter extraction"]
    QA --> HY["💡 2. Query Expansion (HyDE)<br/>Hypothetical passage generation"]
    
    HY --> DS["🎯 3a. Dense Search (FAISS)<br/>Top 15 semantic vector matches"]
    HY --> SS["🔤 3b. Sparse Search (BM25)<br/>Top 15 keyword term matches"]
    
    DS --> RRF["🔀 4. Hybrid Fusion (RRF)<br/>Merge dense + sparse rank lists"]
    SS --> RRF
    
    RRF --> GEN["✨ 5. Gemini Generation & Grounding<br/>Synthesize answer & verify contexts"]
    GEN --> OUT["📦 6. Final Structured Response<br/>Answer, contexts & reasoning JSON"]

    classDef stage fill:#eef2ff,stroke:#4f46e5,stroke-width:1.5px
    classDef output fill:#ecfdf5,stroke:#059669,stroke-width:2px

    class Q,QA,HY,DS,SS,RRF,GEN stage
    class OUT output
```

---

## 🚀 Key Features

* **Hybrid Retrieval (Dense + Sparse)**: Combines **FAISS** (dense vector similarity for semantic understanding) and **BM25** (sparse keyword search for exact actor names, titles, and proper nouns).
* **Reciprocal Rank Fusion (RRF)**: Merges ranked results from dense and sparse search without score scale mismatches.
* **HyDE (Hypothetical Document Embeddings)**: Generates a hypothetical plot passage for dense embedding search to bridge the phrasing gap between questions and plot summaries.
* **Query Routing & Metadata Filtering**: Analyzes query scope and extracts filters for `Genre` and `Release Year` before retrieval. Out-of-corpus queries short-circuit early.
* **Word-Level Chunking**: Splits plot summaries into 300-word chunks (50-word overlap) along natural sentence/paragraph boundaries.
* **Actor & Title Indexing**: Enriches text payloads with `Title` and `Cast` metadata so queries targeting specific actors (e.g. *"Michael Rennie"*, *"Denzel Washington"*) hit exact keyword matches.
* **Strict Grounded Output**: Uses Pydantic models (`RAGAnswer`) to enforce validated JSON schema (`answer`, `contexts`, `reasoning`) grounded strictly in retrieved context.

---

## 📋 Structured JSON Output Schema

The system outputs a verified JSON payload conforming to the following structure:

```json
{
  "answer": "The movie 'The Day the Earth Stood Still' (1951) features an artificial intelligence robot named Gort and Klaatu who land in Washington, D.C.",
  "contexts": [
    "Title: The Day the Earth Stood Still\nCast: Michael Rennie, Patricia Neal\nPlot: When a flying saucer lands in Washington, D.C., the Army quickly surrounds it. A humanoid (Michael Rennie) emerges..."
  ],
  "reasoning": "The retrieved context explicitly confirms the 1951 film featuring Klaatu and Gort landing in Washington D.C."
}
```

---

## 🛠️ Project Structure

```text
Team D/
├── data/
│   └── wiki_movie_plots_deduped.csv   # Kaggle Movie Plots dataset (downloaded locally)
├── rag_movie_plots_langchain.ipynb     # Main RAG notebook implementation
├── ingestion_pipeline.mermaid          # Ingestion Architecture Diagram
├── query_pipeline.mermaid              # Query Architecture Diagram
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
└── .env                                # Environment config
```

---

## 📦 Getting Started

### Prerequisites

* Python **3.10+**
* Google Gemini API Key or GCP Vertex AI configuration.

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone git@github.com:Thiwanka-Sandakalum/Mini-RAG-System-Movie-Plots.git
cd Mini-RAG-System-Movie-Plots
pip install -r requirements.txt
```

### 2. Dataset Setup

Download the [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) dataset from Kaggle, create the `data/` directory, and place the CSV file inside it named `wiki_movie_plots_deduped.csv`:

```bash
mkdir -p data
# Place downloaded CSV in the data/ folder and ensure it is named:
# data/wiki_movie_plots_deduped.csv
```

### 3. Environment Setup

Set up your Gemini / Google Cloud environment variables in a `.env` file or terminal session:

```bash
# Option A: Google Gemini API Key
export GEMINI_API_KEY="your-api-key-here"

# Option B: Google Cloud Vertex AI
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

### 4. Running the Pipeline

Open and run `rag_movie_plots_langchain.ipynb` using Jupyter Notebook or VS Code:

```bash
jupyter notebook rag_movie_plots_langchain.ipynb
```

Execute cells sequentially to:
1. Load dataset & preprocess 300 movie plot samples.
2. Build FAISS and BM25 search indexes.
3. Test demo queries across semantic, hybrid actor matching, self-query filtering, and out-of-scope cases.

---

## 🎯 Architecture Summary

| Component | Choice / Metric |
| :--- | :--- |
| **LLM** | `gemini-2.5-flash` |
| **Embeddings** | `gemini-embedding-001` (Asymmetric Document/Query tasks) |
| **Vector Store** | In-memory `FAISS` |
| **Keyword Index** | `BM25Retriever` (Rank-BM25) |
| **Fusion Metric** | Reciprocal Rank Fusion ($k=60$) |
| **Chunking** | `RecursiveCharacterTextSplitter` ($300\text{ words}$, $50\text{ overlap}$) |
| **Output Format** | Pydantic validated JSON (`RAGAnswer`) |
