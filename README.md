# Football Tactics Q&A

A RAG (Retrieval-Augmented Generation) application that answers questions about football formations, tactical concepts, match analysis, and playing styles.

Built as a project for [LLM Zoomcamp 2026](https://github.com/DataTalksClub/llm-zoomcamp) by DataTalks.Club.


## Problem Statement

Football tactics are discussed across hundreds of YouTube videos, match reports, Wikipedia articles, and blogs. Fans and analysts often need quick, accurate answers to tactical questions like "What is gegenpressing?", "How did Ancelotti set up Real Madrid's midfield?", or "Why do teams play with inverted full-backs?"

This application builds a knowledge base from three sources, then uses hybrid search with re-ranking and an LLM to answer tactical questions with proper context and citations.


## Data Sources

| Source | Documents | Description |
|---|---|---|
| StatsBomb Open Data | ~200 | Match narratives from FIFA WC 2022, Euro 2020, La Liga, Premier League (free, open data) |
| YouTube Transcripts | ~50 | Tifo Football's "Tactics Explained" series via yt-dlp |
| Wikipedia | ~50 | Tactical concepts, formations, player roles, manager profiles |

Total: ~300 documents, ~720 chunks after splitting.

Data is auto-collected by the ingestion pipeline. A pre-built `data/processed/documents.json` is included for reproducibility.


## Architecture

```
User Question
    |
    v
Query Rewriting (LLM expands abbreviations, adds tactical terms)
    |
    v
Hybrid Search (BM25 + kNN cosine via Elasticsearch, merged with RRF)
    |
    v
Cross-Encoder Re-ranking (ms-marco-MiniLM-L-6-v2)
    |
    v
LLM Generation (Groq llama-3.1-8b-instant, free tier)
    |
    v
Auto-Judge (LLM evaluates relevance) + Log to PostgreSQL
    |
    v
Answer + Sources displayed in Streamlit UI
```


## Tech Stack

| Component | Tool |
|---|---|
| LLM | Groq (llama-3.1-8b-instant) or OpenAI (gpt-4o-mini) |
| Embeddings | sentence-transformers (multi-qa-MiniLM-L6-cos-v1, 384 dims) |
| Search | Elasticsearch 8.9 (BM25 + dense vector) |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| UI | Streamlit |
| Ingestion | Prefect (orchestration) |
| Monitoring | Grafana + PostgreSQL |
| Containerization | Docker Compose |


## How to Run

### Prerequisites

- Docker and Docker Compose
- A free Groq API key from [console.groq.com](https://console.groq.com)
- `yt-dlp` installed (for transcript extraction): `pip install yt-dlp` or `brew install yt-dlp`

### Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/nikhilkhinchi/football-tactics-qa.git
cd football-tactics-qa
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 2. Start all services
docker-compose up -d

# 3. Open the app
# Streamlit UI: http://localhost:8501
# Grafana dashboard: http://localhost:3000 (admin/admin)
```

The app container runs the ingestion pipeline on first start, then launches Streamlit.

### Local Development

```bash
# Install dependencies
uv sync

# Start infrastructure
docker-compose up -d elasticsearch postgres grafana

# Run ingestion
python -m ingestion.prefect_flow --no-prefect

# Start Streamlit
streamlit run app/streamlit_app.py
```


## Evaluation Results

### Retrieval Evaluation

Compared 5 retrieval approaches on 240 ground truth Q&A pairs.

| Approach | Hit Rate | MRR |
|---|---|---|
| text_only (BM25) | 0.579 | 0.447 |
| hybrid_linear | 0.617 | 0.466 |
| vector_only (kNN) | 0.733 | 0.556 |
| hybrid_rrf | 0.779 | 0.592 |
| **hybrid_rrf_reranked** | **0.808** | **0.668** |

The hybrid RRF approach with cross-encoder re-ranking performed best, improving hit rate by 39% over text-only search.

### RAG Evaluation

Compared 3 prompt templates using LLM-as-judge on 30 questions (% answers rated RELEVANT).

| Model | Prompt | % Relevant |
|---|---|---|
| llama-3.1-8b-instant | basic | 80.0% |
| llama-3.1-8b-instant | **expert** | **86.7%** |
| llama-3.1-8b-instant | structured | 86.7% |

The **expert** prompt template with `llama-3.1-8b-instant` was selected for the production pipeline. It scored 86.7% relevance with the best balance of accuracy and conciseness.

The 70b model hit Groq's free-tier daily token limit during evaluation. Its partial results (basic: 56.7% on 18/30 questions) suggest it performs well when not rate-limited, but the 8b model is the better fit for a free-tier deployment.

Full results in `data/rag_eval_results.csv`.


## Monitoring

User feedback (thumbs up/down) is collected in the Streamlit UI and stored in PostgreSQL.

Grafana dashboard at `localhost:3000` with 7 panels:

1. Response time over time
2. Relevance distribution (pie chart)
3. Token usage over time
4. Cumulative cost
5. User feedback summary
6. Model usage breakdown
7. Retrieval method distribution


## Project Structure

```
football-tactics-qa/
├── app/                    # RAG pipeline, search, UI, monitoring
│   ├── streamlit_app.py    # Streamlit interface
│   ├── rag.py              # Main RAG orchestration
│   ├── search.py           # Text, vector, hybrid, RRF search
│   ├── reranker.py         # Cross-encoder re-ranking
│   ├── query_rewriter.py   # LLM query expansion
│   ├── judge.py            # LLM-as-judge auto-evaluation
│   ├── db.py               # PostgreSQL logging
│   └── llm_client.py       # Groq/OpenAI abstraction
├── ingestion/              # Data collection and indexing
│   ├── prefect_flow.py     # Prefect-orchestrated pipeline
│   ├── collect_statsbomb.py
│   ├── collect_transcripts.py
│   ├── collect_wiki.py
│   ├── chunk.py
│   ├── embed.py
│   └── index_elasticsearch.py
├── monitoring/             # Grafana provisioning + DB schema
├── scripts/                # Evaluation and ground truth generation
├── data/                   # Processed documents and eval results
├── docker-compose.yml
├── Dockerfile
└── Makefile
```


## Evaluation Criteria Mapping

| Criterion | Points | Where to find it |
|---|---|---|
| Problem description | 2 | This README |
| Retrieval flow (KB + LLM) | 2 | `app/rag.py`, `app/search.py` |
| Retrieval evaluation (multiple approaches) | 2 | `scripts/eval_retrieval.py`, `data/retrieval_eval_results.csv` |
| LLM evaluation (multiple prompts/models) | 2 | `scripts/eval_rag.py`, `data/rag_eval_results.csv` |
| Interface | 2 | `app/streamlit_app.py` (Streamlit UI) |
| Ingestion pipeline (automated) | 2 | `ingestion/prefect_flow.py` (Prefect) |
| Monitoring (feedback + dashboard) | 2 | `app/db.py`, `monitoring/grafana/` (7 panels) |
| Containerization | 2 | `docker-compose.yml`, `Dockerfile` |
| Reproducibility | 2 | This README, `data/processed/documents.json` committed |
| Hybrid search | 1 | `app/search.py` (BM25 + vector + RRF) |
| Re-ranking | 1 | `app/reranker.py` (cross-encoder) |
| Query rewriting | 1 | `app/query_rewriter.py` (LLM expansion) |


## Acknowledgments

- [StatsBomb](https://statsbomb.com/) for free open match data
- [Tifo Football](https://www.youtube.com/@Tifo) for tactical analysis content
- [DataTalks.Club](https://datatalks.club/) for the LLM Zoomcamp course
