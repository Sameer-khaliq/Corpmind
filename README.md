# CorpMind

Multi-supplier catalog reconciliation and enrichment agent. CorpMind ingests raw product feeds from multiple suppliers, resolves cross-supplier duplicates, fills missing attributes via web-search-grounded enrichment, and outputs a reconciled, audit-traceable catalog — with every automated decision gated by an evaluation step before it's accepted.

E-commerce sellers who source from multiple suppliers end up with structurally inconsistent catalog data (different field names, formats, and missing attributes). This is normally fixed by hand in spreadsheets, which stops scaling past a few hundred SKUs across three or more supplier feeds. CorpMind automates that reconciliation pipeline end-to-end.

## Core Features

- **Multi-format ingestion** — CSV/XLSX supplier feeds, normalized into a fixed product schema (title, brand, category, color, material, size, price, SKU, description)
- **Cross-supplier duplicate matching** — hybrid search (BM25 + dense embeddings + RRF) with connected-components clustering for intra-batch duplicates
- **Web-grounded enrichment** — ReAct tool-calling agent fills missing/low-confidence fields via live web search, with every filled value traceable to a source URL
- **Evaluation gate** — RAGAS-style faithfulness scoring on every accepted match and enrichment before it's allowed into the final catalog; nothing is auto-published below the configured threshold
- **Adversarial injection defense** — deterministic regex-based marker gate on enrichment sources, closes a source to 0.0 faithfulness if a prompt-injection marker is found anywhere in it, checked before any LLM call
- **Human review queue** — anything below the evaluation threshold, or genuinely ambiguous, is routed for manual review instead of guessed
- **Full audit trail** — every merge, enrichment, and flag decision is logged with which agent made it and why
- **Async, concurrency-capped orchestration** — batch runner combines LangGraph's native concurrency with a token-bucket rate limiter to stay within provider rate limits
- **Streamlit demo UI** — upload a feed, watch the pipeline run, review flagged items
- **Dockerized deployment** — single-tenant demo container, deployable to Hugging Face Spaces

## Architecture

```
Ingestion Node
      |
      v
Extraction & Normalization Agent (LLM)
      |
      v
Matching / Dedup Agent (hybrid search: BM25 + dense + RRF)
      |
      +-- MATCHED_EXISTING --> Enrichment Agent (web-search tool-calling) --> Evaluation Agent
      |
      +-- NEW_PRODUCT -------------------------------------------------> Evaluation Agent
      |
      +-- AMBIGUOUS ---------------------------------------------------> Evaluation Agent
                                                                                |
                                                              +-----------------+------------------+
                                                              |                                     |
                                                       Above threshold                        Below threshold
                                                       Report Agent                          Human Review Queue
```

Orchestrated as a LangGraph state machine with `Send`-based fan-out per item and a single batch-level accumulator for graph state. A dedicated batch runner layers concurrency-capping and rate-limiting on top of the graph's native async execution.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Orchestration | LangGraph |
| Schema / Validation | Pydantic |
| LLM Providers | Groq (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`), Google Gemini (`gemini-2.5-flash`) |
| Web Search / Grounding | Tavily |
| Vector Store | ChromaDB (persistent client) |
| Retrieval | Hybrid search — BM25 + dense embeddings + Reciprocal Rank Fusion |
| UI | Streamlit |
| Tracing | LangSmith (optional — only active when `LANGCHAIN_TRACING_V2=true`) |
| Package Management | uv |
| Containerization | Docker, Docker Compose |
| Deployment Target | Hugging Face Spaces |

## Repository Structure

```
corpmind/
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env
├── config/
│   ├── taxonomy.yaml
│   └── rate_limits.yaml
├── ui/
│   └── streamlit_app.py
├── src/
│   └── corpmind/
│       ├── config.py
│       ├── logging_config.py
│       ├── tracing_config.py
│       ├── agents/
│       │   ├── ingestion.py
│       │   ├── extraction.py
│       │   ├── matching.py
│       │   ├── enrichment.py
│       │   └── evaluation.py
│       ├── retrieval/
│       │   └── vector_store.py
│       ├── schemas/
│       │   ├── raw.py
│       │   ├── extraction.py
│       │   ├── matching.py
│       │   ├── enrichment.py
│       │   ├── evaluation.py
│       │   ├── consistent.py
│       │   ├── audit.py
│       │   └── state.py
│       ├── graph/
│       │   ├── nodes.py
│       │   ├── edges.py
│       │   └── build_graph.py
│       ├── orchestration/
│       │   └── batch_runner.py
│       └── reporting/
│           └── audit_report.py
├── scripts/
│   ├── load_test.py
│   ├── debug_extraction.py
│   └── verify_extraction_day4.py
├── tests/
│   └── unit/
│       ├── test_enrichment.py
│       └── ...
└── data/
    └── chroma/          # ChromaDB persistent store (gitignored, Docker volume)
```

> Some paths above (e.g. `agents/extraction.py`, `graph/`, `orchestration/`) reflect the module layout as understood from project history — verify against your actual tree before treating this as final.

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- Docker + Docker Compose (for containerized runs)
- API keys: Groq, Google (Gemini), Tavily

## Local Setup

```bash
# Clone and enter the project
cd corpmind

# Install dependencies (creates .venv automatically)
uv sync

# Copy and fill in environment variables
cp .env.example .env    # if you keep a template; otherwise create .env directly

# Run the test suite
uv run pytest

# Launch the Streamlit UI
uv run streamlit run ui/streamlit_app.py
```

Adding a new dependency:

```bash
uv add <package-name>
```

Never install packages with plain `pip install` in this project — anything installed outside `uv add` won't be registered in `pyproject.toml` and will break the Docker build.

## Docker Deployment

Build the image:

```bash
docker compose build
```

Run the container:

```bash
docker compose up
```

The Streamlit UI will be available at `http://localhost:8501`.

Stop the container:

```bash
docker compose down
```

The ChromaDB store is persisted via a named Docker volume (`chroma_data`), mapped to `/app/data/chroma` inside the container — it survives container restarts and rebuilds. API keys are injected at runtime via `env_file: .env` in `docker-compose.yml` and are never baked into the image.

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key — primary LLM provider for extraction, matching disambiguation, evaluation |
| `GOOGLE_API_KEY` | Gemini API key — embeddings and fallback LLM calls |
| `TAVILY_API_KEY` | Tavily API key — web search for enrichment grounding |
| `CHROMA_DB_PATH` | Path to the ChromaDB persistent store (default: `./data/chroma`) |
| `ENVIRONMENT` | `development` / `production` |
| `LOG_LEVEL` | Logging verbosity (e.g. `INFO`, `DEBUG`) |
| `LANGCHAIN_TRACING_V2` | Optional — set `true` to enable LangSmith tracing |

## Pipeline Stages (Reference)

| Stage | Responsibility |
|---|---|
| Ingestion | Load raw CSV/XLSX supplier feeds |
| Extraction | Normalize free-text/structured fields into the fixed product schema via LLM, with per-field source provenance |
| Matching | Hybrid search (BM25 + dense + RRF) + connected-components clustering to detect cross-supplier and intra-batch duplicates |
| Enrichment | ReAct tool-calling loop fills missing/low-confidence fields via web search, only for `MATCHED_EXISTING` items |
| Evaluation | RAGAS-style faithfulness scoring on matches and enrichments; LLM disambiguation for `AMBIGUOUS` matches |
| Reporting | Generates a markdown audit report and CSV/JSON catalog export, joined against the full audit log |

## Known Limitations (v1 Scope)

- No OCR / scanned catalog support
- Batch processing only — no real-time feed sync
- Single-tenant architecture
- No direct marketplace push (Shopify/Amazon API)
- English-language catalogs only