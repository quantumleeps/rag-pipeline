# rag-pipeline

A RAG system that indexes EPA water treatment regulations and evaluates how chunking strategy and embedding model choice affect retrieval quality.

## What this does

Ingests 12 publicly available EPA documents (PFAS treatment, disinfection profiling, ozone chemistry, water quality standards) and runs them through a 3x3 evaluation matrix:

- **3 chunking strategies**: fixed-size, semantic, and hierarchical
- **3 embedding models**: `voyage-3-large`, `voyage-3.5`, and `voyage-law-2`

Each combination is stored in pgvector and evaluated with [RAGAS](https://docs.ragas.io/) to measure retrieval precision, answer relevancy, and faithfulness.

## How it works

PDFs are loaded via LlamaIndex and split into chunks using three strategies: **fixed-size** (token-window with overlap), **semantic** (embedding-similarity breakpoints), and **hierarchical** (parent-child at multiple granularities). Each set of chunks is embedded with three Voyage AI models — `voyage-3-large` (best general-purpose), `voyage-3.5` (balanced), and `voyage-law-2` (legal/regulatory domain) — producing 9 index variants stored in pgvector. RAGAS evaluates each variant on retrieval precision, answer relevancy, and faithfulness to surface which combination works best for regulatory text.

## Quickstart

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker (for pgvector)
- A [Voyage AI](https://www.voyageai.com/) API key
- An [Anthropic](https://console.anthropic.com/) API key

### Setup

```bash
# Clone and install
git clone https://github.com/quantumleeps/rag-pipeline.git
cd rag-pipeline
uv sync --all-extras

# Configure environment
cp .env.example .env
# Edit .env with your VOYAGE_API_KEY and ANTHROPIC_API_KEY

# Download EPA corpus
bash scripts/download_data.sh

# Start pgvector
docker compose up -d
```

### Run

```bash
# Run tests
uv run pytest

# Index all 9 variants into pgvector
uv run python -m rag_pipeline.run

# Query interactively
uv run python -m rag_pipeline.query "What is the MCL for bromate?"
uv run python -m rag_pipeline.query "What are the BAT for PFAS removal?" --strategy semantic --model voyage-3-large --show-contexts

# Evaluate with RAGAS (resumes from previous results by default)
uv run python -m eval.evaluate
uv run python -m eval.evaluate --fresh  # re-run all variants from scratch
```

## Project Structure

```
rag-pipeline/
  src/rag_pipeline/
    ingest.py          # PDF loading via LlamaIndex
    chunkers.py        # Fixed, semantic, hierarchical strategies
    embed.py           # Voyage AI embedding model factory
    store.py           # pgvector storage, per-variant tables
    query.py           # Retrieval + Claude LLM generation
    run.py             # Pipeline orchestrator
  eval/
    evaluate.py        # RAGAS evaluation harness
  tests/
    test_chunkers.py   # Chunker unit tests
    test_embed.py      # Embedding model tests
    test_store.py      # Table naming tests
    test_query.py      # Query config tests
  scripts/
    download_data.sh   # Fetches EPA PDFs
  docker-compose.yml   # pgvector service
```

## Contributing

PRs welcome. Run `pre-commit install` after cloning and ensure `uv run pytest` passes before submitting.

## License

MIT
