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

### Setup

```bash
# Clone and install
git clone <repo-url>
cd rag-pipeline
uv sync --all-extras

# Configure environment
cp .env.example .env
# Edit .env with your VOYAGE_API_KEY

# Download EPA corpus
bash scripts/download_data.sh

# Start pgvector
docker compose up -d
```

### Run

```bash
# Run tests
uv run pytest

# Run the pipeline (coming soon)
# uv run python -m rag_pipeline.run
```

## Project Structure

```
rag-pipeline/
  src/rag_pipeline/
    ingest.py          # PDF loading via LlamaIndex
    chunkers.py        # Fixed, semantic, hierarchical strategies
    embed.py           # Voyage AI embedding (wip)
    store.py           # pgvector storage (wip)
    query.py           # Retrieval + LLM generation (wip)
  eval/
    evaluate.py        # RAGAS evaluation harness (wip)
  tests/
    test_chunkers.py   # Chunker unit tests
  scripts/
    download_data.sh   # Fetches EPA PDFs
  docker-compose.yml   # pgvector service
```

## Contributing

PRs welcome. Run `pre-commit install` after cloning and ensure `uv run pytest` passes before submitting.

## License

MIT
