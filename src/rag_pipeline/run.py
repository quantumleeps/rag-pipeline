import logging
import sys

from rag_pipeline.chunkers import ChunkStrategy, get_chunker
from rag_pipeline.embed import EmbedModelName, get_embed_model
from rag_pipeline.ingest import load_documents
from rag_pipeline.store import build_index, make_table_name

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STRATEGIES = list(ChunkStrategy)
MODELS = list(EmbedModelName)


def run_pipeline(data_dir: str = "data") -> None:
    log.info("Loading documents from %s", data_dir)
    documents = load_documents(data_dir)
    log.info("Loaded %d documents", len(documents))

    for strategy in STRATEGIES:
        log.info("Chunking with strategy=%s", strategy.value)
        embed_model = None
        if strategy == ChunkStrategy.SEMANTIC:
            embed_model = get_embed_model(EmbedModelName.VOYAGE_3_5)
        chunker = get_chunker(strategy, embed_model=embed_model)
        nodes = chunker.chunk(documents)
        log.info("  Produced %d nodes", len(nodes))

        for model in MODELS:
            table = make_table_name(strategy, model)
            log.info("  Indexing into %s with %s", table, model.value)
            build_index(nodes, strategy, model)
            log.info("  Done: %s", table)

    log.info("All 9 variants indexed")


if __name__ == "__main__":
    run_pipeline(sys.argv[1] if len(sys.argv) > 1 else "data")
