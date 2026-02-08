import os
import re

from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.postgres import PGVectorStore

from rag_pipeline.chunkers import ChunkStrategy
from rag_pipeline.embed import EmbedModelName, get_embed_model

load_dotenv()

EMBED_DIM = 1024


def make_table_name(strategy: ChunkStrategy, model: EmbedModelName) -> str:
    raw = f"{strategy.value}_{model.value}"
    return re.sub(r"[^a-z0-9]", "_", raw.lower())


def get_vector_store(strategy: ChunkStrategy, model: EmbedModelName) -> PGVectorStore:
    return PGVectorStore.from_params(
        database=os.environ["POSTGRES_DB"],
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        password=os.environ["POSTGRES_PASSWORD"],
        port=os.environ.get("POSTGRES_PORT", "5432"),
        user=os.environ["POSTGRES_USER"],
        table_name=make_table_name(strategy, model),
        embed_dim=EMBED_DIM,
    )


def build_index(
    nodes: list[BaseNode],
    strategy: ChunkStrategy,
    model: EmbedModelName,
) -> VectorStoreIndex:
    vector_store = get_vector_store(strategy, model)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = get_embed_model(model)
    return VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )


def load_index(strategy: ChunkStrategy, model: EmbedModelName) -> VectorStoreIndex:
    vector_store = get_vector_store(strategy, model)
    embed_model = get_embed_model(model)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
