from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.llms.anthropic import Anthropic

from rag_pipeline.chunkers import ChunkStrategy
from rag_pipeline.embed import EmbedModelName
from rag_pipeline.store import load_index

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_TOP_K = 5


def get_query_engine(
    strategy: ChunkStrategy,
    model: EmbedModelName,
    llm_model: str = DEFAULT_MODEL,
    similarity_top_k: int = DEFAULT_TOP_K,
) -> BaseQueryEngine:
    llm = Anthropic(model=llm_model)
    index = load_index(strategy, model)
    return index.as_query_engine(llm=llm, similarity_top_k=similarity_top_k)


def query(engine: BaseQueryEngine, question: str) -> RESPONSE_TYPE:
    return engine.query(question)
