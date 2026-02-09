import argparse

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query the RAG pipeline interactively",
    )
    parser.add_argument("question", help="Question to ask")
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in ChunkStrategy],
        default="fixed",
    )
    parser.add_argument(
        "--model",
        choices=[m.value for m in EmbedModelName],
        default="voyage-3-large",
    )
    parser.add_argument("--llm", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--show-contexts",
        action="store_true",
        help="Print retrieved chunks with similarity scores",
    )
    args = parser.parse_args()

    strategy = ChunkStrategy(args.strategy)
    embed_model = EmbedModelName(args.model)

    engine = get_query_engine(
        strategy, embed_model, llm_model=args.llm, similarity_top_k=args.top_k
    )
    response = query(engine, args.question)

    print(f"\n[{strategy.value} | {embed_model.value} | {args.llm}]\n")
    print(response)

    if args.show_contexts:
        print(f"\n--- Retrieved Contexts ({len(response.source_nodes)}) ---")
        for i, node in enumerate(response.source_nodes, 1):
            score = getattr(node, "score", None)
            score_str = f" (score: {score:.4f})" if score is not None else ""
            print(f"\n[{i}]{score_str}")
            print(node.get_content()[:500])
            if len(node.get_content()) > 500:
                print(f"  ... ({len(node.get_content())} chars total)")
