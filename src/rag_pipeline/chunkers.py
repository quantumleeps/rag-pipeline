from enum import Enum
from typing import Protocol

from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
)
from llama_index.core.schema import BaseNode, Document
from llama_index.embeddings.voyageai import VoyageEmbedding


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"


class Chunker(Protocol):
    def chunk(self, documents: list[Document]) -> list[BaseNode]: ...


class FixedSizeChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self._parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, documents: list[Document]) -> list[BaseNode]:
        return self._parser.get_nodes_from_documents(documents, show_progress=True)


class SemanticChunker:
    def __init__(
        self,
        embed_model: VoyageEmbedding | None = None,
        breakpoint_percentile: int = 95,
        buffer_size: int = 1,
    ):
        if embed_model is None:
            embed_model = VoyageEmbedding(model_name="voyage-3.5")

        self._parser = SemanticSplitterNodeParser(
            embed_model=embed_model,
            breakpoint_percentile_threshold=breakpoint_percentile,
            buffer_size=buffer_size,
        )

    def chunk(self, documents: list[Document]) -> list[BaseNode]:
        return self._parser.get_nodes_from_documents(documents, show_progress=True)


class HierarchicalChunker:
    def __init__(self, chunk_sizes: list[int] | None = None):
        if chunk_sizes is None:
            chunk_sizes = [2048, 512, 128]

        self._parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=chunk_sizes,
        )

    def chunk(self, documents: list[Document]) -> list[BaseNode]:
        return self._parser.get_nodes_from_documents(documents, show_progress=True)


def get_chunker(
    strategy: ChunkStrategy,
    embed_model: VoyageEmbedding | None = None,
) -> Chunker:
    match strategy:
        case ChunkStrategy.FIXED:
            return FixedSizeChunker()
        case ChunkStrategy.SEMANTIC:
            return SemanticChunker(embed_model=embed_model)
        case ChunkStrategy.HIERARCHICAL:
            return HierarchicalChunker()
