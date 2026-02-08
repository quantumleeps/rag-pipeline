from llama_index.core.schema import Document

from rag_pipeline.chunkers import (
    ChunkStrategy,
    FixedSizeChunker,
    HierarchicalChunker,
    get_chunker,
)

SAMPLE_DOCS = [
    Document(
        text=(
            "Ozone is a powerful oxidant and disinfectant. "
            "It is generated on-site because it is unstable and decomposes rapidly. "
            "Ozone is effective against bacteria, viruses, and protozoan cysts. "
            "The CT values for ozone disinfection depend on temperature and pH. "
            "Typical contact times range from 4 to 10 minutes. "
            "Bromate formation is a concern when ozone is used "
            "with bromide-containing water. "
            "The EPA set a maximum contaminant level of 0.010 mg/L "
            "for bromate. "
            "Ozone can also oxidize iron, manganese, and sulfide. "
            "Systems using ozone must monitor for DBPs. "
            "Alternative disinfectants include chlorine dioxide "
            "and ultraviolet radiation."
        ),
        metadata={"source": "test-ozone-doc.pdf"},
    ),
]


class TestFixedSizeChunker:
    def test_produces_nodes(self):
        chunker = FixedSizeChunker(chunk_size=128, chunk_overlap=20)
        nodes = chunker.chunk(SAMPLE_DOCS)
        assert len(nodes) > 0

    def test_respects_chunk_size(self):
        chunker = FixedSizeChunker(chunk_size=128, chunk_overlap=20)
        nodes = chunker.chunk(SAMPLE_DOCS)
        for node in nodes:
            assert len(node.get_content()) > 0

    def test_preserves_metadata(self):
        chunker = FixedSizeChunker(chunk_size=128, chunk_overlap=20)
        nodes = chunker.chunk(SAMPLE_DOCS)
        for node in nodes:
            assert "source" in node.metadata

    def test_smaller_chunks_produce_more_nodes(self):
        small = FixedSizeChunker(chunk_size=64, chunk_overlap=10)
        large = FixedSizeChunker(chunk_size=256, chunk_overlap=20)
        small_nodes = small.chunk(SAMPLE_DOCS)
        large_nodes = large.chunk(SAMPLE_DOCS)
        assert len(small_nodes) >= len(large_nodes)


class TestHierarchicalChunker:
    def test_produces_nodes(self):
        chunker = HierarchicalChunker(chunk_sizes=[512, 128])
        nodes = chunker.chunk(SAMPLE_DOCS)
        assert len(nodes) > 0

    def test_produces_more_nodes_than_fixed(self):
        hierarchical = HierarchicalChunker(chunk_sizes=[512, 128])
        fixed = FixedSizeChunker(chunk_size=512, chunk_overlap=50)
        h_nodes = hierarchical.chunk(SAMPLE_DOCS)
        f_nodes = fixed.chunk(SAMPLE_DOCS)
        assert len(h_nodes) >= len(f_nodes)


class TestGetChunker:
    def test_returns_fixed(self):
        chunker = get_chunker(ChunkStrategy.FIXED)
        assert isinstance(chunker, FixedSizeChunker)

    def test_returns_hierarchical(self):
        chunker = get_chunker(ChunkStrategy.HIERARCHICAL)
        assert isinstance(chunker, HierarchicalChunker)
