from rag_pipeline.chunkers import ChunkStrategy
from rag_pipeline.embed import EmbedModelName
from rag_pipeline.store import make_table_name


class TestMakeTableName:
    def test_fixed_voyage_3_large(self):
        name = make_table_name(ChunkStrategy.FIXED, EmbedModelName.VOYAGE_3_LARGE)
        assert name == "fixed_voyage_3_large"

    def test_semantic_voyage_3_5(self):
        name = make_table_name(ChunkStrategy.SEMANTIC, EmbedModelName.VOYAGE_3_5)
        assert name == "semantic_voyage_3_5"

    def test_hierarchical_voyage_law_2(self):
        name = make_table_name(ChunkStrategy.HIERARCHICAL, EmbedModelName.VOYAGE_LAW_2)
        assert name == "hierarchical_voyage_law_2"

    def test_all_combinations_unique(self):
        names = {make_table_name(s, m) for s in ChunkStrategy for m in EmbedModelName}
        assert len(names) == 9

    def test_names_are_valid_pg_identifiers(self):
        for s in ChunkStrategy:
            for m in EmbedModelName:
                name = make_table_name(s, m)
                assert name.isidentifier(), f"{name} is not a valid identifier"
