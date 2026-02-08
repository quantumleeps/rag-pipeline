from rag_pipeline.query import DEFAULT_MODEL, DEFAULT_TOP_K


class TestQueryDefaults:
    def test_default_model(self):
        assert "claude" in DEFAULT_MODEL

    def test_default_top_k(self):
        assert DEFAULT_TOP_K > 0
