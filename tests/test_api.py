from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from rag_pipeline.api import app

client = TestClient(app)


class TestHealth:
    def test_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestStrategies:
    def test_returns_all_strategies(self):
        response = client.get("/strategies")
        assert response.status_code == 200
        data = response.json()
        assert "fixed" in data
        assert "semantic" in data
        assert "hierarchical" in data

    def test_returns_list(self):
        response = client.get("/strategies")
        assert isinstance(response.json(), list)


class TestModels:
    def test_returns_all_models(self):
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "voyage-3-large" in data
        assert "voyage-3.5" in data
        assert "voyage-law-2" in data

    def test_returns_list(self):
        response = client.get("/models")
        assert isinstance(response.json(), list)


class TestQuery:
    @patch("rag_pipeline.api.get_query_engine")
    @patch("rag_pipeline.api.query")
    def test_returns_answer_and_sources(self, mock_query, mock_get_engine):
        mock_node = MagicMock()
        mock_node.get_content.return_value = "EPA regulation text about bromate"
        mock_node.score = 0.95

        mock_response = MagicMock()
        mock_response.__str__ = lambda _: "The MCL for bromate is 0.010 mg/L."
        mock_response.source_nodes = [mock_node]

        mock_query.return_value = mock_response
        mock_get_engine.return_value = MagicMock()

        response = client.post(
            "/query",
            json={"question": "What is the MCL for bromate?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The MCL for bromate is 0.010 mg/L."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "EPA regulation text about bromate"
        assert data["sources"][0]["score"] == 0.95

    @patch("rag_pipeline.api.get_query_engine")
    @patch("rag_pipeline.api.query")
    def test_uses_default_strategy_and_model(self, mock_query, mock_get_engine):
        mock_response = MagicMock()
        mock_response.__str__ = lambda _: "answer"
        mock_response.source_nodes = []
        mock_query.return_value = mock_response
        mock_get_engine.return_value = MagicMock()

        client.post("/query", json={"question": "test"})

        call_args = mock_get_engine.call_args
        from rag_pipeline.chunkers import ChunkStrategy
        from rag_pipeline.embed import EmbedModelName

        assert call_args[0][0] == ChunkStrategy.FIXED
        assert call_args[0][1] == EmbedModelName.VOYAGE_3_LARGE

    @patch("rag_pipeline.api.get_query_engine")
    @patch("rag_pipeline.api.query")
    def test_custom_strategy_and_model(self, mock_query, mock_get_engine):
        mock_response = MagicMock()
        mock_response.__str__ = lambda _: "answer"
        mock_response.source_nodes = []
        mock_query.return_value = mock_response
        mock_get_engine.return_value = MagicMock()

        client.post(
            "/query",
            json={
                "question": "test",
                "strategy": "semantic",
                "model": "voyage-3.5",
                "top_k": 3,
            },
        )

        call_args = mock_get_engine.call_args
        from rag_pipeline.chunkers import ChunkStrategy
        from rag_pipeline.embed import EmbedModelName

        assert call_args[0][0] == ChunkStrategy.SEMANTIC
        assert call_args[0][1] == EmbedModelName.VOYAGE_3_5
        assert call_args[1]["similarity_top_k"] == 3

    def test_missing_question_returns_422(self):
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_invalid_strategy_returns_error(self):
        response = client.post(
            "/query",
            json={"question": "test", "strategy": "invalid"},
        )
        assert response.status_code != 200


class TestMockMode:
    @patch("rag_pipeline.api.MOCK_MODE", True)
    def test_returns_canned_response(self):
        response = client.post("/query", json={"question": "anything"})
        assert response.status_code == 200
        data = response.json()
        assert "[mock]" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["score"] == 0.95

    @patch("rag_pipeline.api.MOCK_MODE", True)
    def test_does_not_call_query_engine(self):
        with patch("rag_pipeline.api.get_query_engine") as mock_engine:
            client.post("/query", json={"question": "anything"})
            mock_engine.assert_not_called()
