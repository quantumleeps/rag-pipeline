import os
from unittest.mock import patch

from llama_index.embeddings.voyageai import VoyageEmbedding

from rag_pipeline.embed import EmbedModelName, get_embed_model


class TestEmbedModelName:
    def test_enum_values(self):
        assert EmbedModelName.VOYAGE_3_LARGE.value == "voyage-3-large"
        assert EmbedModelName.VOYAGE_3_5.value == "voyage-3.5"
        assert EmbedModelName.VOYAGE_LAW_2.value == "voyage-law-2"

    def test_enum_has_three_members(self):
        assert len(EmbedModelName) == 3


@patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"})
class TestGetEmbedModel:
    def test_returns_voyage_embedding(self):
        for model in EmbedModelName:
            embed = get_embed_model(model)
            assert isinstance(embed, VoyageEmbedding)

    def test_model_name_matches(self):
        for model in EmbedModelName:
            embed = get_embed_model(model)
            assert embed.model_name == model.value
