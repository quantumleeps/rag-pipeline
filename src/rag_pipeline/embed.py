from enum import Enum

from llama_index.embeddings.voyageai import VoyageEmbedding


class EmbedModelName(str, Enum):
    VOYAGE_3_LARGE = "voyage-3-large"
    VOYAGE_3_5 = "voyage-3.5"
    VOYAGE_LAW_2 = "voyage-law-2"


def get_embed_model(model: EmbedModelName) -> VoyageEmbedding:
    return VoyageEmbedding(model_name=model.value)
