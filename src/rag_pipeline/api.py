import os

from fastapi import FastAPI, HTTPException

from rag_pipeline.chunkers import ChunkStrategy
from rag_pipeline.embed import EmbedModelName
from rag_pipeline.query import get_query_engine, query
from rag_pipeline.schemas import QueryRequest, QueryResponse, Source

MOCK_MODE = os.environ.get("MOCK_MODE", "").lower() in ("1", "true", "yes")

app = FastAPI(title="rag-pipeline")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/strategies")
def strategies() -> list[str]:
    return [s.value for s in ChunkStrategy]


@app.get("/models")
def models() -> list[str]:
    return [m.value for m in EmbedModelName]


MOCK_RESPONSE = QueryResponse(
    answer="[mock] The maximum contaminant level for bromate is 0.010 mg/L.",
    sources=[
        Source(text="[mock] 40 CFR 141.64 — MCLs for disinfection byproducts.", score=0.95)
    ],
)


@app.post("/query")
def handle_query(req: QueryRequest) -> QueryResponse:
    if MOCK_MODE:
        return MOCK_RESPONSE

    try:
        strategy = ChunkStrategy(req.strategy)
        model = EmbedModelName(req.model)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    engine = get_query_engine(strategy, model, similarity_top_k=req.top_k)
    response = query(engine, req.question)
    sources = [
        Source(
            text=node.get_content()[:500],
            score=getattr(node, "score", None),
        )
        for node in response.source_nodes
    ]
    return QueryResponse(answer=str(response), sources=sources)
