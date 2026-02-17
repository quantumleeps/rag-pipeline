from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    strategy: str = "fixed"
    model: str = "voyage-3-large"
    top_k: int = 5


class Source(BaseModel):
    text: str
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
