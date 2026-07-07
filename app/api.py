from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import rag

app = FastAPI(title="Document Q&A RAG API")

class AskRequest(BaseModel):
    query: str
    k: int = 8
    source_filter: Optional[str] = None

class ContradictRequest(BaseModel):
    doc_a: str
    doc_b: str
    topic: Optional[str] = None

@app.get("/sources")
def sources():
    return {"sources": rag.list_sources()}

@app.post("/ask")
def ask(req: AskRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    return rag.answer_query(req.query, k=req.k, source_filter=req.source_filter)

@app.post("/contradict")
def contradict(req: ContradictRequest):
    valid_sources = rag.list_sources()
    for d in (req.doc_a, req.doc_b):
        if d not in valid_sources:
            raise HTTPException(
                status_code=400,
                detail=f"'{d}' is not an ingested document. Valid options: {valid_sources}",
            )
    return rag.check_contradiction(req.doc_a, req.doc_b, topic=req.topic)