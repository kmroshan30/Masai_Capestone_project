"""FastAPI wrapper around the LangGraph support assistant.

Run locally (MOCK_LLM at its default = mock mode):
    uvicorn main:app --host 0.0.0.0 --port 7860

POST /ask  {"query": "..."}  ->  {"answer": str, "sources": [...],
                                  "confidence": float}
"""

from __future__ import annotations

from fastapi import FastAPI

from graph import Answer, AskRequest, mock_mode, run_agent

app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG over Zepto policy docs, LangGraph-routed, "
                "MOCK_LLM-gated generation.",
    version="1.0.0",
)


@app.on_event("startup")
def _startup() -> None:
    # Build/refresh the ChromaDB collection once at startup (local
    # embeddings; no API key). Cheap if already indexed.
    from ingest import build_collection

    build_collection()
    print(f"[startup] MOCK_LLM mock mode = {mock_mode()}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mock_llm": mock_mode()}


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest) -> Answer:
    result = run_agent(req.query)
    return Answer(**result["response"])
