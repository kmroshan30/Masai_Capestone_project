"""LangGraph orchestration for the Zepto support assistant.

StateGraph nodes:
  classify_intent      -> routes policy vs general (keyword heuristic in mock)
  retrieve_and_answer  -> real cosine retrieval + (mock | real) generation
  direct_answer        -> (mock | real) generation with no retrieval
  validate_output      -> enforces the Pydantic schema on the final answer

Every generation step branches on the MOCK_LLM env var:
  MOCK_LLM unset or "1"  -> deterministic rule-based mock (graded baseline,
                            no network call to any LLM provider)
  MOCK_LLM="0"           -> optional real-LLM extension (Groq / OpenAI-style)
"""

from __future__ import annotations

import json
import os
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

from ingest import retrieve
from prompts import CORRECTIVE_INSTRUCTION, build_prompt

# --------------------------------------------------------------------------- #
# MOCK toggle
# --------------------------------------------------------------------------- #
def mock_mode() -> bool:
    """True (default) => fully offline deterministic mock generation."""
    return os.environ.get("MOCK_LLM", "1") != "0"


POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]

GENERAL_CANNED = "I can only answer questions about Zepto policies right now."
REAL_LLM_ERROR = (
    "ERROR: the real LLM response failed schema validation after retries."
)


# --------------------------------------------------------------------------- #
# Structured output schema
# --------------------------------------------------------------------------- #
class Answer(BaseModel):
    answer: str = Field(..., description="Grounded answer string")
    sources: list[str] = Field(
        default_factory=list,
        description="Chunk ids used; empty for general_question answers",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)


class AskRequest(BaseModel):
    query: str


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class GraphState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved: list[dict]
    raw_generation: str
    answer: str
    sources: list[str]
    confidence: float
    validated: bool


# --------------------------------------------------------------------------- #
# Node 1: classify_intent
# --------------------------------------------------------------------------- #
def classify_intent(state: GraphState) -> dict:
    query = state.get("query", "")
    if mock_mode():
        # GRADED BASELINE: keyword heuristic, no LLM call.
        lowered = query.lower()
        intent = (
            "policy_question"
            if any(kw in lowered for kw in POLICY_KEYWORDS)
            else "general_question"
        )
        return {"intent": intent}

    # Optional MOCK_LLM=0 extension: let the LLM classify.
    prompt = (
        "Classify the customer query as exactly one of: policy_question, "
        "general_question.\n"
        "policy_question = asks about Zepto delivery/returns/refunds/"
        "membership/tracking/cancellation/gift cards/support hours.\n"
        "Return ONLY JSON: {\"intent\": \"...\"}\n"
        f"Query: {query}"
    )
    raw = _call_llm(prompt)
    try:
        intent = json.loads(raw).get("intent", "general_question")
    except Exception:
        intent = "general_question"
    if intent not in ("policy_question", "general_question"):
        intent = "general_question"
    return {"intent": intent}


# --------------------------------------------------------------------------- #
# Node 2: retrieve_and_answer  (for policy_question)
# --------------------------------------------------------------------------- #
def retrieve_and_answer(state: GraphState) -> dict:
    query = state.get("query", "")

    # Retrieval ALWAYS runs for real in both modes (local embeddings +
    # ChromaDB, no API key, no LLM network call).
    chunks = retrieve(query, k=3)

    if mock_mode():
        # GRADED BASELINE: canned templated answer from the top chunk.
        top = chunks[0] if chunks else {"id": "", "text": ""}
        snippet = top["text"][:200]
        answer = f"Based on the retrieved context: {snippet}"
        sources = [c["id"] for c in chunks]
        return {
            "retrieved": chunks,
            "raw_generation": answer,
            "answer": answer,
            "sources": sources,
            "confidence": 1.0,
        }

    # Optional MOCK_LLM=0 extension: prompt the real LLM with the
    # structured template, retrying up to 2 extra times on validation failure.
    context = "\n\n".join(f"[{c['id']}]\n{c['text']}" for c in chunks)
    prompt = build_prompt(context, query)
    answer_obj, raw, ok = _call_llm_structured(prompt)
    if ok:
        return {
            "retrieved": chunks,
            "raw_generation": raw,
            "answer": answer_obj.answer,
            "sources": answer_obj.sources,
            "confidence": answer_obj.confidence,
            "validated": True,
        }
    # clearly marked error response after retries
    return {
        "retrieved": chunks,
        "raw_generation": raw,
        "answer": REAL_LLM_ERROR,
        "sources": [],
        "confidence": 0.0,
        "validated": False,
    }


# --------------------------------------------------------------------------- #
# Node 3: direct_answer  (for general_question)
# --------------------------------------------------------------------------- #
def direct_answer(state: GraphState) -> dict:
    if mock_mode():
        # GRADED BASELINE: fixed canned string, no retrieval, no LLM call.
        return {
            "retrieved": [],
            "raw_generation": GENERAL_CANNED,
            "answer": GENERAL_CANNED,
            "sources": [],      # empty for general_question answers
            "confidence": 1.0,
        }

    # Optional MOCK_LLM=0 extension: prompt the LLM directly, no retrieval.
    prompt = (
        "You are Zepto's support assistant. The user asked a non-policy "
        "question. Politely say you can only answer questions about Zepto "
        "policies. Return ONLY JSON: {\"answer\": str, \"sources\": [], "
        "\"confidence\": float}\n"
        f"Query: {state.get('query', '')}"
    )
    answer_obj, raw, ok = _call_llm_structured(prompt)
    if ok:
        return {
            "retrieved": [],
            "raw_generation": raw,
            "answer": answer_obj.answer,
            "sources": answer_obj.sources,
            "confidence": answer_obj.confidence,
            "validated": True,
        }
    return {
        "retrieved": [],
        "raw_generation": raw,
        "answer": REAL_LLM_ERROR,
        "sources": [],
        "confidence": 0.0,
        "validated": False,
    }


# --------------------------------------------------------------------------- #
# Node 4: validate_output (schema enforcement on the final answer)
# --------------------------------------------------------------------------- #
def validate_output(state: GraphState) -> dict:
    """Populate/verify the Pydantic Answer on the final state.

    Mock mode: everything was already built deterministically in code
    (there is no LLM output that could fail validation) - this node simply
    constructs and checks the schema. Real-LLM mode: retries already ran in
    the generating node; here we mark validity of the final object.
    """
    obj = Answer(
        answer=state.get("answer", ""),
        sources=state.get("sources", []),
        confidence=float(state.get("confidence", 0.0)),
    )
    # write back the validated (coerced) values
    return {
        "answer": obj.answer,
        "sources": obj.sources,
        "confidence": obj.confidence,
        "validated": True,
    }


# --------------------------------------------------------------------------- #
# Conditional edge: mirror of an intent router (not MOCK-dependent)
# --------------------------------------------------------------------------- #
def route_by_intent(state: GraphState) -> str:
    return (
        "retrieve_and_answer"
        if state.get("intent") == "policy_question"
        else "direct_answer"
    )


# --------------------------------------------------------------------------- #
# Optional real-LLM helpers (MOCK_LLM=0) - never hit in the graded baseline
# --------------------------------------------------------------------------- #
def _call_llm(prompt: str) -> str:
    """Minimal keyless-gated LLM call (Groq free tier, OpenAI-compatible)."""
    import requests

    api_key = os.environ.get("GROQ_API_KEY", "")
    base = os.environ.get("GROQ_BASE_URL",
                          "https://api.groq.com/openai/v1/chat/completions")
    model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    if not api_key:
        return json.dumps({"error": "GROQ_API_KEY not set"})
    resp = requests.post(
        base,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        return json.dumps({"error": f"HTTP {resp.status_code}"})
    return resp.json()["choices"][0]["message"]["content"]


def _call_llm_structured(prompt: str, max_retries: int = 2):
    """Call the real LLM and validate against Answer; retry up to 2 more
    times with a corrective instruction before giving up."""
    current = prompt
    raw = ""
    for attempt in range(max_retries + 1):  # 1 initial + up to 2 retries
        raw = _call_llm(current)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        try:
            data = json.loads(cleaned)
            obj = Answer(**data)
            return obj, raw, True
        except (json.JSONDecodeError, ValidationError):
            current = prompt + CORRECTIVE_INSTRUCTION
            continue
    return None, raw, False


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("classify_intent", classify_intent)
    g.add_node("retrieve_and_answer", retrieve_and_answer)
    g.add_node("direct_answer", direct_answer)
    g.add_node("validate_output", validate_output)

    g.set_entry_point("classify_intent")
    g.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )
    g.add_edge("retrieve_and_answer", "validate_output")
    g.add_edge("direct_answer", "validate_output")
    g.add_edge("validate_output", END)
    return g.compile()


app_graph = build_graph()


def run_agent(query: str) -> dict:
    """Run the graph and return a validated Answer dict + routing info."""
    state: GraphState = {"query": query}
    final: dict = app_graph.invoke(state)
    obj = Answer(
        answer=final.get("answer", ""),
        sources=final.get("sources", []),
        confidence=float(final.get("confidence", 0.0)),
    )
    return {
        "intent": final.get("intent"),
        "response": obj.model_dump(),
        "retrieved_ids": [c["id"] for c in final.get("retrieved", [])],
    }


if __name__ == "__main__":
    for q in (
        "What is the delivery fee below INR 149?",
        "What is the capital of France?",
    ):
        print(q, "->", json.dumps(run_agent(q), indent=2))
