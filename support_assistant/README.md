# Module 3 — Support Assistant (`/support_assistant`)

A small but complete GenAI service for Zepto: an embedded document corpus, a
LangGraph-orchestrated intent router with grounded retrieval, a
Pydantic-guaranteed JSON output, and a FastAPI wrapper.

**The graded baseline is fully offline:** with `MOCK_LLM` left at its
default (unset or `1`) every LLM call is replaced by deterministic,
rule-based mock logic — no signup, no API key, no network call to any LLM
provider. Retrieval still runs for real, because local embeddings +
ChromaDB need no API key either.

## Install / Run

```bash
pip install -r support_assistant/requirements.txt

# index the corpus once (also runs automatically at API startup)
python support_assistant/ingest.py

# run the API (MOCK_LLM at its default = mock mode)
cd support_assistant
uvicorn main:app --host 0.0.0.0 --port 7860

# exercise it
python run_examples.py           # writes example_calls.json
```

Docker (required, graded baseline — buildable and runnable locally):

```bash
cd support_assistant
docker build -t zepto-support-assistant .
docker run --rm -p 7860:7860 zepto-support-assistant
# then: POST http://localhost:7860/ask  {"query": "..."}
```

| File | Role |
|---|---|
| `docs/doc_01.txt` … `doc_08.txt` | The 8-document policy corpus (exact text) |
| `ingest.py` | Load → chunk → embed (all-MiniLM-L6-v2) → ChromaDB; cosine retrieval |
| `prompts.py` | Structured prompt template (role–context–task–format–length + negative constraint + few-shot) for the optional real-LLM path |
| `graph.py` | LangGraph `StateGraph`, TypedDict state, 3+ nodes, conditional edge, `MOCK_LLM` branches, Pydantic `Answer` schema |
| `main.py` | FastAPI app, `POST /ask` |
| `Dockerfile` | Local build + run serving `/ask` on port 7860 |
| `run_examples.py` / `example_calls.json` | Recorded example calls + raw JSON responses |
| `chroma_db/` | Persistent ChromaDB index (regenerate any time with `python ingest.py`) |

---

## Architecture — the RAG pipeline, stage by stage

```
 docs/doc_01..08.txt
        │  (1) INGESTION            ingest.py :: load_and_chunk()
        ▼                            one chunk per document, ids doc_01..doc_08
   chunk list
        │  (2) EMBEDDING            ingest.py :: get_model() + build_collection()
        ▼                            sentence-transformers all-MiniLM-L6-v2
   ChromaDB collection "zepto_policies"   (persistent, hnsw:space = cosine)
        │
 user query ──► classify_intent (graph.py) ──conditional edge──┐
        │                    │ policy_question                 │ general_question
        │                    ▼                                  ▼
        │   (3) RETRIEVAL  retrieve_and_answer            direct_answer
        │        embed query (ingest.py :: retrieve())     no retrieval
        │        top-3 by cosine from ChromaDB             mock: fixed canned
        │                    │                             string
        │                    ▼                                  │
        │   (4) GENERATION   mock: "Based on the              │
        │        retrieved context: <top-200 chars>"          │
        │        (optional real LLM via prompts.py)           │
        │                    └──────────────┬─────────────────┘
        ▼                                   ▼
   validate_output (Pydantic Answer: answer / sources / confidence)
        ▼
   FastAPI POST /ask  →  JSON response
```

1. **Ingestion** — `ingest.py :: load_and_chunk()` reads the 8 files in
   `docs/` and produces one chunk per document (the policy paragraphs are
   short, so per-document chunking preserves complete policy statements),
   with ids `doc_01` … `doc_08`.
2. **Embedding** — `ingest.py :: build_collection()` embeds every chunk with
   `sentence-transformers/all-MiniLM-L6-v2` (runs locally, free, keyless)
   and upserts the vectors into the persistent ChromaDB collection
   **`zepto_policies`** (`chroma_db/`, configured with `hnsw:space =
   "cosine"`).
3. **Retrieval** — the LangGraph node **`retrieve_and_answer`** calls
   `ingest.py :: retrieve()`, which embeds the incoming query with the same
   model and queries ChromaDB for the **top-3 most similar chunks by cosine
   similarity**. **This stage always runs for real** in both mock and real
   modes — it never branches on `MOCK_LLM`.
4. **Generation** — also inside `retrieve_and_answer` (policy questions) or
   **`direct_answer`** (general questions), the final answer text is
   produced. **This is the stage that branches on `MOCK_LLM`:**
   - **Default / mock (`MOCK_LLM` unset or `1`) — graded baseline:**
     `retrieve_and_answer` returns the canned template
     `f"Based on the retrieved context: {top_chunk_snippet}"` built from the
     first ~200 characters of the single most similar chunk; `direct_answer`
     returns the fixed string `"I can only answer questions about Zepto
     policies right now."`. **No network call to any LLM is made.**
     `sources`/`confidence` are populated deterministically in code
     (retrieved ids / `[]`, and `1.0`).
   - **Optional real LLM (`MOCK_LLM=0`) — ungraded extension:** both nodes
     prompt a real LLM (`prompts.py` template for retrieval-grounded answers)
     and the raw output is validated against the Pydantic schema with up to
     **2 corrective retries** before returning a clearly marked error
     response.

**Data flow between stages:** raw files → chunk dicts → (embedding vectors +
documents) in ChromaDB → at query time, query vector → top-3 chunk dicts →
state fields (`retrieved`, `answer`, `sources`, `confidence`) → Pydantic
`Answer` → HTTP JSON.

Routing itself — the conditional edge from `classify_intent` — does **not**
depend on `MOCK_LLM`; only the generation step inside each node does.

---

## Task 1 — Corpus + embeddings + ChromaDB

`python ingest.py` indexes all 8 documents:

```
[ingest] indexed 8 chunks into 'zepto_policies'
[ingest] done - ['doc_01', 'doc_02', 'doc_03', 'doc_04', 'doc_05', 'doc_06', 'doc_07', 'doc_08']
```

The collection is also (re)built automatically on API startup, so a fresh
clone works with zero manual steps.

## Task 2 — Structured prompt template (actual text)

Defined in `prompts.py` as `PROMPT_TEMPLATE` — all five skeleton components,
the negative constraint, and a few-shot example, verbatim:

```
ROLE: You are Zepto's official customer-support assistant. You answer only
questions about Zepto's delivery, returns, membership, tracking, cancellation,
damaged-item, gift-card, and support-hours policies. You are concise, factual,
and never speculative.

CONTEXT: The only source of truth is the CONTEXT block below, retrieved by
cosine similarity from Zepto's policy corpus (docs/doc_01.txt ...
docs/doc_08.txt):

--- BEGIN CONTEXT ---
{context}
--- END CONTEXT ---

TASK: Answer the customer's QUESTION using ONLY the information that appears
in the CONTEXT block. Quote or paraphrase the relevant policy wording. If the
CONTEXT does not contain the answer, say that you do not have that
information in the provided policy documents.

FORMAT: Respond with a single JSON object and nothing else, exactly in this shape:
{"answer": "<1-3 sentence answer grounded in the context>",
  "sources": ["<chunk ids used, e.g. doc_01>"],
  "confidence": <float between 0 and 1>}
The "sources" list must contain only chunk ids actually used, and must be
empty if you could not answer from the context.

LENGTH: The "answer" value must be at most 3 sentences (max ~80 words). Do
not add greetings, signatures, or any text outside the JSON object.

NEGATIVE CONSTRAINT: Do not answer using information not present in the
provided context. Do not use prior knowledge, common sense about grocery
apps, or anything outside the CONTEXT block - even if you are confident it
is true. If the required fact is absent from the CONTEXT, return an answer
stating the policy document does not cover it, with "sources": [] and a low
confidence.

FEW-SHOT EXAMPLE:
Customer QUESTION: What is the standard delivery fee on small orders?
Correct assistant response:
{"answer": "Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.",
  "sources": ["doc_01"],
  "confidence": 0.95}

Now answer the actual customer QUESTION from the CONTEXT only:

Customer QUESTION: {question}
```

The template is consumed only by the optional `MOCK_LLM=0` extension —
the graded mock path generates its answers from code, never from an LLM.

## Task 3 — LangGraph graph

State: `GraphState(TypedDict)` with `query, intent, retrieved, answer,
sources, confidence, validated`.

Nodes (**4, ≥3 required**):

| Node | Role | Mock branch (`MOCK_LLM` default) | Real branch (`MOCK_LLM=0`) |
|---|---|---|---|
| `classify_intent` | policy vs general | Keyword heuristic — `policy_question` if the lowercased query contains any of `delivery, return, refund, membership, tracking, cancel, gift card, support hours`; else `general_question`. **No LLM call.** | LLM classification |
| `retrieve_and_answer` | policy path | Real cosine top-3 retrieval, then canned `Based on the retrieved context: {top_chunk_snippet}` (first ~200 chars) | Real retrieval + structured-template LLM answer with schema-retry |
| `direct_answer` | general path | Fixed canned string, **no retrieval, no LLM call** | Direct LLM prompt, no retrieval |
| `validate_output` | schema enforcement | Constructs/checks the Pydantic `Answer` (deterministic — nothing to fail) | Same final check after retries |

Conditional edge: `classify_intent --route_by_intent--> retrieve_and_answer |
direct_answer` (routing logic itself is MOCK-independent).

### Demonstrated runs (MOCK_LLM at default, no LLM call)

**Policy query → `retrieve_and_answer`:**

```json
{
  "intent": "policy_question",
  "response": {
    "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
    "sources": ["doc_01", "doc_03", "doc_05"],
    "confidence": 1.0
  },
  "retrieved_ids": ["doc_01", "doc_03", "doc_05"]
}
```

Retrieval quality: the delivery-fee question's **top chunk is `doc_01`
(Delivery Policy)** — the retrieved content matches the question asked.
The answer follows the required canned template exactly.

**General query → `direct_answer`:**

```json
{
  "intent": "general_question",
  "response": {
    "answer": "I can only answer questions about Zepto policies right now.",
    "sources": [],
    "confidence": 1.0
  },
  "retrieved_ids": []
}
```

## Task 4 — Pydantic output schema

```python
class Answer(BaseModel):
    answer: str
    sources: list[str]      # empty for general_question answers
    confidence: float       # 0-1
```

- **Mock mode:** populated deterministically in code — `sources` = ids of
  the retrieved chunks (policy) or `[]` (general); `confidence` = `1.0`.
  There is no LLM output that could fail validation.
- **Real-LLM mode:** raw output is parsed and validated against `Answer`;
  on failure the prompt is re-sent with a corrective instruction, **up to 2
  additional retries**, after which a clearly marked error response
  (`"ERROR: the real LLM response failed schema validation after
  retries."`, `confidence 0.0`) is returned. (Retry code is present in
  `graph.py :: _call_llm_structured` even though it never triggers in the
  graded mock path.)

## Task 5 — FastAPI example calls (`MOCK_LLM` left at default)

`POST /ask` with `{"query": str}` → `Answer`.

**Example 1 — policy-style (triggers retrieval):**

```
POST /ask   {"query": "What is the delivery fee on orders below INR 149?"}
HTTP 200
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": ["doc_01", "doc_05", "doc_03"],
  "confidence": 1.0
}
```

**Example 2 — unrelated (no retrieval):**

```
POST /ask   {"query": "What is the capital of France?"}
HTTP 200
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

Both transcripts (requests + raw JSON responses) are saved in
`example_calls.json`; `GET /health` returns
`{"status": "ok", "mock_llm": true}`.

## Task 6 — Dockerfile

`Dockerfile` builds the service and serves it locally:

```bash
cd support_assistant
docker build -t zepto-support-assistant .
docker run --rm -p 7860:7860 zepto-support-assistant
# POST http://localhost:7860/ask  {"query": "..."}
```

- Base `python:3.11-slim`, installs `requirements.txt`, copies `docs/` +
  app code, pre-builds the ChromaDB collection at image build time, and
  defaults to `ENV MOCK_LLM=1`.
- CMD: `["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]`.
- This local build/run path is the required graded baseline; no registry
  push or cloud deployment is needed. (A Hugging Face Spaces deployment,
  if attempted, would be an optional ungraded stretch and its API key would
  live only in Space secrets — never in this repository.)

## Optional ungraded extensions (not used for grading)

- `MOCK_LLM=0` + `GROQ_API_KEY`: real LLM generation through Groq's free
  tier (code present in `graph.py :: _call_llm` / `_call_llm_structured`),
  including the 2-retry schema-correction loop.
- Currency-style external lookups: none anywhere in this module —
  retrieval and embeddings are fully local.
