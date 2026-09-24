"""Record example /ask transcripts (MOCK_LLM at its default = mock mode).

Usage:
    # terminal 1
    uvicorn main:app --port 7860
    # terminal 2
    python run_examples.py
"""

from __future__ import annotations

import json
import sys

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"

EXAMPLES = [
    # policy-style query -> classify_intent routes to retrieve_and_answer
    ("policy", "What is the delivery fee on orders below INR 149?"),
    # unrelated query -> classify_intent routes to direct_answer
    ("general", "What is the capital of France?"),
]

transcripts = []
for kind, query in EXAMPLES:
    resp = requests.post(f"{BASE}/ask", json={"query": query}, timeout=60)
    entry = {
        "kind": kind,
        "request": {"query": query},
        "status_code": resp.status_code,
        "response": resp.json(),
    }
    transcripts.append(entry)
    print(f"--- {kind}: {query}")
    print(json.dumps(entry["response"], indent=2))

with open("example_calls.json", "w", encoding="utf-8") as fh:
    json.dump(transcripts, fh, indent=2, ensure_ascii=False)
print("\n[saved] example_calls.json")
