# Zepto Data & AI Project

Three-module assignment: a raw-to-relational data pipeline, an end-to-end
analytics/modeling workflow, and an offline-mock GenAI support service.

| Module | Path | What it is |
|---|---|---|
| 1 — Data Pipeline (25 marks) | [`data_pipeline/`](data_pipeline/) | Scrape books.toscrape.com → clean → GBP→INR at fixed 1 GBP = 105.50 INR → normalized SQLite (PK/FK) → 6 SQL queries → `read_sql` vs `pd.merge` equivalence proof |
| 2 — Analytics Pipeline (50 marks) | [`analytics/`](analytics/) | Titanic: one network load → committed `titanic.csv` → profiling/cleaning/EDA (threshold rule, IQR, masking, heatmap, 5-chart story, z-score check) → stratified split → 3 classifiers + imbalance comparison + GridSearch/OOB + fare regression → comparison table → `joblib` full pipeline |
| 3 — Support Assistant (25 marks) | [`support_assistant/`](support_assistant/) | 8-doc corpus → all-MiniLM-L6-v2 embeddings → ChromaDB → LangGraph 3+ node intent router with `MOCK_LLM` branches → Pydantic `answer/sources/confidence` → FastAPI `POST /ask` → Dockerfile |

## Quick start

```bash
pip install requests beautifulsoup4 pandas seaborn matplotlib scikit-learn \
            imbalanced-learn joblib fastapi uvicorn pydantic langgraph \
            chromadb sentence-transformers

# Module 1 (needs internet for the scrape)
python data_pipeline/pipeline.py

# Module 2 (first run needs internet for sns.load_dataset; later runs offline)
python analytics/01_eda.py
python analytics/02_modeling.py

# Module 3 (fully offline; MOCK_LLM defaults to mock mode)
cd support_assistant
python ingest.py
uvicorn main:app --port 7860
# POST /ask {"query": "What is the delivery fee below INR 149?"}
```

Per-module design decisions, outputs, and interpretations live in each
module's own README:

- [`data_pipeline/README.md`](data_pipeline/README.md)
- [`analytics/README.md`](analytics/README.md)
- [`support_assistant/README.md`](support_assistant/README.md)

## Key fixed constants (project-defined, graded)

- **Currency baseline:** 1 GBP = **105.50 INR** (Module 1) — fixed rate,
  no API, no date reference.
- **LLM toggle:** `MOCK_LLM` (Module 3) — unset/`1` = deterministic mock
  (graded default); `0` = optional real-LLM extension.
