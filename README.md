# RAG Pipeline (PDF/CSV/PPTX)

FastAPI wrapper around a LangChain-based RAG pipeline with multi-query + HyDE retrieval. Supports ingesting PDF, CSV, and PPTX; runs RAG or LLM-only for comparison.

## Setup
1) Create/activate venv (example):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2) Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3) Add your OpenAI key to `.env`:
   ```
   OPENAI_API_KEY=sk-...
   ```

## Config
- `config.json` controls models, temps, retrieval params, chunking, and prompts.
- Default LLM: `gpt-4o-mini`, temp `0.0`; embedding: `text-embedding-3-small`.
- Retrieval defaults: `k=4`; multi-query and HyDE enabled; max 8 context docs.
- Prompts for multi-query, HyDE, and final answer live in `config.json`.

## Run API
```bash
uvicorn fastapi_app:app --reload
```

## Endpoints
- `GET /health` — health check.
- `POST /ingest` — form-data file upload (`file=@your.pdf|csv|pptx`); ingests and updates vector store.
- `POST /rag` — JSON: `{"question": "..."}`; returns answer + sources.
- `POST /llm` — JSON: `{"question": "..."}`; LLM-only (no retrieval).

## Reset vector store
Delete the persisted Chroma directory:
```bash
rm -rf Data/vector_store
```
Re-ingest documents afterwards.

## Core methods (RAG_pipeline.py)
- `add_file_to_rag(file_path, source_name=None)` — load PDF/CSV/PPTX, chunk, embed, and update Chroma. `source_name` tags the original filename in metadata.
- `rag_query(question, k=None, model=None, temperature=None)` — multi-query + HyDE retrieval, dedup context, answer with LLM and cite sources.
- `llm_only(question, model=None, temperature=None)` — LLM without retrieval for hallucination comparison.

Advanced retrieval implemented:
- Multi-query reformulations (configurable count).
- HyDE synthetic answer generation for retrieval.
- Dedup and context cap to avoid overloading the LLM.
