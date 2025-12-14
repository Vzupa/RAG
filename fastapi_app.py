import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from RAG_pipeline import add_file_to_rag, llm_only, rag_query

# Allowed file types for ingestion
ALLOWED_SUFFIXES = {".pdf", ".csv", ".pptx"}

app = FastAPI(title="RAG Pipeline API", version="0.1.0")


class QueryRequest(BaseModel):
    question: str


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    # Upload a PDF/CSV/PPTX, process it, and update the vector store.
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())

        stats = add_file_to_rag(tmp_path, source_name=file.filename)
        return {"message": "ingested", "stats": stats}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/rag/ask")
def rag(req: QueryRequest):
    # Run a retrieval-augmented query.
    try:
        result = rag_query(
            question=req.question
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/llm/ask")
def llm(req: QueryRequest):
    # Run a base LLM-only query (no retrieval).
    try:
        answer = llm_only(
            question=req.question
        )
        return {"answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# To run locally:
# uvicorn fastapi_app:app --reload
