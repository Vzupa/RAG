import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from RAG_pipeline import (
    CONFIG,
    add_file_to_rag,
    delete_source_file,
    get_vector_store_stats,
    list_source_files,
    llm_only,
    rag_query,
)

# --- API configuration ---
ALLOWED_SUFFIXES = {".pdf", ".csv", ".pptx", ".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi", ".mp3"}
app = FastAPI(title="RAG Pipeline API", version="0.1.0")


# --- Request models ---
class QueryRequest(BaseModel):
    """Payload for question-based endpoints."""
    question: Optional[str] = None
    multiquery: Optional[bool] = None
    hyde: Optional[bool] = None
    top_k: Optional[int] = None


# --- Routes ---
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """
    Validate and ingest uploaded files into the vector store.

    Args:
        files (List[UploadFile]): One or more incoming files to ingest.

    Returns:
        dict: Summary message and per-file ingestion stats.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    stats_list = []
    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {suffix}"
            )

        tmp_path = None
        try:
            # Use a temporary file to stream the upload content
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name
                tmp.write(await file.read())

            # Ingest into the RAG pipeline and capture stats for the response payload
            stats = add_file_to_rag(tmp_path, source_name=file.filename)
            stats_list.append({"filename": file.filename, "stats": stats})

        except Exception as exc:
            # Surface file-specific errors without leaking stack traces
            raise HTTPException(
                status_code=400,
                detail=f"Error processing '{file.filename}': {exc}",
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return {"message": f"{len(files)} file(s) uploaded successfully.", "details": stats_list}


@app.post("/rag/ask")
def rag(req: QueryRequest):
    """
    Run a retrieval-augmented query with optional Multi-Query and HyDE.

    Args:
        req (QueryRequest): Question payload with optional knobs.

    Returns:
        dict: Answer plus sources and optional query/HyDE details.
    """
    if not req.question:
        raise HTTPException(status_code=400, detail="A 'question' is required.")
    try:
        start_time = time.perf_counter()
        result = rag_query(
            question=req.question,
            k=req.top_k,
            multiquery=req.multiquery,
            hyde=req.hyde,
        )
        end_time = time.perf_counter()
        result["backend_latency"] = end_time - start_time
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/llm/ask")
def llm(req: QueryRequest):
    """
    Run an LLM-only query without retrieval for comparison.

    Args:
        req (QueryRequest): Question payload.

    Returns:
        dict: Answer text only.
    """
    if not req.question:
        raise HTTPException(status_code=400, detail="A 'question' is required.")
    try:
        start_time = time.perf_counter()
        result = llm_only(question=req.question)
        end_time = time.perf_counter()
        result["backend_latency"] = end_time - start_time
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/documents", response_model=List[str])
def documents():
    """
    List filenames currently indexed in the vector store.

    Returns:
        List[str]: Sorted list of source filenames.
    """
    try:
        return list_source_files()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """
    Delete all chunks associated with a source filename.

    Args:
        filename (str): Name of the source file to remove.

    Returns:
        dict: Confirmation message and deleted chunk count.
    """
    try:
        result = delete_source_file(filename)
        return {"message": f"'{filename}' deleted", "result": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# To run locally:
# uvicorn backend:app --reload
