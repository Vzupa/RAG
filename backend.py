import os
import tempfile
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

# Allowed file types for ingestion
ALLOWED_SUFFIXES = {".pdf", ".csv", ".pptx", ".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi", ".mp3"}

app = FastAPI(title="RAG Pipeline API", version="0.1.0")


class QueryRequest(BaseModel):
    question: Optional[str] = None
    multiquery: Optional[bool] = None
    hyde: Optional[bool] = None
    top_k: Optional[int] = None
    mmr: Optional[bool] = None


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    # Upload one or more PDF/CSV/PPTX files, process them, and update the vector store.
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
            # Use a temporary file to handle the upload
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name
                tmp.write(await file.read())

            # Process the file and add it to the RAG pipeline
            stats = add_file_to_rag(tmp_path, source_name=file.filename)
            stats_list.append({"filename": file.filename, "stats": stats})

        except Exception as exc:
            # Provide a more informative error message
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
    # Run a retrieval-augmented query.
    if not req.question:
        raise HTTPException(status_code=400, detail="A 'question' is required.")
    try:
        result = rag_query(
            question=req.question,
            k=req.top_k,
            mmr=req.mmr,
            multiquery=req.multiquery,
            hyde=req.hyde,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/llm/ask")
def llm(req: QueryRequest):
    # Run a base LLM-only query (no retrieval).
    if not req.question:
        raise HTTPException(status_code=400, detail="A 'question' is required.")
    try:
        answer = llm_only(question=req.question)
        return {"answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/documents", response_model=List[str])
def documents():
    # Return a list of indexed documents.
    try:
        return list_source_files()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    # Delete a document and its chunks from the vector store.
    try:
        result = delete_source_file(filename)
        return {"message": f"'{filename}' deleted", "result": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# To run locally:
# uvicorn backend:app --reload
