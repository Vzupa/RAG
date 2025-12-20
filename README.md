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
   # Also ensure langchain-chroma is installed for ChromaDB functionality
   # pip install langchain-chroma
   pip install streamlit requests python-dotenv # For frontend streamlit application
   ```
3) Add your OpenAI key to `.env`:
   ```
   OPENAI_API_KEY=sk-...
   ```

## Config
- `config.json` controls models, temps, retrieval params, chunking, and prompts.
- Default LLM: `gpt-4o-mini`, temp `0.0`; embedding: `text-embedding-3-small`.
- Retrieval defaults: `k=4`; multi-query and HyDE are disabled by default (toggle per request or in config); max 8 context docs.
- Prompts for multi-query, HyDE, and final answer live in `config.json`.

## Run API
```bash
uvicorn fastapi_app:app --reload
```

## Run Streamlit Frontend
```bash
streamlit run interface.py
```

## Endpoints
- `GET /health` — health check.
- `POST /upload` — form-data file upload (`files=@your.pdf|csv|pptx`); ingests and updates vector store. Now supports multiple files in one request.
- `GET /documents` — Returns a list of indexed document filenames.
- `DELETE /documents/{filename}` — Deletes a document and its chunks from the vector store.
- `POST /rag/ask` — JSON: `{"question": "...", "multiquery": optional bool, "hyde": optional bool}`; returns answer + sources (plus queries/hyde when enabled).
- `POST /llm/ask` — JSON: `{"question": "..."}`; LLM-only (no retrieval). No multiquery/HyDE flags here.

## Features & Improvements
- **Persistent Chat History**: Chat conversations in the Streamlit frontend are now saved to `chat_history.json` and persist across application restarts.
- **File Uploads**: The API's `/upload` endpoint and Streamlit's file uploader have been enhanced to correctly handle multiple file uploads, improving reliability.
- **Clearer Error Messaging**: API endpoints (`/rag/ask`, `/llm/ask`) now provide more specific error details for invalid requests.
- **ChromaDB Integration**: Updated to use `langchain_chroma` for improved compatibility and removed deprecated persistence calls. Document deletion is now more reliable, ensuring removed documents do not influence future queries.

## Reset vector store
Delete the persisted Chroma directory:
```bash
rm -rf Data/vector_store
```
Re-ingest documents afterwards.

## Core methods (RAG_pipeline.py)
- `add_file_to_rag(file_path, source_name=None)` — load PDF/CSV/PPTX, chunk, embed, and update Chroma. `source_name` tags the original filename in metadata.
- `rag_query(question, k=None, model=None, temperature=None, multiquery=None, hyde=None)` — multi-query + HyDE retrieval, dedup context, answer with LLM and cite sources. Only `multiquery`/`hyde` are user-toggled; model/temperature/k come from config.
- `llm_only(question, model=None, temperature=None)` — LLM without retrieval for hallucination comparison.
- `list_source_files()` — Returns a list of unique source filenames present in the vector store.
- `delete_source_file(filename)` — Deletes all chunks associated with a specific source file from the vector store.

Advanced retrieval implemented:
- Multi-query reformulations (configurable count).
- HyDE synthetic answer generation for retrieval.
- Dedup and context cap to avoid overloading the LLM.




flowchart TD
    %% Global Styles
    classDef storage fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef process fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef llm fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef user fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;

    %% -----------------------
    %% 1. INGESTION PIPELINE
    %% -----------------------
    subgraph Ingestion_Flow [Ingestion Pipeline]
        direction TB
        RawFiles[Raw Files] --> Router{Check File Type}
        
        %% Text Path
        Router -- PDF, CSV, PPTX --> TextLoader[Standard Loaders<br/>PyPDF, CSV, PPTX]
        
        %% Video/Audio Path
        Router -- MP4, AVI, MP3 --> MoviePy[MoviePy<br/>Extract Audio]
        MoviePy --> Whisper[OpenAI Whisper<br/>Transcribe Audio]
        
        %% Image Path
        Router -- JPG, PNG --> Vision[GPT-4o Vision<br/>Describe Image]
        
        %% Unification
        TextLoader & Whisper & Vision --> RawDocs[Raw Documents]
        RawDocs --> Splitter[Recursive Character<br/>Text Splitter]
        Splitter --> Embedder1[OpenAI Embeddings]
        Embedder1 --> VectorStore[(ChromaDB<br/>Vector Store)]:::storage
    end

    %% -----------------------
    %% 2. QUERY PIPELINE
    %% -----------------------
    subgraph Query_Flow [RAG Query Pipeline]
        direction TB
        User((User)):::user --> Question[Input Question]
        Question --> Optimization{Optimization<br/>Config}
        
        %% Query Expansion
        Optimization -- Multi-Query --> MultiQ[Generate 3-5 Variants]:::llm
        Optimization -- HyDE --> HydeGen[Generate Hypothetical Answer]:::llm
        Optimization -- None --> RawQ[Raw Question]
        
        %% Retrieval
        MultiQ & HydeGen & RawQ --> Embedder2[OpenAI Embeddings]
        Embedder2 --> Retriever[Retriever Search<br/>MMR or Similarity]
        Retriever --> VectorStore
        VectorStore -- Return Chunks --> Dedup[Deduplication Logic<br/>Source/Loc/Content]
        
        %% Generation
        Dedup --> ContextWindow[Context Window]
        ContextWindow --> FinalPrompt[Construct Prompt]
        FinalPrompt --> ChatModel[ChatOpenAI<br/>GPT-4o/3.5]:::llm
        ChatModel --> FinalAnswer[Final Answer]:::user
    end

    %% Link the flows
    VectorStore -.-> Retriever