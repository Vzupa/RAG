import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pptx import Presentation

# Load environment variables (.env for OPENAI_API_KEY)
load_dotenv()

# Load config from JSON so prompts/params are easy to tweak
CONFIG_PATH = Path("config.json")
if not CONFIG_PATH.exists():
    raise FileNotFoundError("config.json not found. Please create it before running the pipeline.")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# Vector store location and supported file types
DATA_DIR = Path("Data")
VECTOR_DIR = DATA_DIR / "vector_store"
SUPPORTED_EXTS = {"pdf", "pptx", "csv"}

# Ensure vector store directory exists
def _ensure_dirs():
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# Require OpenAI API key
def _require_openai_key():
    # Guard: ensure API key is available before any OpenAI call
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("Error: OPENAI_API_KEY is missing. Add it to your environment or .env file.")
        raise ValueError("OPENAI_API_KEY not set")


# Basic PPTX text extraction without unstructured dependency.
def _pptx_to_documents(file_path: Path) -> List[Document]:
    # Extract text from PPTX slides and attach metadata
    prs = Presentation(str(file_path))
    docs: List[Document] = []
    for idx, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                texts.append(shape.text)
        if not texts:
            continue
        content = "\n".join(texts)
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "slide": idx,
                    "type": "pptx",
                },
            )
        )
    return docs

# Load file as documents, these can be pdf, csv, or pptx
def _load_file_as_documents(file_path: Path) -> List[Document]:
    # Dispatch loader by file extension and tag basic metadata
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        docs = PyPDFLoader(str(file_path)).load()
        for d in docs:
            d.metadata["type"] = "pdf"
        return docs
    if ext == ".csv":
        docs = CSVLoader(str(file_path), encoding="utf-8").load()
        for idx, d in enumerate(docs, start=1):
            # Ensure row number is recorded; CSVLoader may already add row indices.
            d.metadata.setdefault("row", idx)
            d.metadata["type"] = "csv"
        return docs
    if ext == ".pptx":
        return _pptx_to_documents(file_path)
    raise ValueError(f"Unsupported file type: {ext}")

# Split documents into chunks
def _split_documents(documents: List[Document]):
    # Split documents into overlapping chunks for embedding
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["CHUNK_SIZE"],
        chunk_overlap=CONFIG["CHUNK_OVERLAP"],
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def _load_or_create_vector_store(embedding_model: OpenAIEmbeddings):
    # If the directory exists and has a Chroma collection, loading will reuse it
    if VECTOR_DIR.exists():
        return Chroma(persist_directory=str(VECTOR_DIR), embedding_function=embedding_model)
    return None


def _build_llm(model_override: Optional[str] = None, temperature_override: Optional[float] = None):
    # Convenience to build an LLM client with optional overrides
    return ChatOpenAI(
        model=model_override or CONFIG["LLM_MODEL"],
        temperature=CONFIG["LLM_TEMPERATURE"] if temperature_override is None else temperature_override,
    )


def _generate_multi_queries(question: str, llm: ChatOpenAI) -> List[str]:
    # Generate multiple reformulations of the user query
    prompt = CONFIG["PROMPT_MULTIQUERY"].format(question=question)
    resp = llm.invoke(prompt)
    variants = [line.strip() for line in resp.content.splitlines() if line.strip()]
    if not variants:
        return [question]
    # Limit to configured count
    return [question] + variants[: CONFIG.get("MULTIQUERY_VARIANTS", 2)]


def _generate_hyde_text(query: str, llm: ChatOpenAI) -> str:
    # Generate a hypothetical answer to embed for retrieval (HyDE)
    prompt = CONFIG["PROMPT_HYDE"].format(question=query)
    resp = llm.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)


def _dedup_docs(docs: List[Document], max_docs: int) -> List[Document]:
    # Drop duplicates by source+position+content prefix and cap the count
    seen = set()
    unique = []
    for doc in docs:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("slide"),
            doc.metadata.get("row"),
            doc.page_content[:200],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
        if len(unique) >= max_docs:
            break
    return unique


def _normalize_source_meta(doc: Document):
    # Normalize source name to filename and page/slide/row to human-friendly numbers
    raw_source = doc.metadata.get("source")
    source_name = Path(raw_source).name if raw_source else None

    page_val = doc.metadata.get("page")
    slide_val = doc.metadata.get("slide")
    row_val = doc.metadata.get("row")
    page_or_loc = page_val if page_val is not None else slide_val if slide_val is not None else row_val

    # If page is numeric, convert to int and +1 to show human page numbers (PyPDFLoader is 0-based)
    display_loc = page_or_loc
    if isinstance(page_or_loc, (int, float)) and page_or_loc >= 0:
        display_loc = int(page_or_loc) + 1
    else:
        # try parsing string digits
        try:
            num = int(str(page_or_loc))
            display_loc = num + 1
        except Exception:
            display_loc = page_or_loc

    return {
        "source": source_name,
        "page": display_loc if page_val is not None else None,
        "type": doc.metadata.get("type"),
        "location": display_loc,
    }


def add_file_to_rag(file_path: str, source_name: Optional[str] = None) -> Dict[str, int]:
    # Reads the file (PDF/CSV/PPTX), chunks it, embeds with OpenAI, and updates the Chroma store.
    _ensure_dirs()
    _require_openai_key()

    path_obj = Path(file_path)
    if not path_obj.exists():
        print("Error: file not found.")
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path_obj.suffix.lower().lstrip(".")
    if ext not in SUPPORTED_EXTS:
        print("Error: unsupported file type.")
        raise ValueError(f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTS)}")

    documents = _load_file_as_documents(path_obj)
    # Tag source as the original filename if provided (or use the incoming path name)
    src_name = Path(source_name).name if source_name else path_obj.name
    for d in documents:
        d.metadata["source"] = src_name
    chunks = _split_documents(documents)

    embedding = OpenAIEmbeddings(model=CONFIG["EMBEDDING_MODEL"])
    vector_store = _load_or_create_vector_store(embedding)

    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=str(VECTOR_DIR),
        )

    vector_store.persist()
    total = vector_store._collection.count() if hasattr(vector_store, "_collection") else -1

    return {"chunks_added": len(chunks), "collection_size": total}


def rag_query(
    question: str,
    k: Optional[int] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    multiquery: Optional[bool] = None,
    hyde: Optional[bool] = None,
):
    # Runs a retrieval-augmented query using the persisted Chroma store.
    _require_openai_key()
    embedding = OpenAIEmbeddings(model=CONFIG["EMBEDDING_MODEL"])
    if not VECTOR_DIR.exists():
        print("Error: vector store not found. Ingest documents first.")
        raise ValueError("Vector store not found. Ingest documents first.")

    vector_store = Chroma(persist_directory=str(VECTOR_DIR), embedding_function=embedding)
    search_kwargs = {
        "k": k or CONFIG["RAG_SEARCH_K"],
    }
    if CONFIG.get("RAG_FETCH_K"):
        search_kwargs["fetch_k"] = CONFIG["RAG_FETCH_K"]
    if CONFIG.get("RAG_SCORE_THRESHOLD") is not None:
        search_kwargs["score_threshold"] = CONFIG["RAG_SCORE_THRESHOLD"]
    if CONFIG.get("RAG_FILTER_METADATA"):
        search_kwargs["filter"] = CONFIG["RAG_FILTER_METADATA"]
    retriever = vector_store.as_retriever(
        search_type=CONFIG["RAG_SEARCH_TYPE"],
        search_kwargs=search_kwargs,
    )

    llm = _build_llm(model_override=model, temperature_override=temperature)

    # Determine feature flags (request overrides config)
    use_multiquery = CONFIG.get("MULTIQUERY_ENABLED") if multiquery is None else multiquery
    use_hyde = CONFIG.get("HYDE_ENABLED") if hyde is None else hyde

    # Build query variants (multi-query) if enabled
    if use_multiquery:
        queries = _generate_multi_queries(question, llm)
    else:
        queries = [question]

    all_docs: List[Document] = []
    hyde_details = []
    for q in queries:
        query_text = q
        if use_hyde:
            query_text = _generate_hyde_text(q, llm)
            hyde_details.append({"original": q, "hyde": query_text})
        else:
            hyde_details.append({"original": q, "hyde": None})
        docs = retriever.invoke(query_text)
        all_docs.extend(docs)

    # Dedup and cap context docs
    context_docs = _dedup_docs(all_docs, max_docs=CONFIG.get("MAX_CONTEXT_DOCS", 8))

    # Format prompt
    context_text = "\n\n".join(doc.page_content for doc in context_docs)
    prompt = CONFIG["PROMPT_ANSWER"].format(context=context_text, question=question)
    response = llm.invoke(prompt)

    # Build a deduped source list for the response with normalized page numbers and filenames
    sources = []
    seen_sources = set()
    for doc in context_docs:
        normalized = _normalize_source_meta(doc)
        key = (normalized["source"], normalized["page"], normalized.get("type"))
        if key in seen_sources:
            continue
        seen_sources.add(key)
        sources.append(
            {
                "source": normalized["source"],
                "page": normalized["page"],
                "type": normalized.get("type"),
            }
        )
    answer_text = response.content if hasattr(response, "content") else str(response)
    result = {
        "answer": answer_text,
        "sources": sources,
    }
    if use_multiquery:
        result["queries"] = queries
    if use_hyde:
        result["hyde"] = hyde_details
    return result


def llm_only(
    question: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
):
    # Calls the LLM directly without retrieval for hallucination comparison.
    _require_openai_key()
    llm = ChatOpenAI(
        model=model or CONFIG["LLM_MODEL"],
        temperature=CONFIG["LLM_TEMPERATURE"] if temperature is None else temperature,
    )
    response = llm.invoke(question)
    return response.content if hasattr(response, "content") else response
