import json
import os
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv

# =========================
# Config
# =========================
load_dotenv()
API_BASE = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
CHAT_HISTORY_FILE = "chat_history.json"

st.set_page_config(
    page_title="RAG Chat",
    page_icon="💬",
    layout="wide",
)

st.title("💬 RAG Chat System")

# =========================
# Helpers
# =========================
def load_chat_history():
    """Loads chat history from a JSON file."""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return {}
    return {}

def save_chat_history(chats):
    """Saves chat history to a JSON file."""
    try:
        with open(CHAT_HISTORY_FILE, "w") as f:
            json.dump(chats, f, indent=2)
    except IOError:
        st.error("Error: Could not save chat history.")


def new_chat():
    cid = str(uuid.uuid4())
    st.session_state.chats[cid] = []
    st.session_state.active_chat_id = cid
    save_chat_history(st.session_state.chats)


def chat_title(msgs):
    for m in msgs:
        if m["role"] == "user" and m["content"]:
            return m["content"][:30]
    return "New chat"


def delete_chat(cid):
    if cid in st.session_state.chats:
        del st.session_state.chats[cid]

    if st.session_state.active_chat_id == cid:
        if st.session_state.chats:
            st.session_state.active_chat_id = next(iter(st.session_state.chats))
        else:
            new_chat()
    save_chat_history(st.session_state.chats)


def delete_all_chats():
    st.session_state.chats = {}
    new_chat()  # This will also save the now empty chats

# =========================
# Session State
# =========================
if "chats" not in st.session_state:
    st.session_state.chats = load_chat_history()

if "active_chat_id" not in st.session_state:
    # Set the active chat to the first one, or create a new one
    if st.session_state.chats:
        st.session_state.active_chat_id = next(iter(st.session_state.chats))
    else:
        new_chat()

if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = set()

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("📚 Chats")

    c1, c2 = st.columns(2)
    if c1.button("➕ New Chat", use_container_width=True):
        new_chat()
        st.rerun()

    if c2.button("🗑️ Delete All", use_container_width=True):
        delete_all_chats()
        st.rerun()

    st.divider()

    # Use a copy to avoid issues with dictionary size changing during iteration
    for cid, msgs in list(st.session_state.chats.items()):
        title = chat_title(msgs)
        active = cid == st.session_state.active_chat_id

        r1, r2 = st.columns([0.8, 0.2])
        if r1.button(("➡️ " if active else "") + title, key=f"open_{cid}", use_container_width=True):
            st.session_state.active_chat_id = cid
            st.rerun()

        if r2.button("🗑️", key=f"del_{cid}", use_container_width=True):
            delete_chat(cid)
            st.rerun()

    st.divider()

    st.header("⚙️ Retrieval Settings")
    use_rag = st.toggle("Use RAG (Grounded answers)", value=True)

    top_k = st.slider(
        "Top-K Chunks",
        1,
        10,
        3,
        disabled=not use_rag
    )

    mmr = st.checkbox(
        "Use MMR (diversity)",
        value=True,
        disabled=not use_rag
    )

    if not use_rag:
        st.caption("⚠️ LLM-only mode (hallucinations possible)")

    st.divider()

    st.header("📄 Upload Document")
    uploaded_files = st.file_uploader(
        "PDF / CSV / PPTX / JGP / JPEG / PNG / MP4 / MOV / AVI / MP3",
        type=["pdf", "csv", "pptx", "jpg", "jpeg", "png", "mp4", "mov", "avi", "mp3"],
        accept_multiple_files=True
    )

    if uploaded_files:
        files_to_upload = []
        for uploaded in uploaded_files:
            if uploaded.name not in st.session_state.uploaded_file_names:
                files_to_upload.append(
                    ("files", (uploaded.name, uploaded.getvalue()))
                )

        if files_to_upload:
            r = requests.post(f"{API_BASE}/upload", files=files_to_upload, timeout=120)

            if r.status_code == 200:
                st.success("File(s) uploaded successfully.")
                for _, (name, _) in files_to_upload:
                    st.session_state.uploaded_file_names.add(name)
            else:
                st.error(f"Upload failed: {r.text}")

            st.rerun()

    st.divider()
    st.header("🗂️ Indexed Documents")

    try:
        docs = requests.get(f"{API_BASE}/documents", timeout=5).json()
        for f in docs:
            d1, d2 = st.columns([0.85, 0.15])
            d1.write(f)
            if d2.button("❌", key=f"doc_{f}"):
                requests.delete(f"{API_BASE}/documents/{f}", timeout=10)
                st.session_state.uploaded_file_names.discard(f)
                st.rerun()
    except Exception:
        st.caption("Unable to load documents")

    st.divider()

    try:
        health = requests.get(f"{API_BASE}/health", timeout=5).json()
        st.success(f"Model: {health['model']}")
        st.info(f"Docs indexed: {health['documents_loaded']}")
    except Exception:
        st.error("Backend offline")

# =========================
# Main Chat Panel
# =========================
# Ensure active_chat_id is valid, otherwise reset
if st.session_state.active_chat_id not in st.session_state.chats:
    if st.session_state.chats:
        st.session_state.active_chat_id = next(iter(st.session_state.chats))
    else:
        new_chat()

messages = st.session_state.chats[st.session_state.active_chat_id]

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("meta"):
            citations = msg["meta"].get("citations", [])
            retrieved = msg["meta"].get("retrieved_chunks", [])

            if citations:
                with st.expander("Sources"):
                    for c in citations:
                        st.write(f"- {c['source']} (page {c['page']})")

            if retrieved:
                with st.expander("Retrieved context"):
                    for r in retrieved:
                        st.markdown(
                            f"**{r['source']} – page {r['page']} (score {r['score']:.3f})**"
                        )
                        st.write(r["text"])

# =========================
# Chat Input
# =========================
question = st.chat_input("Ask a question")

if question:
    messages.append({"role": "user", "content": question})
    save_chat_history(st.session_state.chats)
    
    if use_rag:
        endpoint = f"{API_BASE}/rag/ask"
        payload = {"question": question, "top_k": top_k, "mmr": mmr}
    else:
        endpoint = f"{API_BASE}/llm/ask"
        payload = {"question": question}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            r = requests.post(endpoint, json=payload, timeout=120)

        if r.status_code == 200:
            data = r.json()
            answer = data.get("answer", "Sorry, something went wrong.")
            st.markdown(answer)
            
            # For RAG responses, the backend sends back metadata we can display
            meta = data if use_rag else None
            messages.append({
                "role": "assistant",
                "content": answer,
                "meta": meta
            })
            save_chat_history(st.session_state.chats)
        else:
            st.error(r.text)

    st.rerun()
