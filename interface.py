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
    page_title="RAG Chat System",
    page_icon="💬",
    layout="wide",
)

st.title("💬 RAG Chat System")

# =========================
# Helpers
# =========================
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_chat_history(chats):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chats, f, indent=2, ensure_ascii=False)

def new_chat():
    cid = str(uuid.uuid4())
    st.session_state.chats[cid] = []
    st.session_state.active_chat_id = cid
    save_chat_history(st.session_state.chats)

def delete_chat(cid):
    st.session_state.chats.pop(cid, None)
    if not st.session_state.chats:
        new_chat()
    else:
        st.session_state.active_chat_id = next(iter(st.session_state.chats))
    save_chat_history(st.session_state.chats)

def chat_title(msgs):
    for m in msgs:
        if m["role"] == "user" and m["content"].strip():
            return m["content"][:30]
    return "New chat"

# =========================
# Session State
# =========================
if "chats" not in st.session_state:
    st.session_state.chats = load_chat_history()

if "active_chat_id" not in st.session_state:
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

    if c2.button("🗑️ Clear All", use_container_width=True):
        st.session_state.chats = {}
        new_chat()
        st.rerun()

    st.divider()

    for cid, msgs in list(st.session_state.chats.items()):
        title = chat_title(msgs)
        active = cid == st.session_state.active_chat_id

        r1, r2 = st.columns([0.82, 0.18])
        if r1.button(("➡️ " if active else "") + title, key=f"open_{cid}", use_container_width=True):
            st.session_state.active_chat_id = cid
            st.rerun()

        if r2.button("❌", key=f"del_{cid}", use_container_width=True):
            delete_chat(cid)
            st.rerun()

    st.divider()

    st.header("⚙️ Retrieval Settings")

    use_rag = st.toggle("Use RAG", value=True)

    top_k = st.slider("Top-K", 1, 10, 3, disabled=not use_rag)
    multiquery = st.checkbox("Multi-Query", value=False, disabled=not use_rag)
    hyde = st.checkbox("HyDE", value=False, disabled=not use_rag)

    st.divider()

    st.header("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "PDF / CSV / PPTX",
        type=["pdf", "csv", "pptx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        files_to_upload = [
            ("files", (f.name, f.getvalue()))
            for f in uploaded_files
            if f.name not in st.session_state.uploaded_file_names
        ]

        if files_to_upload:
            with st.spinner("Uploading & indexing…"):
                r = requests.post(f"{API_BASE}/upload", files=files_to_upload, timeout=300)

            if r.status_code == 200:
                st.success("Documents indexed")
                for _, (name, _) in files_to_upload:
                    st.session_state.uploaded_file_names.add(name)
            else:
                st.error(r.text)

            st.rerun()

    st.divider()
    st.header("🗂️ Indexed Documents")

    try:
        docs = requests.get(f"{API_BASE}/documents", timeout=5).json()
        for d in docs:
            c1, c2 = st.columns([0.85, 0.15])
            c1.write(d)
            if c2.button("🗑️", key=f"doc_{d}"):
                requests.delete(f"{API_BASE}/documents/{d}", timeout=10)
                st.session_state.uploaded_file_names.discard(d)
                st.rerun()
    except Exception:
        st.caption("Backend unavailable")

# =========================
# Main Chat Panel
# =========================
messages = st.session_state.chats[st.session_state.active_chat_id]

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        meta = msg.get("meta", {})
        if meta:
            if meta.get("citations"):
                with st.expander("Sources"):
                    for c in meta["citations"]:
                        st.write(f"- {c['source']} (page {c['page']})")

            if meta.get("queries"):
                with st.expander("Multi-Query Expansions"):
                    for q in meta["queries"]:
                        st.write(q)

            if meta.get("hyde"):
                with st.expander("HyDE – Hypothetical Documents"):
                    for h in meta["hyde"]:
                        st.markdown(f"**Original Query:** {h['original']}")
                        st.write(h["hyde"])

# =========================
# Chat Input
# =========================
question = st.chat_input("Ask a question")

if question and question.strip():
    messages.append({"role": "user", "content": question})
    save_chat_history(st.session_state.chats)

    payload = {
        "question": question,
        "top_k": top_k,
        "mmr": True,  # always enabled internally
        "multiquery": multiquery,
        "hyde": hyde,
    }

    endpoint = "/rag/ask" if use_rag else "/llm/ask"

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=120)

        if r.status_code == 200:
            data = r.json()
            answer = data.get("answer", "")
            st.markdown(answer)

            messages.append({
                "role": "assistant",
                "content": answer,
                "meta": data if use_rag else {},
            })
            save_chat_history(st.session_state.chats)
        else:
            st.error(r.text)

    st.rerun()
