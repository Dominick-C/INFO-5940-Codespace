# chat_with_pdf.py
# Streamlit RAG app built from the class example (chunks 1–23), adapted for multi-file upload and UI.

import os
import uuid
import tempfile
from typing import List, TypedDict

import streamlit as st

# =========================
# Read credentials from environment (set in .devcontainer/devcontainer.json)
# =========================
API_KEY = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = (
    os.environ.get("OPENAI_BASE_URL")
    or os.environ.get("BASE_URL")
    or "https://api.ai.it.cornell.edu"
)
BASE_URL = BASE_URL.rstrip("/")  # normalize

# =========================
# LangChain + Chroma
# =========================
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# ---- Chroma clients (memory vs disk) ----
from chromadb import Client as ChromaClient
# from chromadb import PersistentClient as ChromaPersistentClient  # not used now

# ---- Loaders, splitters, prompts ----
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# ---- LangGraph (retrieve -> generate) ----
from langgraph.graph import START, StateGraph


# =========================
# Streamlit UI Config
# =========================
st.set_page_config(page_title="INFO 5940 — RAG Chat", layout="wide")
st.title("📄🔎 RAG Chat (LangChain + Chroma)")

# Friendly warning if env isn’t populated after rebuild
if not API_KEY:
    st.warning(
        "OPENAI_API_KEY / API_KEY is not set in the environment. "
        "Update `.devcontainer/devcontainer.json` or Codespaces Secrets, then Rebuild Container."
    )

with st.sidebar:
    st.header("Settings")

    # Persistence mode—this is the key toggle per TA advice
    persist_mode = st.selectbox(
        "Vector DB mode",
        ["In-memory (recommended)", "Disk (/tmp per session)"],
        index=0,
        help="In-memory avoids read-only DB errors on Codespaces. Disk mode writes under /tmp with a fresh folder per session."
    )

    chunk_size = st.number_input("Chunk size", 100, 4000, 800, 50)
    chunk_overlap = st.number_input("Chunk overlap", 0, 1000, 120, 10)
    k_results = st.number_input("Top-k retrieval", 1, 50, 6, 1)

    # Base path only used in disk mode (we’ll make a unique subfolder per run)
    base_persist_dir = st.text_input("Base persist dir (disk mode only)", "/tmp/chroma_db")

    colA, colB = st.columns(2)
    with colA:
        reset_btn = st.button("Reset chat", use_container_width=True)
    with colB:
        clear_index_btn = st.button("Clear index", use_container_width=True)

# session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list[dict(role, content, sources?)]
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "indexed" not in st.session_state:
    st.session_state["indexed"] = False
# active collection and run path
if "collection_name" not in st.session_state:
    st.session_state["collection_name"] = f"rag_{uuid.uuid4().hex[:8]}"
if "run_persist_dir" not in st.session_state:
    # For disk mode we’ll create a unique writable subfolder under /tmp
    st.session_state["run_persist_dir"] = os.path.join(base_persist_dir, f"run_{uuid.uuid4().hex[:8]}")

# React to mode/path changes live
if persist_mode == "Disk (/tmp per session)":
    # ensure /tmp-based unique path is set (fresh each app start)
    if not st.session_state.get("run_persist_dir"):
        st.session_state["run_persist_dir"] = os.path.join(base_persist_dir, f"run_{uuid.uuid4().hex[:8]}")
else:
    # in-memory has no persist dir
    st.session_state["run_persist_dir"] = None

if reset_btn:
    st.session_state["messages"].clear()
    st.rerun()

if clear_index_btn:
    # Don’t delete any folders; just rotate to a new collection & (if disk) a new run folder
    st.session_state["vectorstore"] = None
    st.session_state["indexed"] = False
    st.session_state["messages"] = []
    st.session_state["collection_name"] = f"rag_{uuid.uuid4().hex[:8]}"
    if persist_mode == "Disk (/tmp per session)":
        st.session_state["run_persist_dir"] = os.path.join(base_persist_dir, f"run_{uuid.uuid4().hex[:8]}")
    st.success("Index cleared (new collection/run will be created on next build).")
    st.rerun()


# =========================
# Model & Prompt (chunks 1,2,18) — uses env creds
# =========================
llm = ChatOpenAI(
    model="openai.gpt-4o",        # NOTE: 'openai.' prefix required by the Cornell gateway
    temperature=0.2,
    api_key=API_KEY,
    base_url=BASE_URL,
)

# Include {history} so multi-turn context is considered
qa_template = """
You are a grounded question-answering assistant. Use ONLY the supplied Context.
If the answer is not present, say: "I don’t know from the provided documents."
Keep the answer to at most three concise sentences.

Conversation History (most recent last):
{history}

Question:
{question}

Context:
{context}

Answer (then add a line starting with 'Sources:' listing filename and page numbers you used):
"""
prompt = PromptTemplate.from_template(qa_template)


# =========================
# Helpers: load, chunk, index
# =========================
def load_txt_file(file) -> List[Document]:
    raw = file.getvalue()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="ignore")
    return [Document(page_content=text, metadata={
        "filename": file.name, "filetype": "txt", "page": 1
    })]


def load_pdf_file(file) -> List[Document]:
    # PyPDFLoader expects a path; write to temp then load
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        for d in docs:
            d.metadata["filename"] = file.name
            d.metadata["filetype"] = "pdf"
            d.metadata["page"] = d.metadata.get("page", 1)
        return docs
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def chunk_documents(documents: List[Document], size: int, overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = splitter.split_documents(documents)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


# Create ONE embeddings object with env creds; reuse it
embeddings = OpenAIEmbeddings(
    model="openai.text-embedding-3-small",  # safer bet on most gateways
    api_key=API_KEY,
    base_url=BASE_URL,
)


def get_or_create_vectorstore(persist_mode_label: str) -> Chroma:
    """
    Build a Chroma vectorstore either in-memory (recommended) or on-disk under /tmp.
    We never delete/overwrite paths—each "clear index" just rotates the collection and/or run dir.
    """
    collection = st.session_state["collection_name"]

    if persist_mode_label == "In-memory (recommended)":
        # Pure in-memory chroma client (no disk writes)
        client = ChromaClient()
        vs = Chroma(
            client=client,
            collection_name=collection,
            embedding_function=embeddings,
        )
        return vs

    # Disk mode under /tmp — create a fresh per-session folder
    run_dir = st.session_state["run_persist_dir"]
    assert run_dir, "run_persist_dir must be set in disk mode"
    os.makedirs(run_dir, exist_ok=True)
    vs = Chroma(
        collection_name=collection,
        persist_directory=run_dir,
        embedding_function=embeddings,
    )
    return vs


def add_chunks_to_index(vectorstore: Chroma, chunks: List[Document]):
    """Add documents; no .persist() required on the LangChain wrapper."""
    if not chunks:
        return
    vectorstore.add_documents(chunks)


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# =========================
# LangGraph state & nodes (chunks 19–21)
# =========================
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    sources: List[dict]  # [{"filename":..., "page":..., "chunk_id":..., "score":...}]


def _history_str(max_turns: int = 6) -> str:
    """Build a compact history string from the last few turns."""
    hist = []
    for m in st.session_state.get("messages", [])[-max_turns:]:
        role = "User" if m["role"] == "user" else "Assistant"
        hist.append(f"{role}: {m['content']}")
    return "\n".join(hist)


def retrieve(state: State):
    question = state["question"]
    # Use similarity_search_with_score to capture scores + metadata
    results = st.session_state["vectorstore"].similarity_search_with_score(question, k=k_results)
    docs = [doc for doc, _ in results]
    sources = []
    for doc, score in results:
        meta = doc.metadata or {}
        sources.append({
            "filename": meta.get("filename", "unknown"),
            "page": meta.get("page", None),
            "chunk_id": meta.get("chunk_id", None),
            "score": float(score),
        })
    return {"context": docs, "sources": sources}


def generate(state: State):
    ctx = format_docs(state["context"])
    hist = _history_str()
    messages = prompt.invoke({"question": state["question"], "context": ctx, "history": hist})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


# =========================
# Upload → Index area
# =========================
st.subheader("1) Upload documents (.txt and/or .pdf)")

uploads = st.file_uploader(
    "You can select multiple files",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

docs_all: List[Document] = []

if uploads:
    with st.spinner("Reading files..."):
        for f in uploads:
            if f.name.lower().endswith(".txt"):
                docs_all.extend(load_txt_file(f))
            elif f.name.lower().endswith(".pdf"):
                docs_all.extend(load_pdf_file(f))

    st.write(f"Loaded **{len(docs_all)}** document units (pages for PDFs).")
    with st.expander("Preview first two docs"):
        for d in docs_all[:2]:
            st.write(f"**{d.metadata.get('filename')}** — page {d.metadata.get('page', 1)}")
        if docs_all:
            st.code(docs_all[0].page_content[:500] + ("..." if len(docs_all[0].page_content) > 500 else ""))

    if st.button("➕ Build / Update Index", type="primary"):
        if not docs_all:
            st.warning("No documents to index.")
        else:
            with st.spinner("Chunking and indexing..."):
                chunks = chunk_documents(docs_all, chunk_size, chunk_overlap)
                vs = get_or_create_vectorstore(persist_mode)
                add_chunks_to_index(vs, chunks)
                st.session_state["vectorstore"] = vs
                st.session_state["indexed"] = True
            st.success(f"Indexed {len(chunks)} chunks into Chroma.")
else:
    st.caption("Tip: Upload a few small files from the /data folder to test.")


# =========================
# Chat area
# =========================
st.subheader("2) Chat with your documents")

# show history
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.write(f"- **{s['filename']}** (page {s.get('page', '?')}, chunk {s.get('chunk_id', '?')}) — score {s['score']:.3f}")

# input
question = st.chat_input("Ask a question about your uploaded documents…")

if question:
    if not st.session_state["indexed"] or st.session_state["vectorstore"] is None:
        st.error("No index yet. Upload files and click **Build / Update Index** first.")
    else:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating…"):
                result = graph.invoke({"question": question})
                answer = result["answer"]
                sources = result.get("sources", [])
                st.markdown(answer)

                # Show matched chunks (the actual retrieved text)
                if result.get("context"):
                    with st.expander("Show matched chunks"):
                        for doc in result["context"]:
                            meta = doc.metadata
                            st.markdown(f"**{meta.get('filename','?')} — page {meta.get('page','?')} (chunk {meta.get('chunk_id','?')})**")
                            snippet = doc.page_content
                            if len(snippet) > 1000:
                                snippet = snippet[:1000] + "..."
                            st.code(snippet)

                # Sources section
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.write(f"- **{s['filename']}** (page {s.get('page', '?')}, chunk {s.get('chunk_id', '?')}) — score {s['score']:.3f}")
            st.session_state["messages"].append({"role": "assistant", "content": answer, "sources": sources})
