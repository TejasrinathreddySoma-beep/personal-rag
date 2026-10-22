import os

import streamlit as st

import loaders
import gategrind
import rag
from config import GEMINI_MODEL, OLLAMA_MODEL
from vectorstore import VectorStore

st.set_page_config(page_title="Personal RAG", page_icon="📚", layout="wide")


@st.cache_resource
def get_store():
    return VectorStore()


store = get_store()


def show_sources(hits):
    with st.expander("Sources"):
        for i, h in enumerate(hits, 1):
            page = f" · page {h['page']}" if h.get("page") else ""
            st.markdown(f"**[{i}] {h['source']}**{page} · distance {h['score']:.2f}")
            st.caption(h["text"][:300] + "...")


def report(counts):
    for source, n in counts.items():
        st.success(f"{source}: {n} chunks")


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    choice = st.radio("LLM", [f"Local (Ollama: {OLLAMA_MODEL})", f"Gemini ({GEMINI_MODEL})"])
    provider = "ollama" if choice.startswith("Local") else "gemini"
    gemini_key = st.text_input("Gemini API key", type="password",
                               value=os.getenv("GEMINI_API_KEY", ""),
                               help="Needed for Gemini answers and for indexing images.")
    top_k = st.slider("Chunks retrieved (top-k)", 1, 10, 5)
    if provider == "gemini":
        st.caption("⚠️ Retrieved chunks are sent to Google. Use Local mode for private files.")

    st.divider()
    st.subheader("Knowledge base")
    sources = store.sources()
    if not sources:
        st.caption("Empty. Add something in the 'Add knowledge' tab.")
    for src, n in sources.items():
        c1, c2 = st.columns([5, 1])
        c1.caption(f"{src} · {n} chunks")
        if c2.button("🗑", key=f"del-{src}", help="Remove"):
            store.delete_source(src)
            st.rerun()

chat_tab, add_tab = st.tabs(["💬 Chat", "➕ Add knowledge"])

# ---------------- add knowledge ----------------
with add_tab:
    st.subheader("Files")
    files = st.file_uploader("PDF, Word, text/markdown, code, CSV/Excel, images",
                             accept_multiple_files=True,
                             type=[e.lstrip(".") for e in loaders.SUPPORTED])
    if st.button("Index files") and files:
        for f in files:
            try:
                with st.spinner(f"Indexing {f.name}..."):
                    docs = loaders.load_file(f.name, f.getvalue(), gemini_key)
                    report(rag.ingest(store, docs))
            except Exception as e:
                st.error(f"{f.name}: {e}")

    st.subheader("Web page")
    url = st.text_input("URL (GeeksforGeeks, W3Schools, any article...)")
    if st.button("Fetch page") and url:
        try:
            with st.spinner("Fetching..."):
                report(rag.ingest(store, loaders.load_url(url)))
        except Exception as e:
            st.error(str(e))

    st.subheader("The GATE Grind question crawler")
    st.caption("Crawls only GATE Grind paper/subject pages, checks robots.txt, waits between requests, "
               "and keeps extracted text in data/gategrind_cache.sqlite3.")
    gate_url = st.text_input("Catalog or paper/subject URL", value=gategrind.CATALOG_URL,
                             key="gategrind-url")
    c1, c2 = st.columns(2)
    max_pages = c1.number_input("Maximum question pages", min_value=1, max_value=gategrind.MAX_PAGES,
                                value=10, key="gategrind-limit")
    delay = c2.number_input("Seconds between requests", min_value=0.5, max_value=10.0,
                            value=1.0, step=0.5, key="gategrind-delay")
    if st.button("Crawl and index GATE Grind"):
        status = st.empty()
        try:
            result = gategrind.crawl(gate_url, max_pages=int(max_pages), delay_seconds=float(delay),
                                     progress=status.info)
            if result.documents:
                report(rag.ingest(store, result.documents))
            status.success(f"Fetched {result.pages_seen} pages; indexed {result.pages_changed}; "
                           f"unchanged cached pages {result.cached_pages}.")
            for error in result.errors:
                st.warning(error)
        except Exception as e:
            status.empty()
            st.error(str(e))
    if st.button("Index saved GATE Grind pages"):
        try:
            docs = gategrind.cached_documents()
            if not docs:
                st.info("No saved GATE Grind pages yet. Run the crawler first.")
            else:
                report(rag.ingest(store, docs))
        except Exception as e:
            st.error(str(e))

    st.subheader("Wikipedia")
    wq = st.text_input("Topic")
    if st.button("Fetch Wikipedia") and wq:
        try:
            with st.spinner("Fetching..."):
                report(rag.ingest(store, loaders.load_wikipedia(wq)))
        except Exception as e:
            st.error(str(e))

    st.subheader("arXiv")
    aq = st.text_input("Search terms or arXiv ID (e.g. 1706.03762)")
    n_papers = st.number_input("Max papers", 1, 5, 2)
    if st.button("Fetch papers") and aq:
        try:
            with st.spinner("Downloading and indexing papers..."):
                report(rag.ingest(store, loaders.load_arxiv(aq, int(n_papers))))
        except Exception as e:
            st.error(str(e))

# ---------------- chat ----------------
with chat_tab:
    if "history" not in st.session_state:
        st.session_state.history = []  # [{"role", "content", "hits"?}]

    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("hits"):
                show_sources(m["hits"])

    if q := st.chat_input("Ask your knowledge base..."):
        with st.chat_message("user"):
            st.markdown(q)
        past = [{"role": m["role"], "content": m["content"]} for m in st.session_state.history[-6:]]
        with st.chat_message("assistant"):
            hits = []
            try:
                hits, stream = rag.ask(store, q, past, provider, gemini_key, top_k)
                reply = st.write_stream(stream)
            except Exception as e:
                reply = f"⚠️ {e}"
                st.error(reply)
            if hits:
                show_sources(hits)
        st.session_state.history += [{"role": "user", "content": q},
                                     {"role": "assistant", "content": reply, "hits": hits}]
