from collections import Counter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter

import llm
from config import CHUNK_OVERLAP, CHUNK_SIZE

SYSTEM = (
    "You are a personal knowledge assistant. Answer using the numbered context snippets from the "
    "user's own knowledge base and cite them like [1], [2]. Answer only from the supplied context. "
    "If the context does not explicitly contain the answer, reply exactly: 'Not found in the indexed "
    "source.' Do not create, infer, or add questions from general knowledge. "
    "Be concise and accurate; use code blocks for code."
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder("history"),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def ingest(store, docs):
    """Split -> embed -> FAISS. Returns {source: number_of_chunks}."""
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    if not chunks:
        raise ValueError("No extractable text found (is it a scanned PDF?)")
    by_source = {}
    for c in chunks:
        by_source.setdefault(c.metadata["source"], []).append(c)
    for source, group in by_source.items():
        store.add(source, group)
    return dict(Counter({s: len(g) for s, g in by_source.items()}))


def format_context(hits):
    if not hits:
        return "(The knowledge base is empty.)"
    return "\n\n".join(
        f"[{i}] (source: {h['source']}" + (f", page {h['page']}" if h.get("page") else "") + f")\n{h['text']}"
        for i, h in enumerate(hits, 1))


def ask(store, question, history, provider, gemini_key=None, k=5):
    """history: [{"role": "user"|"assistant", "content": str}]. Returns (hits, token_stream)."""
    hits = store.search(question, k)
    chain = PROMPT | llm.get_llm(provider, gemini_key) | StrOutputParser()
    turns = [("human" if m["role"] == "user" else "ai", m["content"]) for m in history]
    stream = chain.stream({"history": turns, "context": format_context(hits), "question": question})
    return hits, stream
