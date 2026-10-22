"""LangChain document loaders. Every function returns a list of Documents whose
metadata["source"] is the name shown in the knowledge base."""
import io
import re
import tempfile
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    ArxivLoader, CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader,
    WebBaseLoader, WikipediaLoader,
)

from config import CODE_EXT, IMAGE_EXT, TEXT_EXT

SUPPORTED = sorted({".pdf", ".docx", ".csv", ".xlsx", ".xls"} | TEXT_EXT | CODE_EXT | set(IMAGE_EXT))


def _tmp_path(name, data):
    p = Path(tempfile.mkdtemp()) / Path(name).name
    p.write_bytes(data)
    return str(p)


def _tag(docs, source):
    for d in docs:
        d.metadata["source"] = source
    return docs


def _excel_docs(name, data):
    docs = []
    for sheet, df in pd.read_excel(io.BytesIO(data), sheet_name=None).items():
        df = df.fillna("").astype(str)
        rows = [", ".join(f"{c}: {v}" for c, v in r.items()) for _, r in df.iterrows()]
        for i in range(0, len(rows), 15):
            docs.append(Document(page_content=f"{name}/{sheet}\n" + "\n".join(rows[i:i + 15]),
                                 metadata={"sheet": sheet, "rows": f"{i + 1}-{min(i + 15, len(rows))}"}))
    return docs


def load_file(name, data, gemini_key=None):
    ext = Path(name).suffix.lower()
    if ext in (".xlsx", ".xls"):
        return _tag(_excel_docs(name, data), name)
    if ext in IMAGE_EXT:  # multimodal: Gemini describes the image, we index the description
        import llm
        text = llm.describe_image(data, IMAGE_EXT[ext], gemini_key)
        return [Document(page_content=text, metadata={"source": name, "type": "image"})]

    path = _tmp_path(name, data)
    if ext == ".pdf":
        docs = PyPDFLoader(path).load()
        for d in docs:
            d.metadata["page"] = d.metadata.get("page", 0) + 1  # 1-based for humans
    elif ext == ".docx":
        docs = Docx2txtLoader(path).load()
    elif ext == ".csv":
        docs = CSVLoader(path).load()
    elif ext in TEXT_EXT | CODE_EXT:
        docs = TextLoader(path, encoding="utf-8", autodetect_encoding=True).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return _tag(docs, name)


def load_url(url):
    loader = WebBaseLoader([url], header_template={"User-Agent": "Mozilla/5.0 (personal-rag)"},
                           requests_kwargs={"timeout": 20})
    docs = loader.load()
    for d in docs:
        d.page_content = re.sub(r"\n\s*\n+", "\n\n", d.page_content).strip()
        d.metadata["source"] = d.metadata.get("title") or url
        d.metadata["url"] = url
    return docs


def load_wikipedia(query):
    docs = WikipediaLoader(query=query, load_max_docs=1, doc_content_chars_max=100_000).load()
    if not docs:
        raise ValueError(f"No Wikipedia article found for '{query}'")
    for d in docs:
        d.metadata["url"] = d.metadata.get("source", "")
        d.metadata["source"] = f"Wikipedia: {d.metadata.get('title', query)}"
    return docs


def load_arxiv(query, max_results=2):
    """query = search terms or an arXiv id such as 1706.03762."""
    docs = ArxivLoader(query=query.strip(), load_max_docs=max_results,
                       doc_content_chars_max=150_000).load()
    if not docs:
        raise ValueError("No arXiv results")
    for d in docs:
        d.metadata["source"] = f"arXiv: {d.metadata.get('Title', query)}"
    return docs
