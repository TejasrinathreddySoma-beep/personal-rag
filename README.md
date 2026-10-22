# Personal RAG (LangChain + FAISS + Streamlit)

Ask questions over your own files and web sources. Answers come from a local Ollama model or Gemini.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Local LLM
ollama pull llama3.1:8b          # or set OLLAMA_MODEL to any model you already have

# Gemini (free key from aistudio.google.com)
export GEMINI_API_KEY=your_key   # Windows: set GEMINI_API_KEY=your_key

streamlit run app.py
```

## How it works

| Step | LangChain piece |
|---|---|
| Load | `PyPDFLoader`, `Docx2txtLoader`, `CSVLoader`, `TextLoader`, `WebBaseLoader`, `WikipediaLoader`, `ArxivLoader` (+ pandas for Excel, Gemini for images) |
| Split | `RecursiveCharacterTextSplitter` (1000 chars, 150 overlap) |
| Embed | `HuggingFaceEmbeddings` (all-MiniLM-L6-v2, runs locally) |
| Store | `FAISS`, saved to `data/faiss_index` |
| Answer | LCEL chain: `prompt | ChatOllama / ChatGoogleGenerativeAI | StrOutputParser` |

## Notes

- Images are described by Gemini and the description is indexed, so image indexing needs a Gemini key.
- In Gemini mode, retrieved chunks are sent to Google. Use Local mode for private files.
- Re-adding a file or URL with the same name replaces its old chunks.
- Scanned PDFs have no text layer and will fail to index (they need OCR).
- Change models via env vars: `OLLAMA_MODEL`, `GEMINI_MODEL`.
- The **GATE Grind question crawler** starts at `https://thegategrind.online/pyq` (or one
  paper/subject URL), follows only the site's `/exam/<id>` and `/subject/<id>` pages,
  honours `robots.txt` when available, rate-limits requests, and caps a run at 100 pages.
  It stores extracted page text and hashes in `data/gategrind_cache.sqlite3`; unchanged
  pages are not embedded again. **Index saved GATE Grind pages** can rebuild the vector
  index from this local cache without another web crawl. Use a small scope first (for
  example, one CS paper).
- Answers are strict-RAG: if retrieved source text does not explicitly support an answer,
  the assistant returns `Not found in the indexed source.` rather than inventing a question.
