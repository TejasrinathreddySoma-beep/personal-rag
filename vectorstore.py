"""LangChain FAISS vector store, persisted to disk."""
import shutil
from collections import Counter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import DATA_DIR, EMBED_MODEL

INDEX_DIR = DATA_DIR / "faiss_index"


class VectorStore:
    def __init__(self):
        self.emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL,
                                         encode_kwargs={"normalize_embeddings": True})
        self.db = None
        if (INDEX_DIR / "index.faiss").exists():
            # safe here: we only load an index this app wrote itself
            self.db = FAISS.load_local(str(INDEX_DIR), self.emb, allow_dangerous_deserialization=True)

    def _save(self):
        if self.db is None:
            shutil.rmtree(INDEX_DIR, ignore_errors=True)
        else:
            self.db.save_local(str(INDEX_DIR))

    def add(self, source, docs):
        """Re-adding a source replaces its old chunks."""
        self.delete_source(source, save=False)
        if self.db is None:
            self.db = FAISS.from_documents(docs, self.emb)
        else:
            self.db.add_documents(docs)
        self._save()

    def delete_source(self, source, save=True):
        if self.db is None:
            return
        ids = [i for i, d in self.db.docstore._dict.items() if d.metadata.get("source") == source]
        if not ids:
            return
        self.db.delete(ids)
        if not self.db.docstore._dict:
            self.db = None
        if save:
            self._save()

    def search(self, query, k=5):
        """Returns [{"text", "source", "page"?, "score"}]; score is L2 distance (lower = closer)."""
        if self.db is None:
            return []
        k = min(k, len(self.db.docstore._dict))
        return [{"text": d.page_content, **d.metadata, "score": float(s)}
                for d, s in self.db.similarity_search_with_score(query, k=k)]

    def sources(self):
        if self.db is None:
            return Counter()
        return Counter(d.metadata.get("source", "?") for d in self.db.docstore._dict.values())
