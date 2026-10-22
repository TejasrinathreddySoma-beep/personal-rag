import os
from pathlib import Path

DATA_DIR = Path(os.getenv("RAG_DATA_DIR", "data"))
DATA_DIR.mkdir(exist_ok=True)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # runs locally, free
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

TEXT_EXT = {".txt", ".md", ".rst"}
CODE_EXT = {".py", ".c", ".cpp", ".h", ".java", ".js", ".ts", ".sql", ".go",
            ".rs", ".html", ".css", ".json", ".yaml", ".yml", ".sh"}
IMAGE_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
