
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4
COLLECTION_NAME = "rag_documents_local"
 