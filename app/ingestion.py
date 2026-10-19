
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from app.config import (
    CHROMA_DIR,
    DOCUMENTS_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
)


class LocalEmbeddings(Embeddings):
    """Use Chroma's local all-MiniLM-L6-v2 embedding model."""

    def __init__(self) -> None:
        self._embedding_function = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(value) for value in embedding]
            for embedding in self._embedding_function(texts)
        ]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def load_documents():
    """Load supported documents from the documents directory."""
    documents = []

    for path in DOCUMENTS_DIR.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(path)).load()

        elif path.suffix.lower() == ".txt":
            loaded = TextLoader(
                str(path),
                encoding="utf-8",
            ).load()

        else:
            continue

        for document in loaded:
            document.metadata["source"] = path.name
            document.metadata["file_path"] = str(path)

        documents.extend(loaded)

    return documents


def split_documents(documents):
    """Split documents while retaining source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    return splitter.split_documents(documents)


def get_vector_store():
    """Open the persistent Chroma collection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=LocalEmbeddings(),
    )


def ingest_documents():
    """Load files and index their chunks."""
    documents = load_documents()

    if not documents:
        raise ValueError(
            f"No PDF or TXT files found in {DOCUMENTS_DIR}"
        )

    chunks = split_documents(documents)
    vector_store = get_vector_store()

    # Add IDs so chunks can be traced to their source.
    ids = [
        f"{doc.metadata.get('source', 'document')}-"
        f"{doc.metadata.get('page', 'na')}-"
        f"{doc.metadata.get('start_index', i)}"
        for i, doc in enumerate(chunks)
    ]

    # Re-ingestion of the same files may create duplicates.
    # This example is intended for an initial indexing run.
    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    return {
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
    }


if __name__ == "__main__":
    result = ingest_documents()
    print("Ingestion complete:", result)
