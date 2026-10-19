from app.ingestion import LocalEmbeddings
from app import rag_pipeline
from app import main
from fastapi.testclient import TestClient


client = TestClient(main.app)


def test_local_embeddings_generate_document_and_query_vectors(monkeypatch):
    class FakeEmbeddingFunction:
        def __call__(self, texts):
            return [[float(len(text))] for text in texts]

    embeddings = LocalEmbeddings()
    monkeypatch.setattr(
        embeddings,
        "_embedding_function",
        FakeEmbeddingFunction(),
    )

    assert embeddings.embed_documents(["a", "abc"]) == [[1.0], [3.0]]
    assert embeddings.embed_query("query") == [5.0]


def test_answer_question_uses_groq_with_retrieved_context(monkeypatch):
    class FakeDocument:
        page_content = "The answer is 42."
        metadata = {"source": "answer.txt"}

    class FakeVectorStore:
        def similarity_search(self, question, k):
            assert question == "What is the answer?"
            assert k == 4
            return [FakeDocument()]

    class FakeResponse:
        content = "The answer is 42."

    class FakeChatGroq:
        def __init__(self, **kwargs):
            assert kwargs["model"] == rag_pipeline.GROQ_CHAT_MODEL
            assert kwargs["api_key"] == "test-key"
            assert kwargs["temperature"] == 0

        def invoke(self, messages):
            assert "The answer is 42." in messages[1].content
            return FakeResponse()

    monkeypatch.setattr(rag_pipeline, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(rag_pipeline, "get_vector_store", lambda: FakeVectorStore())
    monkeypatch.setattr(rag_pipeline, "ChatGroq", FakeChatGroq)

    assert rag_pipeline.answer_question("What is the answer?") == "The answer is 42."


def test_answer_question_requires_groq_api_key(monkeypatch):
    monkeypatch.setattr(rag_pipeline, "GROQ_API_KEY", None)

    try:
        rag_pipeline.answer_question("What is the answer?")
    except RuntimeError as error:
        assert "GROQ_API_KEY" in str(error)
    else:
        raise AssertionError("Expected a missing-key error.")


def test_home_page_has_upload_index_and_question_controls():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="upload"' in response.text
    assert 'id="ingest"' in response.text
    assert 'id="ask-form"' in response.text


def test_upload_document_saves_supported_file(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)

    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"RAG test content", "text/plain")},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["filename"] == "notes.txt"
    assert (tmp_path / result["stored_filename"]).read_bytes() == b"RAG test content"


def test_upload_document_rejects_unsupported_type():
    response = client.post(
        "/upload",
        files={"file": ("notes.csv", b"not supported", "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF and TXT files are supported."