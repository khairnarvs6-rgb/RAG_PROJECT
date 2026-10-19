from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import GROQ_API_KEY, GROQ_CHAT_MODEL, TOP_K
from app.ingestion import get_vector_store


def answer_question(question: str) -> str:
    """Answer a question using retrieved documents and Groq."""
    if not question.strip():
        raise ValueError("Question must not be empty.")

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add your Groq API key to the project .env file."
        )

    documents = get_vector_store().similarity_search(
        question,
        k=TOP_K,
    )
    if not documents:
        raise ValueError(
            "No relevant documents found. Ingest documents before asking a question."
        )

    context = "\n\n".join(
        f"Source: {document.metadata.get('source', 'unknown')}\n"
        f"{document.page_content}"
        for document in documents
    )

    chat = ChatGroq(
        model=GROQ_CHAT_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )
    response = chat.invoke(
        [
            SystemMessage(
                content=(
                    "Answer using only the supplied document context. "
                    "If the context does not contain the answer, say so."
                )
            ),
            HumanMessage(
                content=f"Document context:\n{context}\n\nQuestion: {question}"
            ),
        ]
    )

    if not isinstance(response.content, str):
        raise TypeError("Groq returned a non-text response.")

    return response.content