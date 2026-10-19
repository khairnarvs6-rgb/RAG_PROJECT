
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.config import DOCUMENTS_DIR
from app.ingestion import ingest_documents
from app.rag_pipeline import answer_question

app = FastAPI(
    title="End-to-End RAG API",
    description="Question answering over PDF and TXT files",
    version="1.0.0",
)

HOME_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Document Q&amp;A</title>
  <style>
    :root { color-scheme: light; font: 16px/1.5 system-ui, sans-serif; color: #182230; background: #f3f6fb; }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 36px 18px; }
    main { max-width: 760px; margin: 0 auto; }
    h1 { margin-bottom: 4px; }
    .intro { color: #526173; margin-top: 0; }
    section { background: white; border: 1px solid #dce3ed; border-radius: 12px; padding: 22px; margin: 18px 0; }
    h2 { font-size: 1.15rem; margin: 0 0 12px; }
    label { display: block; font-weight: 600; margin: 12px 0 6px; }
    input, textarea, button { font: inherit; }
    input[type=file], textarea { width: 100%; padding: 10px; border: 1px solid #b9c5d4; border-radius: 7px; }
    textarea { min-height: 92px; resize: vertical; }
    button { margin-top: 12px; padding: 9px 15px; border: 0; border-radius: 7px; color: white; background: #175cd3; cursor: pointer; }
    button:disabled { opacity: .6; cursor: wait; }
    .secondary { background: #344054; }
    .status { min-height: 1.5em; white-space: pre-wrap; overflow-wrap: anywhere; }
    .answer { white-space: pre-wrap; background: #f8fafc; border-radius: 7px; padding: 12px; }
    a { color: #175cd3; }
  </style>
</head>
<body>
  <main>
    <h1>Ask your documents</h1>
    <p class="intro">Upload PDF or TXT files, index them locally, then ask questions answered with Groq.</p>
    <section>
      <h2>1. Add documents</h2>
      <label for="files">Choose PDF or TXT files</label>
      <input id="files" type="file" accept=".pdf,.txt" multiple>
      <button id="upload" type="button">Upload selected files</button>
      <p id="upload-status" class="status" role="status"></p>
      <button id="ingest" class="secondary" type="button">Index documents</button>
      <p id="ingest-status" class="status" role="status"></p>
    </section>
    <section>
      <h2>2. Ask a question</h2>
      <form id="ask-form">
        <label for="question">Question about your documents</label>
        <textarea id="question" required maxlength="5000" placeholder="For example: What is this book about?"></textarea>
        <button id="ask" type="submit">Ask</button>
      </form>
      <p id="ask-status" class="status" role="status"></p>
      <div id="answer" class="answer" hidden></div>
    </section>
    <p class="intro">API reference: <a href="/docs">/docs</a></p>
  </main>
  <script>
    async function requestJson(url, options) {
      const response = await fetch(url, options);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Request failed (" + response.status + ").");
      }
      return data;
    }

    const uploadButton = document.getElementById("upload");
    uploadButton.addEventListener("click", async () => {
      const files = document.getElementById("files").files;
      const status = document.getElementById("upload-status");
      if (!files.length) {
        status.textContent = "Choose at least one PDF or TXT file first.";
        return;
      }
      uploadButton.disabled = true;
      const uploaded = [];
      try {
        for (const file of files) {
          const body = new FormData();
          body.append("file", file);
          const result = await requestJson("/upload", { method: "POST", body });
          uploaded.push(result.filename);
          status.textContent = "Uploaded: " + uploaded.join(", ");
        }
        status.textContent = "Uploaded " + uploaded.length + " file(s): " + uploaded.join(", ");
      } catch (error) {
        status.textContent = "Upload failed: " + error.message;
      } finally {
        uploadButton.disabled = false;
      }
    });

    const ingestButton = document.getElementById("ingest");
    ingestButton.addEventListener("click", async () => {
      const status = document.getElementById("ingest-status");
      ingestButton.disabled = true;
      status.textContent = "Indexing documents…";
      try {
        const result = await requestJson("/ingest", { method: "POST" });
        status.textContent = "Indexed " + result.documents_loaded +
          " document page(s) into " + result.chunks_created + " chunks.";
      } catch (error) {
        status.textContent = "Indexing failed: " + error.message;
      } finally {
        ingestButton.disabled = false;
      }
    });

    const askForm = document.getElementById("ask-form");
    askForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const button = document.getElementById("ask");
      const status = document.getElementById("ask-status");
      const answer = document.getElementById("answer");
      button.disabled = true;
      status.textContent = "Searching documents and asking Groq…";
      answer.hidden = true;
      try {
        const result = await requestJson("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: document.getElementById("question").value })
        });
        answer.textContent = typeof result === "string" ? result : JSON.stringify(result, null, 2);
        answer.hidden = false;
        status.textContent = "";
      } catch (error) {
        status.textContent = "Question failed: " + error.message +
          " Check the API key and server terminal if the problem continues.";
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=5000,
    )


@app.get("/")
def home() -> HTMLResponse:
    return HTMLResponse(HOME_PAGE)


@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    stored_filename = f"{uuid4().hex}{suffix}"
    destination = DOCUMENTS_DIR / stored_filename
    bytes_written = 0

    try:
        with destination.open("wb") as output:
            while chunk := file.file.read(1024 * 1024):
                output.write(chunk)
                bytes_written += len(chunk)
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Could not save the uploaded file.",
        ) from exc

    if bytes_written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    return {
        "filename": filename,
        "stored_filename": stored_filename,
        "bytes_written": bytes_written,
    }


@app.post("/ingest")
def ingest():
    try:
        return ingest_documents()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post("/ask")
def ask(request: QuestionRequest):
    try:
        return answer_question(request.question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Question answering failed. Check server logs.",
        ) from exc
