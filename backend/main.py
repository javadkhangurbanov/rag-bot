import json
import os
import traceback
from typing import AsyncGenerator, List, Optional

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .rag_store import format_context, ingest_folder, retrieve

load_dotenv()

print(">> DEBUG ENV",
      "AWS_REGION=", os.getenv("AWS_REGION"),
      "MODEL_ID=", os.getenv("BEDROCK_MODEL_ID"))

print(">>> BACKEND STARTED (main.py loaded) <<<", flush=True)

app = FastAPI(title="RAG Chat Backend")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    config=Config(retries={"max_attempts": 3, "mode": "standard"}),
)

class HistoryItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    system_prompt: str | None = "You are a helpful assistant."
    temperature: float | None = 0.2
    max_tokens: int | None = 512
    use_rag: bool | None = True
    top_k: int | None = 4
    history: Optional[List[HistoryItem]] = []

@app.post("/ingest")
def ingest():
    """
    Optional convenience endpoint to re-index the 'knowledge/' folder.
    """
    folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "knowledge"))
    count = ingest_folder(folder)
    return {"ingested_chunks": count}

def build_messages(req: ChatRequest):
    recent = (req.history or [])[-6:]
    msgs = _to_anthropic_messages(recent)

    if req.use_rag:
        snippets = retrieve(req.message, k=req.top_k or 3)
        context = format_context(snippets)
        user_text = (
            "Here is some optional CONTEXT you could use when answering. "
            "ONLY If the context is relevant to the question, prioritize those details "
            "and cite the source filenames in parentheses. "
            "If the context is not relevant, you should answer from your general knowledge.\n\n"
            "=== CONTEXT ===\n"
            f"{context}\n"
            "=== END CONTEXT ===\n\n"
            f"Question: {req.message}"
        )
    else:
        user_text = req.message

    # append the current user message, after history
    msgs.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
    return msgs

def _to_anthropic_messages(history: List[HistoryItem]) -> list[dict]:
    """
    Convert [{"role":"user"/"assistant","content":"..."}] to
    Anthropic's [{"role": "...", "content":[{"type":"text","text":"..."}]}]
    """
    out = []
    for h in history:
        role = "user" if h.role == "user" else "assistant"
        out.append({"role": role, "content": [{"type": "text", "text": h.content}]})
    return out

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    """
    Non-streaming Claude chat.
    """
    try:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "system": req.system_prompt,
            "messages": build_messages(req),
        }
        resp = bedrock.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        out = json.loads(resp["body"].read())
        # claude puts text output under content[0]["text"]
        text = out["content"][0]["text"]
        return {"text": text}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def sse_format(data: str) -> bytes:
    return f"data: {data}\n\n".encode("utf-8")

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    Streaming Claude chat with Server-Sent Events.
    """
    try:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "system": req.system_prompt,
            "messages": build_messages(req),
        }

        response = bedrock.invoke_model_with_response_stream(
            modelId=MODEL_ID,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )

        def event_generator() -> AsyncGenerator[bytes, None]:
            stream = response.get("body")
            if stream is None:
                yield sse_format(json.dumps({"error": "no stream"}))
                return
            for event in stream:
                if "chunk" in event:
                    chunk = event["chunk"]["bytes"]
                    try:
                        payload = json.loads(chunk.decode("utf-8"))
                        if payload.get("type") == "content_block_delta":
                            delta = payload["delta"].get("text", "")
                            if delta:
                                yield sse_format(json.dumps({"token": delta}))
                    except Exception:
                        pass
            yield sse_format(json.dumps({"event": "done"}))

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/debug/retrieve")
def debug_retrieve(q: str = Query(...), k: int = 4):
    try:
        hits = retrieve(q, k=k)
        return {
            "query": q,
            "k": k,
            "results": [
                {
                    "id": h["id"],
                    "source": h["metadata"].get("source"),
                    "distance": h["distance"],
                    "preview": (h["text"][:300] + ("..." if len(h["text"]) > 300 else ""))
                }
                for h in hits
            ]
        }
    except Exception as e:
        traceback.print_exc()
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/augment")
def debug_augment(q: str, k: int = 3):
    class TmpReq:
        message = q
        system_prompt = "You are a helpful assistant."
        temperature = 0.2
        max_tokens = 256
        use_rag = True
        top_k = k
    msgs = build_messages(TmpReq)
    # return only the user content
    return {"messages": msgs}
