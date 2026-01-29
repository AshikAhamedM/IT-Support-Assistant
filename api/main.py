from fastapi import FastAPI
import requests
from rag.retriever import retrieve_it_context
from schemas import ITSupportRequest

app = FastAPI()
VLLM_URL = "http://localhost:8001/v1/chat/completions"

@app.post("/it-support/query")
def resolve_ticket(req: ITSupportRequest):
    context = retrieve_it_context(req.question)

    prompt = f"""
You are an IT support assistant.
Use ONLY the context below.

Context:
{context}

User Issue:
{req.question}
"""

    payload = {
        "model": "it-support-lora-model",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    return requests.post(VLLM_URL, json=payload).json()
