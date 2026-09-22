"""OpenAI-compatible proxy that makes a TRACE model a drop-in target for any attack framework.

    client  --(standard chat messages)-->  this proxy  --([Turn N] + system prompt)-->  vLLM/SGLang
    client  <--(ANSWER only)-------------  this proxy  <--(<STATE>...<ANSWER>...)-------  vLLM/SGLang

Point an attack framework's `base_url` at this server and it will see an ordinary chat model: it
sends the conversation so far, gets back a reply. The trajectory collapse, the system prompt, and
the STATE stripping all happen here, exactly as in the paper's evaluations. The STATE block is
returned alongside the reply in a non-standard `trace_state` field for logging, and never leaks
into the reply text.

Run:
    python -m trace_defense.server --backend http://localhost:30003/v1 --model Dipto084/Llama3.1-8B-TRACE --port 8000
"""

import argparse
import logging
import time
import uuid
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .formatting import build_messages, parse_output

log = logging.getLogger("trace_defense.server")
app = FastAPI(title="TRACE proxy")
CFG: Dict[str, Any] = {}


def _served_name() -> str:
    return CFG.get("served_model_name") or CFG["model"]


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": _served_name(), "object": "model", "owned_by": "trace"}]}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages: List[Dict[str, str]] = body.get("messages") or []
    if any(m.get("role") == "system" for m in messages):
        # TRACE needs its own system prompt; a client-supplied one cannot be honoured.
        log.warning("dropping client system message; TRACE uses its own system prompt")
    try:
        request_messages = build_messages(messages, CFG["variant"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    payload = {
        "model": CFG["model"],
        "messages": request_messages,
        "temperature": body.get("temperature", CFG["temperature"]),
        "max_tokens": body.get("max_tokens", CFG["max_tokens"]),
    }
    for k in ("top_p", "seed", "stop"):
        if k in body:
            payload[k] = body[k]

    async with httpx.AsyncClient(timeout=CFG["timeout"]) as client:
        r = await client.post(f"{CFG['backend']}/chat/completions", json=payload,
                              headers={"Authorization": f"Bearer {CFG['api_key']}"})
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"backend error {r.status_code}: {r.text[:500]}")
    upstream = r.json()
    raw = upstream["choices"][0]["message"]["content"] or ""
    state, state_text, answer = parse_output(raw)

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _served_name(),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": upstream["choices"][0].get("finish_reason", "stop"),
        }],
        "usage": upstream.get("usage", {}),
        # Non-standard: the model's safety assessment for this turn.
        "trace_state": state if state is not None else state_text,
    })


def main():
    p = argparse.ArgumentParser(description="OpenAI-compatible proxy for TRACE models")
    p.add_argument("--backend", default="http://localhost:30003/v1", help="OpenAI-compatible server hosting the TRACE weights")
    p.add_argument("--model", required=True, help="model name as served by the backend")
    p.add_argument("--served-model-name", default=None, help="name to expose to clients (default: --model)")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--variant", choices=["default", "over_refusal"], default="default")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    CFG.update(vars(args))
    CFG["backend"] = args.backend.rstrip("/")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
