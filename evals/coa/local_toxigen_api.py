"""
Local toxicity API server — wraps the OpenAI Moderation API to match
the interface expected by toxic_detector.py's ToxiGen class.

Original ToxiGen protocol:

  POST /toxicity
  Body: {"texts": ["text1", "text2", ...]}
  Response: {"result": [{"type": str, "toxicity": float}, ...]}

This wrapper calls client.moderations.create() (openai v2.x API) and
maps the per-category scores to the same shape, so no changes to the
existing toxic_detector.py are needed.

Run with:
  venv/bin/uvicorn local_toxigen_api:app --host 0.0.0.0 --port 8002
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
import config

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Local Toxicity API (OpenAI Moderation backend)")

client = OpenAI(api_key=config.OPENAI_API_KEY)


class ToxicityRequest(BaseModel):
    texts: List[str]


@app.post("/toxicity")
def get_toxicity(req: ToxicityRequest):
    if not req.texts:
        return {"result": []}

    try:
        response = client.moderations.create(
            input=req.texts,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI Moderation API error: {e}")

    result = []
    for item in response.results:
        scores = item.category_scores
        # Pick the single highest-scoring category (mirrors OpenAIModerationAPI logic)
        scores_dict = {k: v for k, v in vars(scores).items() if not k.startswith('_')}
        top_category, top_score = max(scores_dict.items(), key=lambda kv: kv[1])
        result.append({
            "type": top_category,
            "toxicity": float(top_score),
        })

    return {"result": result}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("TOXIGEN_PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)
