"""
Local SimCSE-style semantic similarity API server.

Replaces the remote SimCSE endpoint expected by sem_relevence.py.
Uses sentence-transformers (all-MiniLM-L6-v2) to compute cosine similarity
between two texts.  Same request/response shape as the original research cluster:

  POST /similarity
  Body: {"text1": "...", "text2": "..."}
  Response: {"similarity": float}

Run with:
  venv/bin/uvicorn local_simcse_api:app --host 0.0.0.0 --port 8001
"""

import os
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

app = FastAPI(title="Local SimCSE Similarity API")

print("Loading sentence-transformer model (all-MiniLM-L6-v2) on CPU…")
_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
print("Model loaded.")


class SimilarityRequest(BaseModel):
    text1: str
    text2: str


@app.post("/similarity")
def compute_similarity(req: SimilarityRequest):
    embeddings = _model.encode([req.text1, req.text2])
    sim = float(sklearn_cosine([embeddings[0]], [embeddings[1]])[0][0])
    # Clamp to [0, 1] — cosine can be slightly negative for very dissimilar text
    sim = max(0.0, min(1.0, sim))
    return {"similarity": sim}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SIMCSE_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
