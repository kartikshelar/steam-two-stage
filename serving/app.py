"""Single endpoint: retrieval + ranking. Not production traffic."""

from __future__ import annotations

import argparse
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from serving.service import RecsService, load_service

app = FastAPI(title="steam-recsys", version="0.5.0")
_svc: RecsService | None = None


class RecommendRequest(BaseModel):
    user_id: str
    k: int = Field(default=20, ge=1, le=500)


class RecommendResponse(BaseModel):
    user_id: str
    items: list[dict[str, Any]]
    retrieve_k: int
    k: int


@app.on_event("startup")
def startup() -> None:
    global _svc
    _svc = load_service()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    if _svc is None:
        raise HTTPException(status_code=503, detail="not loaded")
    return {"status": "ready"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    if _svc is None:
        raise HTTPException(status_code=503, detail="not loaded")
    return RecommendResponse(**_svc.recommend(req.user_id, req.k))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("serving.app:app", host=args.host, port=args.port, factory=False)


if __name__ == "__main__":
    main()
