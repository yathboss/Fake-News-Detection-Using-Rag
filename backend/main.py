from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.pipeline import ClaimVerificationPipeline
from backend.schemas import VerifyRequest, VerifyResponse


app = FastAPI(
    title="Fake News Detection Phase 1 API",
    version="1.0.0",
    description="Mini prototype: claim -> retrieval -> KG context -> local Llama verdict",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = ClaimVerificationPipeline()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/verify", response_model=VerifyResponse)
def verify(request: VerifyRequest):
    claim = request.claim.strip()
    if len(claim) < 3:
        raise HTTPException(status_code=400, detail="Claim is too short.")

    try:
        result = pipeline.verify_claim(claim=claim, top_k=request.top_k)
        return VerifyResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
