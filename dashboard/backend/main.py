"""
main.py — FastAPI wrapper around your real HMM + Bayesian pipeline.

Run:
    cd backend
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Then open frontend/index.html in a browser (it calls http://localhost:8000).

Training the HMM happens once, at startup, and takes anywhere from a
few seconds to ~1 minute depending on your machine — watch the terminal,
it prints progress. After that, every request is instant (it's just a
forward pass through the already-fitted model).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_service import service

app = FastAPI(title="Abuse-Ring Sentinel API")

# Wide-open CORS since this is a local hackathon demo, not production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    print("Training HMM + fitting Bayesian evidence — this runs once...")
    try:
        service.train()
        print("Model ready.")
        print(service.stats)
    except Exception as exc:  # noqa: BLE001
        service.error = str(exc)
        print(f"MODEL TRAINING FAILED: {exc}")


class Window(BaseModel):
    transaction_count_window: float = 1
    amount_total_window: float = 50.0
    amount_mean_window: float = 50.0
    amount_max_window: float = 50.0
    amount_std_window: float = 0.0
    time_since_previous_transaction: float = 999999.0
    unique_devices_window: float = 1
    device_entity_count: float = 1
    unique_payment_emails_window: float = 1
    unique_receiver_emails_window: float = 1
    unique_addresses_window: float = 1
    unique_products_window: float = 1
    night_activity_rate: float = 0.0
    amount_change_ratio: float = 0.0


class ScoreRequest(BaseModel):
    windows: list[Window] = Field(..., min_length=1)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ready": service.ready,
        "error": service.error,
        "stats": service.stats,
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    if not service.ready:
        raise HTTPException(503, "Model still training — check /health")
    return service.stats


@app.post("/score")
def score(request: ScoreRequest) -> dict[str, Any]:
    if not service.ready:
        raise HTTPException(503, "Model still training — check /health")
    windows = [w.model_dump() for w in request.windows]
    try:
        results = service.score_sequence(windows)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc
    return {"steps": results}
