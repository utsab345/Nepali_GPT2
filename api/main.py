"""FastAPI inference service for NepaliGPT (roadmap Week 3).

Endpoints:
    POST /generate    — autoregressive text completion
    POST /next_token  — top-k next-token probabilities
    GET  /health      — liveness / readiness + model info
    GET  /metrics     — request / latency / error counters

The checkpoint and tokenizer are configured through environment variables
so the same image serves any trained model without a rebuild:

    NEPALIGPT_CKPT   (default: ckpt/best.pt)
    NEPALIGPT_TOK    (default: tokenizer/nepali_bpe.model)
    NEPALIGPT_DEVICE (default: cuda if available, else cpu)

Run locally:

    NEPALIGPT_DEVICE=cpu uvicorn api.main:app --reload
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_CKPT = os.environ.get("NEPALIGPT_CKPT", "ckpt/best.pt")
MODEL_TOK = os.environ.get("NEPALIGPT_TOK", "tokenizer/nepali_bpe.model")
MODEL_DEVICE = os.environ.get("NEPALIGPT_DEVICE", "")  # "" -> auto (cuda if available)


class _Metrics:
    """Thread-safe request / latency / error counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.generated_tokens = 0
        self.latency_total_s = 0.0
        self.started_at = time.monotonic()

    def record(self, latency_s: float, tokens: int, error: bool) -> None:
        with self._lock:
            self.requests += 1
            self.generated_tokens += tokens
            self.latency_total_s += latency_s
            if error:
                self.errors += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            uptime = time.monotonic() - self.started_at
            n = max(self.requests, 1)
            return {
                "requests": self.requests,
                "errors": self.errors,
                "error_rate": self.errors / n,
                "generated_tokens": self.generated_tokens,
                "avg_latency_ms": (self.latency_total_s / n) * 1000,
                "uptime_s": uptime,
            }


metrics = _Metrics()
_state: dict[str, Any] = {}
_lock = threading.Lock()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1024)
    max_new: int = Field(80, ge=1, le=1024)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=0, le=500)
    top_p: float = Field(0.92, gt=0.0, le=1.0)
    stop: list[str] = Field(default_factory=list)


class NextTokenRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1024)
    top_n: int = Field(10, ge=1, le=100)


def get_model() -> tuple[Any, Any, dict[str, Any], Any]:
    """Lazily load the checkpoint + tokenizer on first use (thread-safe)."""
    with _lock:
        if "model" not in _state:
            from nepali_gpt2.generate import load_model_and_tokenizer

            device = MODEL_DEVICE or None
            model, sp, cfg, device = load_model_and_tokenizer(
                MODEL_CKPT, MODEL_TOK, device
            )
            _state.update(
                model=model,
                sp=sp,
                cfg=cfg,
                device=device,
                loaded_at=datetime.now(timezone.utc).isoformat(),
            )
        return _state["model"], _state["sp"], _state["cfg"], _state["device"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        get_model()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to initialise the model on startup") from exc
    yield


app = FastAPI(title="NepaliGPT API", version="1.0.0", lifespan=lifespan)


@app.post("/generate")
def generate_endpoint(req: GenerateRequest) -> dict[str, str]:
    t0 = time.monotonic()
    try:
        model, sp, cfg, device = get_model()
        from nepali_gpt2.generate import generate

        text = generate(
            model,
            sp,
            cfg,
            device,
            prompt=req.prompt,
            max_new=req.max_new,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
        for stop in req.stop:
            if stop:
                idx = text.find(stop)
                if idx != -1:
                    text = text[:idx]
                    break
        metrics.record(time.monotonic() - t0, len(sp.encode(text, out_type=int)), False)
        return {"prompt": req.prompt, "text": text}
    except Exception as exc:  # noqa: BLE001
        metrics.record(time.monotonic() - t0, 0, True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/next_token")
def next_token_endpoint(req: NextTokenRequest) -> dict[str, Any]:
    t0 = time.monotonic()
    try:
        model, sp, cfg, device = get_model()
        from nepali_gpt2.generate import next_words

        preds = next_words(model, sp, cfg, device, req.prompt, top_n=req.top_n)
        metrics.record(time.monotonic() - t0, len(req.prompt), False)
        return {
            "prompt": req.prompt,
            "predictions": [{"token": tok, "probability": prob} for tok, prob in preds],
        }
    except Exception as exc:  # noqa: BLE001
        metrics.record(time.monotonic() - t0, 0, True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": "model" in _state,
        "ckpt": MODEL_CKPT,
        "loaded_at": _state.get("loaded_at"),
    }


@app.get("/metrics")
def metrics_endpoint() -> dict[str, Any]:
    return metrics.snapshot()
