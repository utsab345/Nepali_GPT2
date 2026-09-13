# Inference API (roadmap Week 3)

A small FastAPI service wrapping the trained checkpoint.

Planned endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/generate` | text completion (prompt, max_tokens, temperature, top_p, stop tokens) |
| `POST` | `/next_token` | top-k next tokens + probabilities |
| `GET`  | `/health` | liveness / readiness |
| `GET`  | `/metrics` | request counters, avg latency, errors |

Design notes: input validation, timeouts, simple request batching, and
model path via env vars. Served with uvicorn/gunicorn; companion
`docker/Dockerfile` packages it.

## Status

- [ ] `api/main.py` (FastAPI app)
- [ ] `/generate` and `/next_token`
- [ ] health + metrics
- [ ] Docker packaging and demo