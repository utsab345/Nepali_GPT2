# Inference API (roadmap Week 3)

A FastAPI service wrapping a trained NepaliGPT checkpoint.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/generate` | text completion: `prompt`, `max_new`, `temperature`, `top_k`, `top_p`, `stop` |
| `POST` | `/next_token` | top-k next tokens + probabilities (`prompt`, `top_n`) |
| `GET`  | `/health` | liveness / readiness + model info |
| `GET`  | `/metrics` | request counters, error rate, avg latency, uptime |

Model paths come from env vars (`NEPALIGPT_CKPT`, `NEPALIGPT_TOK`,
`NEPALIGPT_DEVICE`) so one image serves any checkpoint. Docs are at
`/docs` (OpenAPI).

## Run

```bash
NEPALIGPT_DEVICE=cpu uvicorn api.main:app --reload

curl -s http://localhost:8000/health
curl -s http://localhost:8000/generate -H 'Content-Type: application/json' \
  -d '{"prompt": "नेपाल एक सुन्दर", "max_new": 80}'
```

## Demo

`api/demo.py` is a Gradio UI that talks to this API; point it at any
deployed instance with `GRADIO_API_URL`.

## Status

- [x] `api/main.py` (FastAPI app)
- [x] `/generate` and `/next_token`
- [x] health + metrics
- [x] `api/demo.py` Gradio demo
- [ ] Docker packaging — see `docker/README.md`