# Docker (roadmap Week 3)

Serve the NepaliGPT API in a container.

## Build & run

```bash
# build the image
docker build -f docker/Dockerfile -t nepaligpt-api .

# run with your trained checkpoint + tokenizer mounted read-only
docker run --rm -p 8000:8000 \
  -v "$PWD/ckpt:/app/ckpt" \
  -v "$PWD/tokenizer:/app/tokenizer" \
  nepaligpt-api

# or with compose
docker compose -f docker/docker-compose.yml up --build
```

Then hit `http://localhost:8000/health` and `/docs`.

## Notes

- Built for CPU by default (`NEPALIGPT_DEVICE=cpu`); for GPU release use a
  PyTorch CUDA base image (see comment in `Dockerfile`).
- `.dockerignore` keeps data, model weights and outputs out of the image;
  weights/tokenizer are mounted at runtime.
- Build context is the repo root (one level above `docker/`).