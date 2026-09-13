# Docker packaging (roadmap Week 3)

Planned:

- `Dockerfile` — Python + PyTorch base (CPU and/or GPU variants), installs
  the package, copies configs, entrypoint runs uvicorn serving the FastAPI
  app in `api/`.
- `docker-compose.yml` — local dev stack (API + optional monitoring).

## Status

- [ ] CPU Dockerfile
- [ ] GPU Dockerfile
- [ ] docker-compose for local dev