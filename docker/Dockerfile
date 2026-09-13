# NepaliGPT inference service (roadmap Week 3)
#
# Build:   docker build -f docker/Dockerfile -t nepaligpt-api .
# Run:     docker run --rm -p 8000:8000 \
#            -v "$PWD/ckpt:/app/ckpt" -v "$PWD/tokenizer:/app/tokenizer" \
#            nepaligpt-api
#
# For a GPU runtime switch the base to a PyTorch CUDA image and set
# NEPALIGPT_DEVICE=cuda.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NEPALIGPT_CKPT=ckpt/best.pt \
    NEPALIGPT_TOK=tokenizer/nepali_bpe.model \
    NEPALIGPT_DEVICE=cpu

WORKDIR /app

# Install the package and the serving stack.
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
COPY api ./api
RUN python -m pip install --upgrade pip \
    && python -m pip install -e . \
    && python -m pip install "fastapi>=0.110" "uvicorn[standard]>=0.29"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]