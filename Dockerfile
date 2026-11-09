# Quant research environment: pinned Python 3.11 + locked deps (uv.lock)
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /usr/local/bin/

WORKDIR /app

# Install deps first for layer caching
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY Makefile ./

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    OMP_NUM_THREADS=8

CMD ["bash"]
