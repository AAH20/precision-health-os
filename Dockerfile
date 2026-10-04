# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip && pip install -e .

FROM base AS dev
RUN pip install -e ".[dev]"
COPY tests ./tests
COPY examples ./examples
COPY docs ./docs

FROM base AS prod
COPY . .
RUN pip install --no-deps -e .
RUN useradd -m -r phos && chown -R phos:phos /app
USER phos
EXPOSE 8000
CMD ["phos", "serve", "--host", "0.0.0.0", "--port", "8000"]
