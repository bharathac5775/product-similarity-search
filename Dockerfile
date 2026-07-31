# syntax=docker/dockerfile:1
#
# Multi-stage build for the Product Similarity storefront + JSON API.
#
#   Stage 1 (builder)  installs Python dependencies into an isolated venv using
#                      build tooling we do NOT want in the final image.
#   Stage 2 (runtime)  starts from a clean slim base and copies only the venv and
#                      application assets, producing a smaller, hardened image
#                      with good layer caching and fast Kubernetes pod startup.

# ---- Stage 1: builder ----
FROM python:3.10-slim AS builder

# Build without cache and without writing .pyc into the layer.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Isolated virtual environment for our dependencies.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies first so this layer is cached unless requirements change.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# ---- Stage 2: runtime ----
FROM python:3.10-slim AS runtime

# OCI image metadata (professional provenance).
LABEL org.opencontainers.image.title="product-similarity" \
      org.opencontainers.image.description="Content-based product similarity search with a storefront UI and JSON API" \
      org.opencontainers.image.source="https://github.com/bharathac5775/product-similarity-search" \
      org.opencontainers.image.licenses="MIT"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Create the non-root user up front so subsequent COPY --chown lands correctly.
RUN useradd --create-home --uid 10001 appuser

# Bring over just the prebuilt virtualenv (no compilers / build tools shipped).
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Application code + server-rendered storefront assets.
# (Ordered least- to most-frequently-changed for better layer caching.)
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser templates/ ./templates/
COPY --chown=appuser:appuser static/ ./static/
COPY --chown=appuser:appuser app.py ./

# Dataset baked into the image so the container runs standalone (last: largest,
# rarely changes relative to code — kept in its own layer).
COPY --chown=appuser:appuser data/ ./data/

USER appuser

EXPOSE 8000

# Container-level healthcheck (Docker / Compose). Kubernetes uses its own probes.
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else sys.exit(1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
