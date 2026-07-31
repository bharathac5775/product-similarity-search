# syntax=docker/dockerfile:1
#
# Multi-stage build:
#   Stage 1 (builder)  installs all Python dependencies into an isolated venv,
#                      using build tooling that we do NOT want in the final image.
#   Stage 2 (runtime)  starts from a clean slim base and copies only the venv and
#                      the application code, producing a smaller, more secure image
#                      with better layer caching and faster Kubernetes pod startup.

# ---- Stage 1: builder ----
FROM python:3.10-slim AS builder

WORKDIR /app

# Create an isolated virtual environment for our dependencies.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies first (this layer is cached unless requirements change).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: runtime ----
FROM python:3.10-slim AS runtime

WORKDIR /app

# Bring over just the prebuilt virtualenv (no compilers / build tools shipped).
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Application code and dataset.
COPY src/ ./src/
COPY app.py ./
COPY data/ ./data/

# Run as a non-root user for security.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
