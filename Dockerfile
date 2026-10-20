# syntax=docker/dockerfile:1.7
# CPU-only serving image for an exported Heartcut student (no PyTorch inside).
#
#   heartcut export -c configs/experiments/banking77.yaml      # prints the export dir
#   docker build --build-arg MODEL_DIR=<export dir> -t heartcut-serve .
#   docker run --rm -p 8000:8000 heartcut-serve
#
# Build args:
#   MODEL_DIR     exported model directory to bake in (default: ./model)
#   EXTRAS        comma-separated extras; add "teachers" to enable the fallback router
#   PYTHON_IMAGE  base image (any Debian-based CPython 3.11+)
#   UV_VERSION    uv release that installs the dependencies
ARG PYTHON_IMAGE=python:3.11-slim
# 0.13.4 or later: older uv releases trust the compromised eMudhra/emSign CA keys.
ARG UV_VERSION=0.13.4

# uv comes from its official image rather than from PyPI through the base image's pip,
# whose vendored certifi may still trust the compromised roots.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ${PYTHON_IMAGE} AS builder
ARG EXTRAS=serve
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
COPY --from=uv /uv /uvx /bin/
WORKDIR /src
# Dependencies first, so code changes do not invalidate the dependency layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN set -eu; args=""; for e in $(echo "$EXTRAS" | tr ',' ' '); do args="$args --extra $e"; done; \
    uv sync --frozen --no-dev --no-install-project $args
COPY src ./src
# The wheel build packages the price table (see force-include in pyproject.toml).
COPY configs/pricing.yaml ./configs/
RUN set -eu; args=""; for e in $(echo "$EXTRAS" | tr ',' ' '); do args="$args --extra $e"; done; \
    uv sync --frozen --no-dev --no-editable $args

FROM ${PYTHON_IMAGE}
RUN useradd --create-home --uid 1000 app
COPY --from=builder /opt/venv /opt/venv
COPY prompts /app/prompts
COPY configs /app/configs
ARG MODEL_DIR=model
COPY ${MODEL_DIR} /models/student
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HEARTCUT_MODEL_DIR=/models/student
WORKDIR /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "heartcut.serve.app:app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
