# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

# Sample dataset so `procurement-agent demo` / `eval` work inside the container.
COPY eval/dataset ./eval/dataset

# Run as an unprivileged user; uploaded files live in a dedicated volume-friendly directory.
RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/data && chown appuser /app/data
USER appuser

ENV DATA_DIR=/app/data \
    PROVIDER_MODE=mock

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "procurement_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
