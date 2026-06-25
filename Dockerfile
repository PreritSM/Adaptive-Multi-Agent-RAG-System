# ── Stage 1: deps ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: test ─────────────────────────────────────────────────────────────
# Fail the build if any test fails — mandatory quality gate before runtime image
FROM deps AS test

COPY . .

RUN pytest tests/ \
        --tb=short \
        --cov=src/api/schemas \
        --cov-fail-under=85 \
        -q

# ── Stage 3: runtime ──────────────────────────────────────────────────────────
# Copy source *from the test stage* so the runtime image can only be assembled
# after the test stage exits successfully. BuildKit must materialise the test
# stage to satisfy this COPY, making test failures propagate here.
FROM deps AS runtime

WORKDIR /app

COPY --from=test /app/src ./src/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
