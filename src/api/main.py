from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Awaitable

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    EvalRequest,
    EvalResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

# ---------------------------------------------------------------------------
# Logging — structured JSON to stdout
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
_logger: logging.Logger = logging.getLogger("api")


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # TODO: initialise Qdrant client, embedding model, SQLite pool
    yield
    # TODO: graceful teardown


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MultiAgent RAG API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware — structured request logging
# ---------------------------------------------------------------------------

@app.middleware("http")
async def structured_logging_middleware(  # type: ignore[misc]
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    request_id: str = str(uuid.uuid4())
    start: float = time.perf_counter()

    response: Response = await call_next(request)

    latency_ms: float = (time.perf_counter() - start) * 1000
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"

    _logger.info(
        json.dumps(
            {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params) or None,
                "status_code": response.status_code,
                "latency_ms": round(latency_ms, 2),
                "client_ip": request.client.host if request.client else None,
            }
        )
    )

    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    # TODO: ping Qdrant and Chroma and reflect real reachability
    return HealthResponse(
        status="ok",
        version="0.1.0",
        qdrant_reachable=False,
        chroma_reachable=False,
    )


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(body: QueryRequest) -> QueryResponse:
    # TODO: invoke the LangGraph compiled graph
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented yet")


@app.post("/ingest", response_model=IngestResponse, tags=["rag"])
async def ingest(body: IngestRequest) -> IngestResponse:
    # TODO: embed documents and upsert to Qdrant / Chroma
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented yet")


@app.post("/eval", response_model=EvalResponse, tags=["eval"])
async def evaluate(body: EvalRequest) -> EvalResponse:
    # TODO: run RAGAS evaluation pipeline
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented yet")
