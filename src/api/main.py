from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Awaitable, cast

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.agents.dependencies import GraphDependencies, RetrieverFactory
from src.agents.graph import compile_graph
from src.agents.prompt_loader import load_prompts
from src.agents.state import AgentState
from src.api.config import Settings, get_settings
from src.api.schemas import (
    DocumentChunk,
    EvalRequest,
    EvalResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from src.retrieval.core import EmbeddingFn, Retriever, build_retriever, chunk_documents

# ---------------------------------------------------------------------------
# Logging — structured JSON to stdout
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
_logger: logging.Logger = logging.getLogger("api")


# ---------------------------------------------------------------------------
# Runtime state — collaborators built once at startup, injected via app.state
# ---------------------------------------------------------------------------

class _SentenceTransformerEmbedding:
    """Adapts a SentenceTransformer model to the EmbeddingFn protocol."""

    def __init__(self, model: SentenceTransformer) -> None:
        self._model = model

    def __call__(self, text: str) -> list[float]:
        vector: list[float] = self._model.encode(text).tolist()
        return vector


@dataclass
class RuntimeState:
    settings: Settings
    qdrant_client: AsyncQdrantClient
    embedding_fn: EmbeddingFn
    corpus: list[Document] = field(default_factory=list)


def _make_retriever_factory(state: RuntimeState) -> RetrieverFactory:
    def _factory(mode: str) -> Retriever:
        return build_retriever(
            mode,
            state.qdrant_client,
            state.settings.qdrant_collection,
            state.embedding_fn,
            state.corpus,
        )

    return _factory


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)
    embedding_fn: EmbeddingFn = _SentenceTransformerEmbedding(
        SentenceTransformer(settings.embedding_model_name)
    )
    runtime = RuntimeState(settings=settings, qdrant_client=qdrant_client, embedding_fn=embedding_fn)

    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=SecretStr(settings.openai_api_key),
        temperature=0.0,
    )
    prompts = load_prompts(settings.prompts_path)

    deps = GraphDependencies(
        llm=llm,
        retriever_factory=_make_retriever_factory(runtime),
        embedding_fn=embedding_fn,
        prompts=prompts,
        prompt_variant=settings.default_prompt_variant,
        max_expand_attempts=settings.max_expand_attempts,
        max_generation_attempts=settings.max_generation_attempts,
    )

    app.state.runtime = runtime
    app.state.compiled_graph = compile_graph(deps)

    yield

    await qdrant_client.close()


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
async def structured_logging_middleware(
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
async def health(request: Request) -> HealthResponse:
    runtime: RuntimeState = request.app.state.runtime
    qdrant_reachable = False
    try:
        await runtime.qdrant_client.get_collections()
        qdrant_reachable = True
    except Exception:  # noqa: BLE001 — reachability probe must never raise
        qdrant_reachable = False

    return HealthResponse(
        status="ok",
        version="0.1.0",
        qdrant_reachable=qdrant_reachable,
        chroma_reachable=False,
    )


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(body: QueryRequest, request: Request) -> QueryResponse:
    compiled_graph: CompiledStateGraph = request.app.state.compiled_graph
    query_id = str(uuid.uuid4())
    start = time.perf_counter()

    initial_state: AgentState = {
        "query": body.query,
        "query_id": query_id,
        "messages": [],
        "k": body.k,
        "retrieved_docs": [],
        "retrieval_scores": [],
        "retrieval_mode": body.retrieval_mode.value,
        "uncertainty_score": 0.0,
        "uncertainty_threshold": body.uncertainty_threshold,
        "requires_fallback": False,
        "expand_attempts": 0,
        "generation_attempts": 0,
        "answer": "",
        "agent_trace": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_ms": 0.0,
    }

    result = cast(AgentState, await compiled_graph.ainvoke(initial_state))
    latency_ms = (time.perf_counter() - start) * 1000

    sources = [
        DocumentChunk(
            id=str(doc.metadata.get("chunk_id", str(i))),
            content=doc.page_content,
            score=score,
            metadata=doc.metadata,
        )
        for i, (doc, score) in enumerate(
            zip(result["retrieved_docs"], result["retrieval_scores"])
        )
    ]

    return QueryResponse(
        query_id=query_id,
        answer=result["answer"],
        sources=sources,
        uncertainty_score=result["uncertainty_score"],
        agent_trace=result["agent_trace"],
        latency_ms=latency_ms,
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
    )


@app.post("/ingest", response_model=IngestResponse, tags=["rag"])
async def ingest(body: IngestRequest, request: Request) -> IngestResponse:
    runtime: RuntimeState = request.app.state.runtime
    chunks = chunk_documents(body.documents, body.metadata)

    existing_ids = {doc.metadata.get("chunk_id") for doc in runtime.corpus}
    new_chunks = [c for c in chunks if c.metadata.get("chunk_id") not in existing_ids]
    skipped = len(chunks) - len(new_chunks)

    if new_chunks:
        vector_size = len(runtime.embedding_fn(new_chunks[0].page_content))
        existing_collections = {
            c.name for c in (await runtime.qdrant_client.get_collections()).collections
        }
        if body.collection not in existing_collections:
            await runtime.qdrant_client.create_collection(
                collection_name=body.collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_OID, str(chunk.metadata["chunk_id"]))),
                vector=runtime.embedding_fn(chunk.page_content),
                payload={"content": chunk.page_content, "metadata": chunk.metadata},
            )
            for chunk in new_chunks
        ]
        await runtime.qdrant_client.upsert(collection_name=body.collection, points=points)
        runtime.corpus.extend(new_chunks)

    return IngestResponse(collection=body.collection, inserted=len(new_chunks), skipped=skipped)


@app.post("/eval", response_model=EvalResponse, tags=["eval"])
async def evaluate(body: EvalRequest) -> EvalResponse:
    # TODO: run RAGAS evaluation pipeline
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented yet")
