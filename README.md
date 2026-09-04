# MultiAgent RAG

A production-oriented **multi-agent Retrieval-Augmented Generation (RAG) API** built on FastAPI, LangGraph, and Qdrant — with explicit LCEL chains, strict Python typing, and Pydantic contracts throughout. No high-level LangChain "black box" chains anywhere in the codebase.

---

## What it does

Given a natural-language query, the system:

1. **Routes** the query adaptively to dense (Qdrant ANN), sparse (BM25), or hybrid (weighted Reciprocal Rank Fusion) retrieval based on lightweight lexical heuristics.
2. **Grades** retrieved documents for relevance using an LLM-as-judge.
3. **Assesses uncertainty** over the retrieved context (entropy of retrieval scores + mean cosine similarity) and, if confidence is too low, **expands the query** (HyDE / step-back prompting) and re-retrieves — up to a configurable retry cap.
4. **Generates** an answer via an explicit LCEL pipe (`prompt | llm | StrOutputParser()`).
5. **Checks the answer for hallucination** against the retrieved context before returning it, retrying generation if needed (also capped).
6. Returns a fully-typed response with cited sources, per-source confidence scores, uncertainty score, token usage, latency, and an agent execution trace.

Evaluation (RAGAS: faithfulness, answer relevancy, context precision/recall, across prompt-variant ablations) runs via `/eval` or the `python -m src.eval.evaluation` CLI — see [Status](#status).

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    FastAPI (api/)                    │
│         /health   /query   /ingest   /eval           │
└────────────────────┬──────────────────────────────────┘
                     │ invokes compiled LangGraph
┌────────────────────▼────────────────────────────────┐
│              LangGraph Agent (agents/)               │
│                                                       │
│  retrieve → grade_documents → assess_uncertainty     │
│       ↑              ↓ (low confidence)              │
│  expand_query ←──────┘                               │
│       ↓ (sufficient confidence)                      │
│    generate → hallucination_check → END              │
│         ↑___________________|                        │
│           (retry, capped)                             │
└────────────────────┬──────────────────────────────────┘
                     │
        ┌────────────▼─────────────┐
        │        retrieval/         │
        │  DenseRetriever (Qdrant)  │
        │  SparseRetriever (BM25)   │
        │  HybridRetriever (RRF)    │
        │  QueryClassifier          │
        │  chunk_documents()        │
        └────────────────────────────┘
```

**External services (Docker Compose):**
- **Qdrant** (`localhost:6333`) — vector store for dense retrieval (in active use)
- **ChromaDB** (`localhost:8001`) — wired in `docker-compose.yml`, not yet used in code (open question — see [Status](#status))
- **SQLite** (`perf_logs.db`) — performance log persistence via `aiosqlite` + SQLAlchemy

---

## Repository layout

```
src/
  api/
    main.py          FastAPI app: lifespan/startup, structured JSON logging,
                      CORS, /health, /query, /ingest, /eval
    config.py         pydantic-settings Settings (env / .env driven)
    schemas.py         All Pydantic request/response contracts
  agents/
    state.py           AgentState TypedDict — shared LangGraph state
    graph.py            LangGraph topology: build_graph(deps)/compile_graph(deps),
                          six nodes as closures over GraphDependencies
    dependencies.py      GraphDependencies (frozen dataclass) + RetrieverFactory Protocol
    tools.py             Tool factories: retrieve_documents, web_search (Tavily),
                          expand_query, grade_document
    prompt_loader.py      load_prompts(), get_prompt() — {var}-style templates
    prompts.yaml           Prompt templates: default + ablation variants
                            (chain_of_thought, concise)
  retrieval/
    core.py             DenseRetriever, SparseRetriever, HybridRetriever (RRF),
                          QueryClassifier, chunk_documents()
    uncertainty.py        cosine_similarity, pairwise_cosine_matrix,
                          score_uncertainty, is_uncertain
  eval/
    evaluation.py         RAGAS dataset construction + evaluation, ablation CLI
tests/
  test_schemas.py         Full test suite for all Pydantic schemas
  test_eval.py             Eval-pipeline plumbing: dataset loading, RAGAS
                             dataset shaping, graph-driven sample generation
```

### Module boundaries (enforced — see `CLAUDE.md`)

| Package | Responsibility |
|---|---|
| `schemas/` (`api/schemas.py`) | Pydantic models only — no logic |
| `retrievers/` (`retrieval/`) | Document retrieval only — no prompt construction |
| `prompts/` | Prompt templates only — no retrieval or generation logic |
| `nodes/` (`agents/graph.py`) | LangGraph node functions — thin wrappers delegating to retrievers/generators |
| `graphs/` (`agents/graph.py`) | LangGraph graph construction — wiring only |
| `api/` | FastAPI route handlers — input validation and response shaping only |
| `services/` | Stateless business logic — no framework coupling |

A module must not import a sibling at the same layer (e.g. one retriever never imports another directly — composition happens at the graph layer).

---

## Engineering constraints (non-negotiable)

Full detail in [`CLAUDE.md`](./CLAUDE.md). Summary:

- **Strict typing** — every function fully annotated, `mypy --strict` must pass with zero errors, `from __future__ import annotations` everywhere, no bare `Any` outside external boundaries.
- **No blind chains** — `RetrievalQA`, `ConversationalRetrievalChain`, `load_qa_chain`, and similar LangChain "out-of-the-box" wrappers are banned. All retrieval/generation goes through explicit LCEL pipes (`prompt | llm | parser`); all routing/multi-step logic goes through LangGraph with named, unit-testable routing functions (no inline lambdas for routing).
- **Pydantic everywhere** — every API response and every inter-node payload is a `pydantic.BaseModel`; no raw `dict`/`TypedDict`/dataclass crosses a function boundary. FastAPI routes declare explicit `response_model=`.
- **Modular architecture** — one responsibility per module, sibling modules at the same layer never import each other, prompts live in external template files (never inline f-strings), heavy models/vector stores are initialised once at startup and injected (not loaded in hot paths), full documents are only fetched at the final generation step (IDs suffice elsewhere).

---

## Tech stack

| Concern | Choice |
|---|---|
| API framework | FastAPI + Uvicorn |
| Orchestration | LangGraph (explicit graph), LangChain-core (LCEL) |
| LLM | `langchain-openai` (`ChatOpenAI`, default `gpt-4o-mini`) |
| Vector store | Qdrant (`qdrant-client`, `langchain-qdrant`) |
| Sparse retrieval | `rank-bm25` |
| Embeddings | `sentence-transformers` (default `all-MiniLM-L6-v2`) |
| Evaluation | RAGAS + HuggingFace `datasets` |
| Config | `pydantic-settings` |
| Perf logging | SQLite via `aiosqlite` + SQLAlchemy |
| Observability | OpenTelemetry (API/SDK present) |
| Testing | pytest, pytest-asyncio, pytest-cov |
| Typing/lint | mypy (`--strict`), ruff |

See [`requirements.txt`](./requirements.txt) for pinned versions.

---

## Getting started

### Prerequisites
- Python 3.12
- Docker + Docker Compose (for Qdrant / ChromaDB)
- An OpenAI API key (required for real LLM calls; the app boots without one but `/query` will fail at the LLM step)

### Local setup

```bash
git clone <this-repo>
cd MultiAgent_RAG

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# create a .env with at least:
cat > .env <<EOF
OPENAI_API_KEY=sk-...
QDRANT_URL=http://localhost:6333
DATABASE_URL=sqlite+aiosqlite:///./perf_logs.db
EOF

docker compose up -d qdrant chromadb   # start vector stores
uvicorn src.api.main:app --reload      # serves on http://localhost:8000
```

### Configuration

All settings are environment-driven via `src/api/config.py::Settings` (loaded from `.env`). Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | `""` | Required for real LLM calls |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model used by all graph nodes |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant connection |
| `QDRANT_COLLECTION` | `default` | Default collection name |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `DATABASE_URL` | `sqlite+aiosqlite:///./perf_logs.db` | Perf log persistence |
| `TAVILY_API_KEY` | `""` | Web search tool fallback |
| `PROMPTS_PATH` | `src/agents/prompts.yaml` | Prompt template file |
| `EVAL_DATASETS_DIR` | `data/eval_datasets` | Directory of `{name}.jsonl` golden Q&A datasets for `/eval` |
| `DEFAULT_PROMPT_VARIANT` | `default` | Prompt variant (`default`, `chain_of_thought`, `concise`) |
| `DEFAULT_UNCERTAINTY_THRESHOLD` | `0.75` | Confidence gate for triggering query expansion |
| `MAX_EXPAND_ATTEMPTS` | `2` | Retry cap for query expansion loop |
| `MAX_GENERATION_ATTEMPTS` | `2` | Retry cap for hallucination-check loop |

### Running with Docker Compose (full stack)

```bash
docker compose up -d
```

This starts the `api`, `qdrant`, and `chromadb` containers on a shared bridge network. The API container reads `OPENAI_API_KEY` from the host environment.

---

## API reference

| Method | Path | Description | Status |
|---|---|---|---|
| `GET` | `/health` | Liveness + Qdrant/Chroma reachability | ✅ Implemented |
| `POST` | `/query` | Run a query through the full agent graph | ✅ Implemented |
| `POST` | `/ingest` | Chunk, embed, and upsert documents into Qdrant | ✅ Implemented |
| `POST` | `/eval` | Run RAGAS evaluation over a named dataset | ✅ Implemented |

### `POST /query`

Request (`QueryRequest`):
```json
{
  "query": "What is Reciprocal Rank Fusion?",
  "k": 5,
  "retrieval_mode": "hybrid",
  "uncertainty_threshold": 0.75,
  "collection": "default"
}
```

Response (`QueryResponse`): answer text, cited `sources` (with per-chunk score/metadata), `citation_map`, `confidence_scores`, `uncertainty_score`, `agent_trace` (list of node names visited), `latency_ms`, token counts.

### `POST /ingest`

Request (`IngestRequest`): `documents` (raw text list), `collection`, optional per-document `metadata`. Documents are chunked via `chunk_documents()` (metadata-aware, `RecursiveCharacterTextSplitter`), embedded, and upserted to Qdrant. Duplicate `chunk_id`s (a 16-char SHA-256 prefix) are skipped.

### `POST /eval`

Request (`EvalRequest`): `dataset_name` (a `data/eval_datasets/{name}.jsonl` golden Q&A file), `prompt_variant`, `k`. Runs the real compiled agent graph over each golden question, then scores the results with the RAGAS suite (faithfulness, answer relevancy, context precision/recall) and returns an `EvalResponse`. The same pipeline is also exposed as `python -m src.eval.evaluation --dataset <name>`, which runs every `prompts.yaml` variant and ranks the results by faithfulness for ablation studies.

---

## Testing & quality gates

Run before every commit (see `CLAUDE.md`'s tooling checklist):

```bash
mypy --strict .
ruff check .
pytest
grep -r "RetrievalQA\|load_qa_chain\|ConversationalRetrieval" . --include="*.py"   # must be empty
```

Current test coverage: full Pydantic schema validation suite (`tests/test_schemas.py`) and the eval-pipeline plumbing (`tests/test_eval.py` — dataset loading, RAGAS dataset shaping, graph-driven sample generation via a fake LLM/retriever). Retrieval-layer unit tests, graph integration tests, and live RAGAS-metric assertions are planned but not yet written (see [Status](#status)).

---

## Status

This project is under active development. As of the last update:

**Done:**
- Full Pydantic schema layer, FastAPI skeleton with structured logging and lifespan-managed dependency injection
- `retrieval/core.py` — dense/sparse/hybrid retrievers, adaptive `QueryClassifier`, metadata-aware chunking
- `retrieval/uncertainty.py` — entropy + cosine-similarity uncertainty scoring
- Full LangGraph agent: all six nodes implemented (retrieve, grade, assess uncertainty, expand query, generate, hallucination check), with dependency-injected LLM/retriever wiring and retry caps
- `/health`, `/query`, `/ingest` working end-to-end (verified with fake LLM + fake retriever)
- RAGAS evaluation pipeline (`eval/evaluation.py`) and `/eval` endpoint — golden-dataset loading, real graph-driven sample generation, and a prompt-variant ablation CLI (`python -m src.eval.evaluation`)

**Remaining:**
- Retrieval-layer and graph integration test coverage (eval-pipeline plumbing has unit tests in `tests/test_eval.py`; deep RAGAS-metric assertions against a live LLM are still open)
- Decide the ChromaDB question — it's wired in `docker-compose.yml` and `HealthResponse.chroma_reachable` but unused in code; either integrate it or remove it

See `HANDOFF.md` for the original detailed execution plan and design rationale.

---

## Key design notes

- **Compiled graph is not built at import time.** `agents/graph.py::build_graph`/`compile_graph` take a `GraphDependencies` instance and are invoked from `api/main.py`'s `lifespan()`; access the compiled graph via `request.app.state.compiled_graph` and runtime collaborators via `request.app.state.runtime`.
- **Prompt templates use `{var}`-style substitution** (`agents/prompts.yaml`). Any literal example JSON in a new prompt must escape braces as `{{`/`}}`.
- **Infinite-loop protection**: `expand_attempts` / `generation_attempts` on `AgentState`, capped by `Settings.max_expand_attempts` / `max_generation_attempts` (default 2 each).
- **Qdrant payload shape** is fixed: `{"content": str, "metadata": dict}` — `DenseRetriever` reads exactly this.
- **`chunk_id`** is a 16-char SHA-256 prefix used both as the RRF deduplication key across dense/sparse retrieval and as the seed for the Qdrant point UUID (`uuid.uuid5(uuid.NAMESPACE_OID, chunk_id)`), since raw hex strings aren't valid Qdrant point IDs.

---

## License

Not yet specified.
