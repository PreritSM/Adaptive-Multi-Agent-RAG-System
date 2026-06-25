# MultiAgent RAG — Engineering Guidelines

These rules are **non-negotiable** and apply to every file in this project. No exceptions without an explicit comment explaining why and sign-off in a PR.

---

## 1. Strict Python Typing

- Every function signature **must** carry full type annotations: arguments and return type.
- Every module-level variable and class attribute **must** be typed.
- Run `mypy --strict` as part of CI; the build fails on any type error.
- Use `from __future__ import annotations` at the top of every file.
- Prefer `TypeAlias`, `TypeVar`, `Generic`, and `Protocol` over untyped duck typing.
- No bare `Any` unless wrapping an external boundary (e.g., raw JSON from an untrusted API) — annotate it with a `# type: ignore[misc]` comment explaining the reason.

```python
# good
def retrieve(query: str, k: int = 5) -> list[Document]:
    ...

# bad — missing return type and untyped arg
def retrieve(query, k=5):
    ...
```

---

## 2. No Blind Chains — Explicit LCEL and LangGraph Only

**Banned patterns:**
- `RetrievalQA`, `ConversationalRetrievalChain`, `load_qa_chain`, and any other `langchain` high-level "out-of-the-box" chain wrappers.
- Any chain constructed by a helper that hides prompt construction, retrieval, or routing logic from the call site.

**Required patterns:**

### Retrieval / generation — use explicit LCEL pipes
```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### Routing and multi-step logic — use LangGraph with explicit node/edge definitions
```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("grade", grade_node)
graph.add_node("generate", generate_node)
graph.add_conditional_edges("grade", route_fn, {"relevant": "generate", "irrelevant": END})
graph.set_entry_point("retrieve")
```

Every routing decision must be a named, unit-testable function. Inline lambdas for routing are banned.

---

## 3. Pydantic for Every API Response Contract

- Every external API response **and** every inter-service/inter-node payload **must** be modelled with a `pydantic.BaseModel` subclass.
- No `dict`, `TypedDict`, or plain dataclass for anything that crosses a function boundary or is serialised/deserialised.
- Use `model_validator` and `field_validator` for any constraint that cannot be expressed in a type annotation alone.
- API routes (FastAPI) must declare explicit `response_model=` on every endpoint; never return a raw `dict`.
- Keep all Pydantic models in a dedicated `schemas/` package; never define them inline inside route handlers or node functions.

```python
# schemas/retrieval.py
from pydantic import BaseModel, Field

class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048)
    k: int = Field(default=5, ge=1, le=20)

class RetrievalResponse(BaseModel):
    documents: list[DocumentSchema]
    scores: list[float]
```

---

## 4. Modular, Token-Efficient Architecture

The goal is that each module does **one thing** and imports only what it needs — this keeps context windows small when individual modules are passed to an LLM and prevents runaway token usage in agentic loops.

### Module boundaries
| Package | Responsibility |
|---|---|
| `schemas/` | Pydantic models only — no logic |
| `retrievers/` | Document retrieval only — no prompt construction |
| `prompts/` | Prompt templates only — no retrieval or generation logic |
| `nodes/` | LangGraph node functions — thin wrappers that delegate to retrievers/generators |
| `graphs/` | LangGraph graph construction — wiring only, no business logic |
| `api/` | FastAPI route handlers — input validation and response shaping only |
| `services/` | Stateless business logic functions — no framework coupling |

### Rules
- A module **must not** import from a sibling module at the same layer (e.g., a retriever must not import another retriever directly; compose at the graph layer).
- Keep prompts in `.txt` or `.jinja2` files under `prompts/`; load them at startup, never inline f-strings as prompt templates.
- Avoid loading large embedding models or vector stores inside hot-path functions; initialise once at application startup and inject via dependency.
- Never pass full document payloads where a document ID suffices; retrieve the full document only at the final generation step.

---

## Tooling Checklist

Before every commit:
- [ ] `mypy --strict .` — zero errors
- [ ] `ruff check .` — zero lint errors
- [ ] `pytest` — all tests pass
- [ ] No new `langchain` high-level chain imports (`grep -r "RetrievalQA\|load_qa_chain\|ConversationalRetrieval" . --include="*.py"` must return empty)
