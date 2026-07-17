"""
main.py
-------
FastAPI application entrypoint for the Employee Handbook RAG Assistant.

Run directly:
    python main.py

Or via uvicorn (equivalent, supports --reload):
    uvicorn main:app --host 0.0.0.0 --port 5002 --reload

Endpoints:
    POST /chat    -> question -> RAG search -> retrieved context -> Qwen -> answer.
                      Refuses (without calling the model) if nothing relevant
                      is found in the handbook.
    GET  /health  -> liveness check for the API, the vector index, and llama.cpp.

    Auth (see auth.py):
        POST   /auth/signup      -> create an account (role: user | hr-employee | admin)
        POST   /auth/login       -> log in, returns a JWT plus the account's role/status/tabs
        GET    /auth/me          -> the caller's current role/status/panel-tab access
        GET    /auth/users       -> list all users                         (admin only)
        POST   /auth/users       -> create a user with a role/status/tabs  (admin only)
        PUT    /auth/users/{id}  -> update a user's role/status/tabs       (admin only)
        DELETE /auth/users/{id}  -> delete a user                          (admin only)

    Unanswered queries (see queries.py):
        POST   /queries          -> report a chatbot answer for human follow-up (any logged-in user)
        GET    /queries          -> list reported queries                       (needs "queries" tab)
        PATCH  /queries/{id}     -> submit/update an answer; trains that
                                     department's RAG index the moment it's
                                     first marked "answered"                     (needs "queries" tab)
        DELETE /queries/{id}     -> delete a reported query                     (needs "queries" tab)

    Departments / File Management (see departments.py):
        GET    /departments                        -> list departments + status (any logged-in user)
        GET    /departments/{slug}/documents        -> list a department's PDFs        (admin only)
        POST   /departments/upload                  -> upload a PDF; creates the
                                                         department (and its chatbot
                                                         tab) if it's new, and
                                                         (re)indexes it immediately     (admin only)
        DELETE /departments/{slug}/documents/{name} -> remove one PDF, re-index        (admin only)
        DELETE /departments/{slug}                  -> remove a department entirely    (admin only)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    LlamaServerError,
    LlamaServerResponseError,
    LlamaStartupError,
    frontend_server,
    generate_answer,
    llama_process_manager,
    logger,
    settings,
)
from rag import RAGInitializationError, RAGNotReadyError, rag_manager
from auth import init_db, router as auth_router
from queries import init_db as init_queries_db, router as queries_router
from departments import router as departments_router

# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Brings up the whole stack when the API boots — the RAG vector index,
    llama-server, and the frontend static file server — and tears them
    back down on exit.
    """
    logger.info("Starting up: initializing auth database...")
    try:
        init_db()
    except Exception as exc:  # noqa: BLE001 - never let this block API startup
        logger.error("Failed to initialize auth database: %s", exc)
        logger.error("The API will still start, but /auth routes may fail.")

    logger.info("Starting up: initializing unanswered-queries database...")
    try:
        init_queries_db()
    except Exception as exc:  # noqa: BLE001 - never let this block API startup
        logger.error("Failed to initialize queries database: %s", exc)
        logger.error("The API will still start, but /queries routes may fail.")

    logger.info("Starting up: initializing RAG systems (one per department)...")
    try:
        rag_manager.initialize_all()
    except Exception as exc:  # noqa: BLE001 - never let this block API startup
        logger.error("Failed to initialize RAG systems: %s", exc)
        logger.error(
            "The API will still start, but /chat will refuse all questions "
            "for departments whose index failed to load."
        )

    logger.info("Starting up: bringing up llama-server...")
    try:
        llama_process_manager.start()
    except LlamaStartupError as exc:
        logger.error("Failed to start llama-server: %s", exc)
        logger.error(
            "The API will still start, but /chat will fail until llama-server is reachable."
        )

    logger.info("Starting up: bringing up frontend server...")
    try:
        frontend_server.start()
    except Exception as exc:  # noqa: BLE001 - never let this block API startup
        logger.error("Failed to start frontend server: %s", exc)

    yield

    logger.info("Shutting down: stopping frontend server...")
    frontend_server.stop()

    logger.info("Shutting down: stopping llama-server (if we started it)...")
    llama_process_manager.stop()


app = FastAPI(
    title="Employee Handbook RAG Assistant",
    description=(
        "FastAPI backend that answers questions strictly from the employee "
        "handbook, using RAG (FAISS + sentence-transformers) and a local "
        "llama.cpp server."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS: allow the frontend (and any client) to call this API.
# Tighten allow_origins in production to your actual frontend origin(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth routes: POST /auth/signup, POST /auth/login
app.include_router(auth_router)

# Unanswered-queries routes: POST/GET/PATCH/DELETE /queries
app.include_router(queries_router)

# Department file-management routes: GET/POST/DELETE /departments
# (admin-only PDF upload -> creates the department's chatbot tab and
# (re)builds its RAG index immediately)
app.include_router(departments_router)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    """Health check for the API, each department's RAG vector index, and llama-server."""
    llama_status = "ok" if llama_process_manager.is_healthy() else "unreachable"
    per_department = rag_manager.status_summary()
    vector_index_summary = ", ".join(f"{dept}:{status_}" for dept, status_ in per_department.items())
    return HealthResponse(status="ok", vector_index=vector_index_summary, llama_server=llama_status)


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    responses={
        503: {"description": "The vector index or the local model server is unavailable."},
        502: {"description": "The local model server returned an unexpected response."},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Answer a question strictly from the employee handbook:

        question -> RAG search -> retrieved context -> Qwen -> answer

    If RAG search finds nothing relevant, the model is never called and
    a fixed refusal message is returned instead.
    """
    logger.info(
        "Received /chat request: %r (language=%s, department=%s)",
        request.message, request.language, request.department,
    )

    try:
        answer = generate_answer(request.message, language=request.language, department=request.department)
    except RAGNotReadyError as exc:
        logger.error("RAGNotReadyError while handling /chat: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"answer": settings.REFUSAL_MESSAGE, "error": str(exc)},
        )
    except LlamaServerError as exc:
        logger.error("LlamaServerError while handling /chat: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"answer": settings.FALLBACK_ANSWER, "error": str(exc)},
        )
    except LlamaServerResponseError as exc:
        logger.error("LlamaServerResponseError while handling /chat: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"answer": settings.FALLBACK_ANSWER, "error": str(exc)},
        )
    except Exception as exc:  # noqa: BLE001 - final safety net
        logger.exception("Unhandled exception while handling /chat")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"answer": settings.FALLBACK_ANSWER, "error": "Internal server error."},
        )

    return ChatResponse(answer=answer)


# --------------------------------------------------------------------------
# Entrypoint — lets this file be run directly (e.g. `python main.py`,
# or a code runner / debugger) instead of only via the `uvicorn` CLI.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    host = os.environ.get("APP_HOST", "0.0.0.0")
    port = int(os.environ.get("APP_PORT", "5002"))
    reload = os.environ.get("APP_RELOAD", "false").lower() == "true"

    logger.info("Starting FastAPI app via __main__ on %s:%s", host, port)
    uvicorn.run("main:app", host=host, port=port, reload=reload)