"""
main.py
-------
FastAPI application entrypoint for the Employee Handbook RAG Assistant.

Run directly:
    python main.py

Or via uvicorn (equivalent, supports --reload):
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /chat    -> question -> RAG search -> retrieved context -> Qwen -> answer.
                      Refuses (without calling the model) if nothing relevant
                      is found in the handbook.
    GET  /health  -> liveness check for the API, the vector index, and llama.cpp.
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
from rag import RAGInitializationError, RAGNotReadyError, rag_system

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
    logger.info("Starting up: initializing RAG system...")
    try:
        rag_system.initialize()
    except RAGInitializationError as exc:
        logger.error("Failed to initialize RAG system: %s", exc)
        logger.error(
            "The API will still start, but /chat will refuse all questions "
            "until the vector index is loaded."
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


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    """Health check for the API, the RAG vector index, and llama-server."""
    llama_status = "ok" if llama_process_manager.is_healthy() else "unreachable"
    return HealthResponse(status="ok", vector_index=rag_system.status, llama_server=llama_status)


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
    logger.info("Received /chat request: %r", request.message)

    try:
        answer = generate_answer(request.message)
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
    port = int(os.environ.get("APP_PORT", "8000"))
    reload = os.environ.get("APP_RELOAD", "false").lower() == "true"

    logger.info("Starting FastAPI app via __main__ on %s:%s", host, port)
    uvicorn.run("main:app", host=host, port=port, reload=reload)