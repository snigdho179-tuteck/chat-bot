"""
backend.py
----------
Core logic for the Employee Handbook Assistant.

Flow for every question:
    question -> RAG search (rag.py) -> retrieved context -> Qwen (llama.cpp) -> answer

If RAG search finds nothing relevant, the model is never called — we
return a fixed refusal message instead, so the assistant only ever
answers from the supplied handbook context.
"""

from __future__ import annotations

import functools
import http.server
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from rag import RAGInitializationError, RAGNotReadyError, rag_system

# Load variables from a .env file into os.environ, before Settings reads
# them below. We point load_dotenv() at the directory this file lives in
# (rather than relying on the current working directory), so this works
# the same whether you run `python main.py` from the backend/ folder,
# from the project root, or via a tool like VS Code's Code Runner that
# may launch the process from a different working directory.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
_loaded = load_dotenv(dotenv_path=_ENV_PATH)

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

logger = logging.getLogger("ai_assistant.backend")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)

if _loaded:
    logger.info("Loaded environment variables from %s", _ENV_PATH)
else:
    logger.warning(
        "No .env file found at %s — using defaults / shell environment variables only.",
        _ENV_PATH,
    )

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

class Settings:
    """Static configuration for the backend."""

    # LLAMA_HOST is what llama-server BINDS to (0.0.0.0 = all interfaces —
    # correct for a server). It is NOT a valid address for a client to
    # connect to. LLAMA_CLIENT_HOST is what *we* connect to when health
    # checking / calling the server, and should stay 127.0.0.1 unless
    # llama-server is running on a different machine/container.
    LLAMA_HOST: str = os.environ.get("LLAMA_HOST", "0.0.0.0")
    LLAMA_CLIENT_HOST: str = os.environ.get("LLAMA_CLIENT_HOST", "127.0.0.1")
    LLAMA_PORT: int = int(os.environ.get("LLAMA_PORT", "8080"))

    LLAMA_SERVER_URL: str = f"http://{LLAMA_CLIENT_HOST}:{LLAMA_PORT}/v1/chat/completions"
    LLAMA_HEALTH_URL: str = f"http://{LLAMA_CLIENT_HOST}:{LLAMA_PORT}/health"

    SYSTEM_PROMPT: str = (
        "You are an Employee Handbook Assistant.\n\n"
        "Answer ONLY from the supplied handbook context.\n"
        "Never use your own knowledge.\n"
        "The user may ask indirectly, casually, or with non-HR wording; "
        "map their intent to the closest relevant handbook policy only when "
        "the provided context supports it.\n\n"
        "If the context does not contain enough information to answer,\n"
        "reply exactly:\n"
        '"I can only answer questions related to the Employee Handbook."'
    )

    QUERY_REWRITE_SYSTEM_PROMPT: str = (
        "You convert employee questions into clear Employee Handbook search queries.\n\n"
        "Rules:\n"
        "- Do not answer the question.\n"
        "- Do not invent company policy.\n"
        "- Extract the likely HR or handbook policy intent.\n"
        "- Use concise handbook-style keywords.\n"
        "- Include useful synonyms if the user asks indirectly.\n"
        "- Output only the rewritten search query."
    )

    REFUSAL_MESSAGE: str = "I can only answer questions related to the Employee Handbook."

    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 512

    # Network timeout (seconds) for the request to llama.cpp
    REQUEST_TIMEOUT: float = 60.0

    FALLBACK_ANSWER: str = "I can only answer questions related to the Employee Handbook."

    # ---- llama-server process management ----

    # Set AUTO_START_LLAMA=false to disable and manage llama-server yourself.
    AUTO_START_LLAMA: bool = os.environ.get("AUTO_START_LLAMA", "true").lower() != "false"

    # Path to the llama-server binary. Defaults to assuming it's on PATH.
    LLAMA_SERVER_BINARY: str = os.environ.get("LLAMA_SERVER_BINARY", "llama-server")

    # Path to the .gguf model, relative to where main.py is launched from
    # (project/backend/), matching the project's models/ folder.
    LLAMA_MODEL_PATH: str = os.environ.get("LLAMA_MODEL_PATH", "../models/model.gguf")

    # Extra CLI args passed through to llama-server (context size, threads, etc.)
    LLAMA_EXTRA_ARGS: List[str] = os.environ.get("LLAMA_EXTRA_ARGS", "-c 4096").split()

    # How long to wait for llama-server to report healthy before giving up
    LLAMA_STARTUP_TIMEOUT: float = float(os.environ.get("LLAMA_STARTUP_TIMEOUT", "120"))

    # ---- frontend static file server ----

    # Set AUTO_START_FRONTEND=false to disable and open index.html yourself.
    AUTO_START_FRONTEND: bool = os.environ.get("AUTO_START_FRONTEND", "true").lower() != "false"

    # Whether to automatically open the default browser once the frontend
    # server is up.
    AUTO_OPEN_BROWSER: bool = os.environ.get("AUTO_OPEN_BROWSER", "true").lower() != "false"

    # Folder containing index.html, relative to where main.py is launched
    # from (project/backend/), matching the project's frontend/ folder.
    FRONTEND_DIR: str = os.environ.get("FRONTEND_DIR", "../frontend")

    FRONTEND_HOST: str = os.environ.get("FRONTEND_HOST", "0.0.0.0")
    FRONTEND_PORT: int = int(os.environ.get("FRONTEND_PORT", "5003"))


settings = Settings()


# --------------------------------------------------------------------------
# Pydantic models
# --------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Incoming request body for POST /chat."""

    message: str = Field(
        ...,
        min_length=1,
        description="The user's question or message to the assistant.",
        examples=["How many casual leaves are allowed?"],
    )


class ChatResponse(BaseModel):
    """Response body for POST /chat."""

    answer: str = Field(
        ...,
        description="The assistant's answer.",
        examples=["Employees are entitled to 12 casual leaves annually."],
    )


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = Field(default="ok")
    vector_index: str = Field(
        default="not_loaded",
        description="'loaded', 'not_loaded', or 'error' — status of the RAG vector index.",
    )
    llama_server: str = Field(
        default="unknown",
        description="'ok' or 'unreachable' — health of the underlying llama.cpp server.",
    )


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------

class LlamaServerError(Exception):
    """Raised when the llama.cpp server cannot be reached or returns an error."""


class LlamaServerResponseError(Exception):
    """Raised when the llama.cpp server returns an unexpected/malformed payload."""


class LlamaStartupError(Exception):
    """Raised when llama-server fails to start or never becomes healthy."""


# --------------------------------------------------------------------------
# llama-server process manager
# --------------------------------------------------------------------------

class LlamaProcessManager:
    """
    Starts and stops the local llama-server process so that running
    `main.py` brings up the whole stack: FastAPI + llama.cpp, wired
    together via LLAMA_SERVER_URL.

    If a llama-server is already running and healthy on the configured
    host/port, this manager detects that and leaves it alone instead of
    spawning a second instance.
    """

    def __init__(
        self,
        binary: str = settings.LLAMA_SERVER_BINARY,
        model_path: str = settings.LLAMA_MODEL_PATH,
        host: str = settings.LLAMA_HOST,
        port: int = settings.LLAMA_PORT,
        extra_args: Optional[List[str]] = None,
        health_url: str = settings.LLAMA_HEALTH_URL,
        startup_timeout: float = settings.LLAMA_STARTUP_TIMEOUT,
    ) -> None:
        self.binary = binary
        self.model_path = model_path
        self.host = host
        self.port = port
        self.extra_args = extra_args if extra_args is not None else settings.LLAMA_EXTRA_ARGS
        self.health_url = health_url
        self.startup_timeout = startup_timeout

        self._process: Optional[subprocess.Popen] = None
        self._we_started_it = False

    def _is_healthy(self) -> bool:
        try:
            resp = requests.get(self.health_url, timeout=2)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def is_healthy(self) -> bool:
        """Public health check — used by the /health route."""
        return self._is_healthy()

    def start(self) -> None:
        """Start llama-server if it isn't already running, and wait until healthy."""
        if not settings.AUTO_START_LLAMA:
            logger.info("AUTO_START_LLAMA is disabled; assuming llama-server is managed externally.")
            return

        if self._is_healthy():
            logger.info(
                "llama-server already running and healthy at %s:%s — not starting a new instance.",
                self.host,
                self.port,
            )
            return

        resolved_binary = shutil.which(self.binary) or self.binary
        if shutil.which(self.binary) is None and not os.path.isfile(self.binary):
            raise LlamaStartupError(
                f"Could not find the llama-server binary ('{self.binary}'). "
                "Set LLAMA_SERVER_BINARY to its full path, or start llama-server "
                "yourself and set AUTO_START_LLAMA=false."
            )

        if not os.path.isfile(self.model_path):
            raise LlamaStartupError(
                f"Model file not found at '{self.model_path}'. "
                "Set LLAMA_MODEL_PATH to the correct .gguf path."
            )

        cmd = [
            resolved_binary,
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            *self.extra_args,
        ]
        logger.info("Starting llama-server: %s", " ".join(cmd))

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
        except OSError as exc:
            raise LlamaStartupError(f"Failed to launch llama-server: {exc}") from exc

        self._we_started_it = True

        logger.info("Waiting for llama-server to become healthy at %s ...", self.health_url)
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise LlamaStartupError(
                    f"llama-server exited early with code {self._process.returncode}. "
                    "Check the logs above for details."
                )
            if self._is_healthy():
                logger.info("llama-server is up and healthy.")
                return
            time.sleep(1)

        self.stop()
        raise LlamaStartupError(
            f"llama-server did not become healthy within {self.startup_timeout:.0f}s."
        )

    def stop(self) -> None:
        """Stop llama-server, but only if this manager started it."""
        if not self._we_started_it or self._process is None:
            return

        if self._process.poll() is not None:
            self._process = None
            return

        logger.info("Stopping llama-server (pid %s)...", self._process.pid)
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server did not exit in time, killing it.")
            self._process.kill()
            self._process.wait(timeout=5)

        self._process = None
        self._we_started_it = False
        logger.info("llama-server stopped.")


llama_process_manager = LlamaProcessManager()


# --------------------------------------------------------------------------
# Frontend static file server
# --------------------------------------------------------------------------

class FrontendServer:
    """
    Serves frontend/index.html on a local port in a background thread, so
    that running main.py brings up the frontend too — no separate script
    or terminal needed.

    Runs in-process (a daemon thread), not a subprocess, since serving a
    handful of static files doesn't need its own OS process.
    """

    def __init__(
        self,
        directory: str = settings.FRONTEND_DIR,
        host: str = settings.FRONTEND_HOST,
        port: int = settings.FRONTEND_PORT,
        auto_open_browser: bool = settings.AUTO_OPEN_BROWSER,
    ) -> None:
        self.directory = directory
        self.host = host
        self.port = port
        self.auto_open_browser = auto_open_browser

        self._httpd: Optional[http.server.ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def _is_port_in_use(self) -> bool:
        try:
            resp = requests.get(f"http://{self.host}:{self.port}", timeout=1)
            return resp.status_code < 500
        except requests.exceptions.RequestException:
            return False

    def start(self) -> None:
        if not settings.AUTO_START_FRONTEND:
            logger.info("AUTO_START_FRONTEND is disabled; not serving the frontend.")
            return

        resolved_dir = Path(self.directory).resolve()
        if not resolved_dir.is_dir():
            logger.error(
                "Frontend directory not found at %s — skipping frontend server. "
                "Set FRONTEND_DIR to the correct path.",
                resolved_dir,
            )
            return

        index_file = resolved_dir / "index.html"
        if not index_file.is_file():
            logger.warning(
                "No index.html found in %s — the frontend server will start, "
                "but there may be nothing to load.",
                resolved_dir,
            )

        if self._is_port_in_use():
            logger.info(
                "Something is already serving on %s:%s — not starting a second frontend server.",
                self.host,
                self.port,
            )
            url = f"http://{self.host}:{self.port}"
            if self.auto_open_browser:
                threading.Timer(0.3, lambda: webbrowser.open(url)).start()
            return

        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler,
            directory=str(resolved_dir),
        )

        try:
            self._httpd = http.server.ThreadingHTTPServer((self.host, self.port), handler)
        except OSError as exc:
            logger.error("Could not start frontend server on %s:%s — %s", self.host, self.port, exc)
            self._httpd = None
            return

        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

        url = f"http://{self.host}:{self.port}"
        logger.info("Serving frontend from %s at %s", resolved_dir, url)

        if self.auto_open_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    def stop(self) -> None:
        if self._httpd is None:
            return

        logger.info("Stopping frontend server...")
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

        self._httpd = None
        self._thread = None
        logger.info("Frontend server stopped.")


frontend_server = FrontendServer()


# --------------------------------------------------------------------------
# llama.cpp client
# --------------------------------------------------------------------------

class LlamaClient:
    """Thin client for the local llama.cpp OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        server_url: str = settings.LLAMA_SERVER_URL,
        temperature: float = settings.TEMPERATURE,
        max_tokens: int = settings.MAX_TOKENS,
        timeout: float = settings.REQUEST_TIMEOUT,
    ) -> None:
        self.server_url = server_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _build_payload(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }

    def get_answer(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a system prompt + user message to llama.cpp and return the
        model's answer.

        Raises:
            LlamaServerError: if the server is unreachable, times out, or
                returns a non-2xx status code.
            LlamaServerResponseError: if the response body is not in the
                expected OpenAI-compatible chat completion format.
        """
        payload = self._build_payload(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info("Sending request to llama.cpp server at %s", self.server_url)

        try:
            response = requests.post(
                self.server_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
        except requests.exceptions.ConnectTimeout as exc:
            logger.error("Connection to llama.cpp server timed out: %s", exc)
            raise LlamaServerError("Timed out connecting to the model server.") from exc
        except requests.exceptions.ConnectionError as exc:
            logger.error("Could not connect to llama.cpp server: %s", exc)
            raise LlamaServerError(
                "Could not connect to the model server. Is llama-server running?"
            ) from exc
        except requests.exceptions.Timeout as exc:
            logger.error("Request to llama.cpp server timed out: %s", exc)
            raise LlamaServerError("The model server took too long to respond.") from exc
        except requests.exceptions.RequestException as exc:
            logger.error("Unexpected error calling llama.cpp server: %s", exc)
            raise LlamaServerError("Unexpected error contacting the model server.") from exc

        if response.status_code != 200:
            logger.error(
                "llama.cpp server returned status %s: %s",
                response.status_code,
                response.text[:500],
            )
            raise LlamaServerError(
                f"Model server returned an error (status {response.status_code})."
            )

        try:
            data = response.json()
        except ValueError as exc:
            logger.error("Failed to parse JSON from llama.cpp server: %s", exc)
            raise LlamaServerResponseError(
                "Model server returned an invalid JSON response."
            ) from exc

        try:
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected response structure from llama.cpp server: %s", data)
            raise LlamaServerResponseError(
                "Model server response was not in the expected format."
            ) from exc

        answer = (answer or "").strip()
        if not answer:
            logger.warning("Model returned an empty answer, using fallback message.")
            answer = settings.FALLBACK_ANSWER

        logger.info("Received answer from llama.cpp server (%d chars)", len(answer))
        return answer


# Single shared client instance used by the FastAPI app
llama_client = LlamaClient()


# --------------------------------------------------------------------------
# RAG-driven answer generation
# --------------------------------------------------------------------------

# Greetings/small talk should get a friendly, direct reply instead of being
# run through RAG search (where they'll score low and trigger the "I can
# only answer questions related to the Employee Handbook" refusal — correct
# for random unrelated questions, but a poor experience for "hi").
#
# Matched as a WHOLE message (after lowercasing + stripping punctuation),
# not a substring, so this never intercepts real questions that happen to
# start with a greeting-like word (e.g. "hey, what's the leave policy?"
# still falls through to RAG below).
_SMALLTALK_REPLIES = {
    r"hi|hii+|hello+|hey+|yo|hola": (
        "Hello! I'm the Employee Handbook Assistant. "
        "Ask me anything about company policies, leave, benefits, or other HR topics."
    ),
    r"good\s?morning": "Good morning! How can I help with your HR questions today?",
    r"good\s?afternoon": "Good afternoon! How can I help with your HR questions today?",
    r"good\s?evening": "Good evening! How can I help with your HR questions today?",
    r"how are you|how('| i)?s it going|what'?s up": (
        "I'm doing well, thanks for asking! I'm here to help with questions "
        "about the employee handbook — leave, benefits, policies, and more."
    ),
    r"who are you|what are you|what can you do|help": (
        "I'm the Employee Handbook Assistant. I can answer questions about "
        "company policies, leave, benefits, working hours, and anything else "
        "covered in the employee handbook."
    ),
    r"thanks?|thank\s?you|thx|ty": "You're welcome! Let me know if you have any other HR questions.",
    r"bye|goodbye|see\s?you|see\s?ya": "Goodbye! Feel free to come back anytime you have HR questions.",
    r"ok|okay|cool|great|nice|got\s?it": "Great! Let me know if you have any questions about the handbook.",
}

_SMALLTALK_PATTERN = re.compile(
    r"^(" + "|".join(_SMALLTALK_REPLIES.keys()) + r")[\s!.?]*$",
    re.IGNORECASE,
)


def _detect_smalltalk(question: str) -> Optional[str]:
    """
    Return a canned reply if the whole message is a greeting/small-talk
    phrase, else None. Only matches the ENTIRE (normalized) message, so
    real handbook questions are never accidentally intercepted.
    """
    normalized = re.sub(r"\s+", " ", question.strip().lower())
    if not normalized:
        return None

    match = _SMALLTALK_PATTERN.match(normalized)
    if not match:
        return None

    matched_group = match.group(1)
    for pattern, reply in _SMALLTALK_REPLIES.items():
        if re.fullmatch(pattern, matched_group, re.IGNORECASE):
            return reply

    return None


def _build_user_message(context: str, question: str) -> str:
    """Assemble the exact user-turn prompt sent to the model."""
    return (
        f"Handbook Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:"
    )


def generate_retrieval_query(question: str) -> str:
    """
    Use the local model to rewrite an indirect/casual employee question
    into clear Employee Handbook search terms.

    This is NOT the final answer. It only improves RAG retrieval.
    The final answer is still generated only from retrieved handbook context.
    """
    user_message = (
        "User question:\n"
        f"{question}\n\n"
        "Rewritten Employee Handbook search query:"
    )

    try:
        rewritten_query = llama_client.get_answer(
            system_prompt=settings.QUERY_REWRITE_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=0.1,
            max_tokens=120,
        ).strip()
    except (LlamaServerError, LlamaServerResponseError) as exc:
        logger.warning(
            "Model-first retrieval query generation failed. "
            "Falling back to original question. Error: %s",
            exc,
        )
        return question

    if not rewritten_query:
        logger.info("Model generated an empty retrieval query; using original question.")
        return question

    # Keep retrieval query clean if the model accidentally returns labels/quotes.
    rewritten_query = re.sub(
        r"^(search query|rewritten query|rewritten employee handbook search query)\s*:\s*",
        "",
        rewritten_query,
        flags=re.IGNORECASE,
    ).strip().strip('"').strip("'")

    logger.info("Original question: %s", question)
    logger.info("Model-generated retrieval query: %s", rewritten_query)
    return rewritten_query or question


def generate_answer(question: str) -> str:
    """
    The full model-first RAG pipeline:

        question -> smalltalk check -> model-generated retrieval query
        -> RAG search using rewritten query -> retrieved context -> Qwen -> answer

    If the rewritten query finds nothing, we fall back to searching the original
    question once. If no context is found after both attempts, we return the
    refusal message.
    """
    smalltalk_reply = _detect_smalltalk(question)
    if smalltalk_reply is not None:
        logger.info("Detected small talk — replying directly without RAG/Qwen.")
        return smalltalk_reply

    retrieval_query = generate_retrieval_query(question)
    context = rag_system.search_context(retrieval_query)

    if context is None and retrieval_query != question:
        logger.info("No context found for rewritten query — trying original question.")
        context = rag_system.search_context(question)

    if context is None:
        logger.info("No relevant handbook context found — returning refusal message.")
        return settings.REFUSAL_MESSAGE

    user_message = _build_user_message(context, question)
    answer = llama_client.get_answer(
        system_prompt=settings.SYSTEM_PROMPT,
        user_message=user_message,
    )
    return answer