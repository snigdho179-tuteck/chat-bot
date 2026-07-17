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

# deep-translator is an optional dependency: if it's missing (or the network
# is unreachable at request time), translation quietly no-ops and the
# original text passes through untouched, rather than crashing /chat.
try:
    from deep_translator import GoogleTranslator

    _TRANSLATOR_AVAILABLE = True
except ImportError:  # pragma: no cover
    GoogleTranslator = None  # type: ignore[assignment]
    _TRANSLATOR_AVAILABLE = False

from rag import RAGInitializationError, RAGNotReadyError, rag_manager, rag_system

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

if not _TRANSLATOR_AVAILABLE:
    logger.warning(
        "deep-translator is not installed — non-English messages will be sent "
        "to the model as-is, untranslated. Run `pip install deep-translator` "
        "to enable automatic translation."
    )

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

class Settings:
    """Static configuration for the backend."""

    LLAMA_HOST: str = os.environ.get("LLAMA_HOST", "0.0.0.0")
    LLAMA_CLIENT_HOST: str = os.environ.get("LLAMA_CLIENT_HOST", "127.0.0.1")
    LLAMA_PORT: int = int(os.environ.get("LLAMA_PORT", "8080"))

    LLAMA_SERVER_URL = (f"http://{LLAMA_CLIENT_HOST}:{LLAMA_PORT}/v1/chat/completions")

    LLAMA_HEALTH_URL = (f"http://{LLAMA_CLIENT_HOST}:{LLAMA_PORT}/health")

    SYSTEM_PROMPT: str = (
        "You are an Employee Handbook Assistant.\n\n"
        "Answer ONLY from the supplied handbook context.\n"
        "Never use your own knowledge.\n\n"
        "Conversation history is provided for reference only, so you can "
        "resolve follow-up questions (e.g. pronouns, 'what about...') — it "
        "is not a source of facts. Every factual claim must still come "
        "from the handbook context.\n\n"
        "If the context does not contain the answer,\n"
        "reply exactly:\n"
        '"I can only answer questions related to the Employee Handbook."'
    )

    REFUSAL_MESSAGE: str = "I can only answer questions related to the Employee Handbook."

    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 512

    # Network timeout (seconds) for the request to llama.cpp
    REQUEST_TIMEOUT: float = 60.0

    FALLBACK_ANSWER: str = "I can only answer questions related to the Employee Handbook."

    # How many previous (question, answer) turns to keep per session and
    # feed back to the model as conversation history. Kept small — this is
    # for resolving follow-ups like "what about for contract staff?", not
    # for long-term memory, and every turn included costs prompt tokens.
    MAX_HISTORY_TURNS: int = int(os.environ.get("RAG_MAX_HISTORY_TURNS", "5"))

    # ---- llama-server process management ----

    # Set AUTO_START_LLAMA=false to disable and manage llama-server yourself.
    AUTO_START_LLAMA: bool = os.environ.get("AUTO_START_LLAMA", "true").lower() != "false"

    # Path to the llama-server binary. Defaults to assuming it's on PATH.
    LLAMA_SERVER_BINARY: str = os.environ.get("LLAMA_SERVER_BINARY", "llama-server")

    # Path to the .gguf model, relative to where main.py is launched from
    # (project/backend/), matching the project's models/ folder.
    LLAMA_MODEL_PATH: str = os.environ.get("LLAMA_MODEL_PATH", "../models/model.gguf")

    # Extra CLI args passed through to llama-server (context size, threads, etc.)
    LLAMA_EXTRA_ARGS: List[str] = os.environ.get("LLAMA_EXTRA_ARGS", "-c  32768").split()

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
    language: str = Field(
        default="en-IN",
        description=(
            "BCP-47 language code the message was written in, matching the "
            "frontend's language dropdown (e.g. 'en-IN', 'bn-IN', 'hi-IN', "
            "'ta-IN', 'te-IN', 'mr-IN'). The message is translated to English "
            "before RAG search + the model, and the model's answer is "
            "translated back into this language before being returned."
        ),
        examples=["bn-IN"],
    )
    department: str = Field(
        default="hr",
        description=(
            "Which department tab the question was asked from (e.g. 'hr', "
            "'marketing', 'finance', 'sales'). Selects which department's "
            "document index + answered-query training data to search."
        ),
        examples=["hr"],
    )
    session_id: str = Field(
        default="default",
        description=(
            "Identifies a single chat session so follow-up questions ('what "
            "about for contract staff?') can be answered using earlier turns "
            "as conversation history. The frontend should generate one ID per "
            "browser session/tab and send it on every request; if omitted, "
            "all callers share one 'default' history."
        ),
        examples=["b3f1c9d2-4a2e-4e9a-9a6a-6b8f6b1a9e11"],
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

class _LoginFirstRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    Same as SimpleHTTPRequestHandler, except a request for "/" is served
    login.html instead of index.html — so visiting the app always lands
    on the login screen first. index.html itself has its own client-side
    guard that bounces back to login.html if there's no active session.
    """

    def do_GET(self) -> None:  # noqa: N802 - stdlib method name
        if self.path == "/":
            self.path = "/login.html"
        super().do_GET()


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
        login_file = resolved_dir / "login.html"
        if not login_file.is_file():
            logger.warning(
                "No login.html found in %s — requests to \"/\" will 404. "
                "Add login.html so the app opens on the login screen first.",
                resolved_dir,
            )
        if not index_file.is_file():
            logger.warning(
                "No index.html found in %s — the frontend server will start, "
                "but there may be nothing to load after login.",
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
            _LoginFirstRequestHandler,
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

    def _build_payload(self, system_prompt: str, user_message: str) -> Dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

    def get_answer(self, system_prompt: str, user_message: str) -> str:
        """
        Send a system prompt + user message to llama.cpp and return the
        model's answer.

        Raises:
            LlamaServerError: if the server is unreachable, times out, or
                returns a non-2xx status code.
            LlamaServerResponseError: if the response body is not in the
                expected OpenAI-compatible chat completion format.
        """
        payload = self._build_payload(system_prompt, user_message)
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
# Translation (frontend language <-> English)
# --------------------------------------------------------------------------
#
# The handbook, the system prompt, and the model are all English-only. So
# for non-English messages we:
#   1. translate the incoming message into English before RAG search + Qwen
#   2. translate Qwen's English answer back into the original language
#
# Translation failures (missing package, no internet, API hiccup) never
# take down /chat — we log a warning and fall back to the untranslated
# text, so the assistant still responds (just in whatever language it
# has on hand) instead of erroring out.

def _bcp47_to_iso639(language: str) -> str:
    """'bn-IN' -> 'bn', 'en-IN' -> 'en', already-bare codes pass through."""
    return (language or "en").split("-")[0].strip().lower() or "en"


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate `text` from `source_lang` to `target_lang` (ISO 639-1 codes,
    e.g. 'en', 'bn', 'hi'). Returns `text` unchanged if the languages match,
    translation isn't available, or the translation call fails.
    """
    if not text or not text.strip():
        return text

    source_lang = _bcp47_to_iso639(source_lang)
    target_lang = _bcp47_to_iso639(target_lang)

    if source_lang == target_lang:
        return text

    if not _TRANSLATOR_AVAILABLE:
        logger.warning(
            "Translation requested (%s -> %s) but deep-translator isn't installed; "
            "returning original text.",
            source_lang, target_lang,
        )
        return text

    try:
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        if not translated or not translated.strip():
            logger.warning("Translation returned empty result (%s -> %s); using original text.",
                            source_lang, target_lang)
            return text
        return translated
    except Exception as exc:  # noqa: BLE001 - never let translation break /chat
        logger.warning("Translation failed (%s -> %s): %s. Using original text.",
                        source_lang, target_lang, exc)
        return text


# --------------------------------------------------------------------------
# Conversation history (per session, in-memory)
# --------------------------------------------------------------------------
#
# Keeps the last few (question, answer) turns for each (department,
# session_id) pair, so a follow-up question like "what about for interns?"
# can be resolved using what was just discussed. This mirrors the
# conv_history list from the local CLI prototype, but keyed per session
# (rather than one global list) since the HTTP backend serves multiple
# users/tabs concurrently, and capped at MAX_HISTORY_TURNS so it can't
# grow without bound or crowd out the actual handbook context.
#
# In-memory only: history resets on restart and isn't shared across
# multiple backend processes. That's fine for this app's single-process
# deployment; swap in a persisted/shared store if that ever changes.

class ConversationHistoryStore:
    """Thread-safe store of recent Q&A turns per (department, session_id)."""

    def __init__(self, max_turns: int = settings.MAX_HISTORY_TURNS) -> None:
        self.max_turns = max_turns
        self._lock = threading.Lock()
        self._history: Dict[tuple, List[Dict[str, str]]] = {}

    def _key(self, department: str, session_id: str) -> tuple:
        return ((department or "hr").strip().lower(), (session_id or "default").strip() or "default")

    def get_context(self, department: str, session_id: str) -> str:
        """Render prior turns as text for the prompt, oldest first. Empty
        string (not None) if there's no history yet, so callers can splice
        it into a template unconditionally."""
        key = self._key(department, session_id)
        with self._lock:
            turns = list(self._history.get(key, []))

        if not turns:
            return ""

        return "\n---\n".join(f"User: {t['question']}\nAssistant: {t['answer']}" for t in turns)

    def add_turn(self, department: str, session_id: str, question: str, answer: str) -> None:
        key = self._key(department, session_id)
        with self._lock:
            turns = self._history.setdefault(key, [])
            turns.append({"question": question, "answer": answer})
            if len(turns) > self.max_turns:
                del turns[: len(turns) - self.max_turns]

    def clear(self, department: str, session_id: str) -> None:
        """Used when the frontend starts a fresh chat / the user hits 'clear'."""
        key = self._key(department, session_id)
        with self._lock:
            self._history.pop(key, None)


# Single shared instance used by the FastAPI app
conversation_history = ConversationHistoryStore()


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
# Department-specific display name + topic blurb used to build the smalltalk
# replies below. Any department slug NOT listed here (e.g. a brand-new one
# an admin just uploaded a document for) still works — it falls back to a
# title-cased version of the slug and a generic "policies and procedures"
# blurb, so this never needs to be touched when a department is added.
_DEPARTMENT_LABELS: Dict[str, str] = {
    "hr": "HR",
    "finance": "Finance",
    "marketing": "Marketing",
    "sales": "Sales",
    "it": "IT",
    "legal": "Legal",
}

_DEPARTMENT_TOPICS: Dict[str, str] = {
    "hr": "company policies, leave, and benefits",
    "finance": "expense policies, invoices, and budgets",
    "marketing": "marketing guidelines and brand policies",
    "sales": "sales processes and policies",
    "it": "IT policies and systems",
    "legal": "legal and compliance policies",
}


def _department_label(department: str) -> str:
    slug = (department or "hr").strip().lower()
    return _DEPARTMENT_LABELS.get(slug, slug.replace("_", " ").replace("-", " ").title() or "HR")


def _department_topics(department: str) -> str:
    slug = (department or "hr").strip().lower()
    return _DEPARTMENT_TOPICS.get(slug, "policies and procedures")


# Each value is a function of the department slug, so the same greeting
# ("hi", "thanks", etc.) gets a reply scoped to whichever tab the question
# came from, instead of always talking about HR/the employee handbook.
_SMALLTALK_REPLIES = {
    r"hi|hii+|hello+|hey+|yo|hola": (
        lambda d: (
            f"Hello! I'm the {_department_label(d)} Assistant. "
            f"Ask me anything about {_department_topics(d)}."
        )
    ),
    r"good\s?morning": lambda d: f"Good morning! How can I help with your {_department_label(d)} questions today?",
    r"good\s?afternoon": lambda d: f"Good afternoon! How can I help with your {_department_label(d)} questions today?",
    r"good\s?evening": lambda d: f"Good evening! How can I help with your {_department_label(d)} questions today?",
    r"how are you|how('| i)?s it going|what'?s up": (
        lambda d: (
            "I'm doing well, thanks for asking! I'm here to help with questions "
            f"about {_department_topics(d)}."
        )
    ),
    r"who are you|what are you|what can you do|help": (
        lambda d: (
            f"I'm the {_department_label(d)} Assistant. I can answer questions about "
            f"{_department_topics(d)}, and anything else covered in the {_department_label(d)} documents."
        )
    ),
    r"thanks?|thank\s?you|thx|ty": lambda d: f"You're welcome! Let me know if you have any other {_department_label(d)} questions.",
    r"bye|goodbye|see\s?you|see\s?ya": lambda d: f"Goodbye! Feel free to come back anytime you have {_department_label(d)} questions.",
    r"ok|okay|cool|great|nice|got\s?it": lambda d: f"Great! Let me know if you have any questions about {_department_topics(d)}.",
}

_SMALLTALK_PATTERN = re.compile(
    r"^(" + "|".join(_SMALLTALK_REPLIES.keys()) + r")[\s!.?]*$",
    re.IGNORECASE,
)


def _detect_smalltalk(question: str, department: str = "hr") -> Optional[str]:
    """
    Return a canned reply if the whole message is a greeting/small-talk
    phrase, else None. Only matches the ENTIRE (normalized) message, so
    real handbook questions are never accidentally intercepted.

    The reply text is generated for `department`, so the same "hi" gets a
    Finance-flavored reply in the Finance tab and an HR-flavored reply in
    the Hr tab instead of always describing itself as the "Employee
    Handbook Assistant".
    """
    normalized = re.sub(r"\s+", " ", question.strip().lower())
    if not normalized:
        return None

    match = _SMALLTALK_PATTERN.match(normalized)
    if not match:
        return None

    matched_group = match.group(1)
    for pattern, reply_fn in _SMALLTALK_REPLIES.items():
        if re.fullmatch(pattern, matched_group, re.IGNORECASE):
            return reply_fn(department)

    return None


def _build_user_message(context: str, question: str, history_text: str = "") -> str:
    """Assemble the exact user-turn prompt sent to the model.

    `history_text` (from ConversationHistoryStore.get_context) is inserted
    even when empty, so the model always sees the same three-section
    shape and there's no special-casing between a session's first message
    and its later ones.
    """
    return (
        f"Handbook Context:\n{context}\n\n"
        f"Conversation History:\n{history_text or '(none yet)'}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:"
    )


def generate_answer(question: str, language: str = "en-IN", department: str = "hr", session_id: str = "default") -> str:
    """
    The full question -> answer pipeline, with translation at both ends:

        question (any language)
          -> translate to English
          -> smalltalk check -> RAG search (for `department`) -> retrieved context -> Qwen -> answer (English)
          -> translate back to the original language

    Greetings/small talk get a friendly canned reply directly (translated
    back to the original language same as any other answer).
    If RAG search finds nothing relevant, Qwen is never called — we
    return the fixed refusal message instead (also translated back).

    `session_id` scopes conversation history (see ConversationHistoryStore)
    so a follow-up question is answered with the last few turns of this
    same session in mind. History is stored/replayed in English regardless
    of `language`, same as the handbook context and system prompt.

    Raises:
        RAGNotReadyError: if the vector index hasn't finished building yet.
        LlamaServerError / LlamaServerResponseError: if llama.cpp fails.
    """
    lang_code = _bcp47_to_iso639(language)

    english_question = translate_text(question, source_lang=lang_code, target_lang="en")
    if english_question != question:
        logger.info("Translated incoming message %s -> en for processing.", lang_code)

    answer_en = _generate_answer_en(english_question, department=department, session_id=session_id)

    if lang_code == "en":
        return answer_en

    translated_answer = translate_text(answer_en, source_lang="en", target_lang=lang_code)
    if translated_answer != answer_en:
        logger.info("Translated outgoing answer en -> %s before returning.", lang_code)
    return translated_answer


def _generate_answer_en(question: str, department: str = "hr", session_id: str = "default") -> str:
    """The original English-only pipeline: smalltalk -> RAG -> Qwen -> answer."""
    smalltalk_reply = _detect_smalltalk(question, department=department)
    if smalltalk_reply is not None:
        logger.info("Detected small talk — replying directly without RAG/Qwen.")
        # Small talk doesn't get logged as history — it's not a handbook
        # Q&A turn, and folding it in would just waste context on later
        # follow-ups without helping resolve them.
        return smalltalk_reply

    context = rag_manager.search_context(department, question)

    if context is None:
        logger.info("No relevant context found for department '%s' — returning refusal message.", department)
        return settings.REFUSAL_MESSAGE

    history_text = conversation_history.get_context(department, session_id)
    user_message = _build_user_message(context, question, history_text)

    print("\n" + "=" * 80)
    print("final prompt")
    print("=" * 80)
    print("MESSAGE")
    print(settings.SYSTEM_PROMPT)
    print("\nUSER MESSAGE")
    print(user_message)
    print("=" * 80 + "\n")

    answer = llama_client.get_answer(
        system_prompt=settings.SYSTEM_PROMPT,
        user_message=user_message,
    )

    conversation_history.add_turn(department, session_id, question, answer)

    return answer