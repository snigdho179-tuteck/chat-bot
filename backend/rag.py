"""
rag.py
------
Retrieval-Augmented Generation layer for the Employee Handbook Assistant.

Responsibilities:
    1. Load the handbook PDF and extract its text (PyMuPDF).
    2. Split the text into overlapping chunks.
    3. Embed each chunk with a sentence-transformers model.
    4. Build an in-memory FAISS index over those embeddings.
    5. Given a question, embed it, retrieve the top-k most similar chunks,
       and return them as a single combined context string — or None if
       nothing retrieved is actually relevant (score filtering), so the
       caller can refuse to answer instead of hallucinating.

The index is built once at startup via RAGSystem.initialize() and then
reused for every request; it is never rebuilt on the fly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import fitz  # PyMuPDF
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load .env from this file's own directory, same approach as backend.py.
# Safe to call again even if backend.py already loaded it — load_dotenv()
# does not override variables that are already set in the environment.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

logger = logging.getLogger("ai_assistant.rag")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

class RAGSettings:
    """Static configuration for the RAG pipeline."""

    # Path to the handbook PDF, relative to where main.py is launched from
    # (project/backend/). Override via .env if your PDF lives elsewhere.
    PDF_PATH: str = os.environ.get("RAG_PDF_PATH", "../data/Employee_Handbook.pdf")

    CHUNK_SIZE: int = int(os.environ.get("RAG_CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.environ.get("RAG_CHUNK_OVERLAP", "100"))

    EMBEDDING_MODEL_NAME: str = os.environ.get(
        "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    TOP_K: int = int(os.environ.get("RAG_TOP_K", "3"))

    # Cosine-similarity threshold (embeddings are L2-normalized, so FAISS
    # inner product == cosine similarity, range [-1, 1]). Retrieved chunks
    # scoring below this are treated as "not actually relevant".
    SIMILARITY_THRESHOLD: float = float(os.environ.get("RAG_SIMILARITY_THRESHOLD", "0.35"))


rag_settings = RAGSettings()


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------

class RAGError(Exception):
    """Base class for RAG-related errors."""


class RAGInitializationError(RAGError):
    """Raised when the handbook PDF can't be loaded or the index can't be built."""


class RAGNotReadyError(RAGError):
    """Raised when search_context() is called before the index has been built."""


# --------------------------------------------------------------------------
# Text extraction & chunking
# --------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF using PyMuPDF.

    Raises:
        RAGInitializationError: if the file is missing or cannot be read.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise RAGInitializationError(f"Handbook PDF not found at '{path.resolve()}'.")

    try:
        text_parts: List[str] = []
        with fitz.open(path) as doc:
            for page in doc:
                text_parts.append(page.get_text())
        full_text = "\n".join(text_parts).strip()
    except Exception as exc:  # noqa: BLE001 - PyMuPDF can raise several error types
        raise RAGInitializationError(f"Failed to read PDF '{path}': {exc}") from exc

    if not full_text:
        raise RAGInitializationError(
            f"No extractable text found in '{path}'. "
            "It may be a scanned/image-only PDF that needs OCR."
        )

    return full_text


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping fixed-size character chunks.

    E.g. chunk_size=500, overlap=100 means each chunk starts 400 characters
    after the previous one, so consecutive chunks share a 100-character tail.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and less than chunk_size")

    step = chunk_size - overlap
    chunks: List[str] = []

    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


# --------------------------------------------------------------------------
# RAG system
# --------------------------------------------------------------------------

class RAGSystem:
    """
    Owns the embedding model, the FAISS index, and the chunk store.

    Usage:
        rag_system.initialize()               # once, at app startup
        context = rag_system.search_context(q) # per request
    """

    def __init__(
        self,
        pdf_path: str = rag_settings.PDF_PATH,
        chunk_size: int = rag_settings.CHUNK_SIZE,
        chunk_overlap: int = rag_settings.CHUNK_OVERLAP,
        embedding_model_name: str = rag_settings.EMBEDDING_MODEL_NAME,
        top_k: int = rag_settings.TOP_K,
        similarity_threshold: float = rag_settings.SIMILARITY_THRESHOLD,
    ) -> None:
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        self._embedding_model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.Index] = None
        self._chunks: List[str] = []

        self._status: str = "not_loaded"  # "not_loaded" | "loaded" | "error"
        self._last_error: Optional[str] = None

    # ---- public state ----

    @property
    def status(self) -> str:
        """'not_loaded', 'loaded', or 'error' — used by GET /health."""
        return self._status

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def is_ready(self) -> bool:
        return self._status == "loaded" and self._index is not None

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    # ---- initialization (build once at startup) ----

    def initialize(self) -> None:
        """
        Load the PDF, chunk it, embed the chunks, and build the FAISS
        index. Safe to call more than once — subsequent calls are no-ops
        if the index is already loaded.

        Raises:
            RAGInitializationError: if any step fails. Callers should
                decide whether that's fatal for their use case.
        """
        if self.is_ready:
            logger.info("RAG index already built — skipping re-initialization.")
            return

        try:
            logger.info("Loading employee handbook...")
            raw_text = extract_text_from_pdf(self.pdf_path)

            logger.info("Building vector index...")
            chunks = chunk_text(raw_text, self.chunk_size, self.chunk_overlap)
            if not chunks:
                raise RAGInitializationError(
                    "Handbook produced zero chunks — check CHUNK_SIZE/CHUNK_OVERLAP."
                )

            model = SentenceTransformer(self.embedding_model_name)
            embeddings = model.encode(
                chunks,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")

            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # inner product on normalized vectors = cosine similarity
            index.add(embeddings)

            self._embedding_model = model
            self._index = index
            self._chunks = chunks
            self._status = "loaded"
            self._last_error = None

            logger.info(
                "Vector index loaded successfully. (%d chunks, dim=%d)",
                len(chunks),
                dimension,
            )

        except RAGInitializationError as exc:
            self._status = "error"
            self._last_error = str(exc)
            logger.error("RAG initialization failed: %s", exc)
            raise
        except Exception as exc:  # noqa: BLE001 - surface anything unexpected clearly
            self._status = "error"
            self._last_error = str(exc)
            logger.exception("Unexpected error during RAG initialization")
            raise RAGInitializationError(f"Unexpected error building the vector index: {exc}") from exc

    # ---- retrieval ----

    def _embed_query(self, question: str) -> np.ndarray:
        assert self._embedding_model is not None  # guarded by is_ready check in search_context
        return self._embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

    def _top_matches(self, question: str) -> List[Tuple[float, str]]:
        """Return (score, chunk_text) pairs for the top-k matches, best first."""
        query_embedding = self._embed_query(question)
        scores, indices = self._index.search(query_embedding, self.top_k)

        matches: List[Tuple[float, str]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS pads with -1 when there are fewer than top_k results
            matches.append((float(score), self._chunks[idx]))

        return matches

    def search_context(self, question: str) -> Optional[str]:
        """
        Embed the question, retrieve the top-k most similar chunks, and
        return them combined into a single context string.

        Returns:
            The combined context string, or None if no retrieved chunk
            meets the similarity threshold (i.e. the question is likely
            unrelated to the handbook).

        Raises:
            RAGNotReadyError: if the index hasn't been built yet.
        """
        if not self.is_ready:
            raise RAGNotReadyError(
                "The vector index is not ready yet. Call initialize() at startup first."
            )

        question = (question or "").strip()
        if not question:
            return None

        matches = self._top_matches(question)

        if not matches:
            logger.info("No matches returned by the vector index for this question.")
            return None

        best_score = matches[0][0]
        logger.info(
            "Top match score=%.4f (threshold=%.4f) for question: %r",
            best_score,
            self.similarity_threshold,
            question,
        )

        if best_score < self.similarity_threshold:
            logger.info("Best match score is below threshold — treating as out-of-scope.")
            return None

        relevant_chunks = [chunk for score, chunk in matches if score >= self.similarity_threshold]
        combined_context = "\n\n---\n\n".join(relevant_chunks)
        return combined_context


# Single shared instance used by the FastAPI app. Built once via
# rag_system.initialize() in main.py's startup/lifespan handler.
rag_system = RAGSystem()