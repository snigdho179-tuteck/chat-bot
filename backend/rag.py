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
import re
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

    # Larger than the original 500/100. Small chunk sizes were splitting
    # enumerated lists (e.g. the 6 types of leave) across two overlapping
    # chunks, so the model ended up seeing the same list twice in two
    # different truncated forms and would drop items when summarizing.
    CHUNK_SIZE: int = int(os.environ.get("RAG_CHUNK_SIZE", "900"))
    CHUNK_OVERLAP: int = int(os.environ.get("RAG_CHUNK_OVERLAP", "150"))

    EMBEDDING_MODEL_NAME: str = os.environ.get(
        "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Retrieve more chunks so indirect/casual wording has a better chance
    # of finding the right policy section. Override from .env if needed.
    TOP_K: int = int(os.environ.get("RAG_TOP_K", "8"))

    # Lower default improves recall for indirect queries. The final answer is
    # still constrained by retrieved handbook context in backend.py.
    SIMILARITY_THRESHOLD: float = float(os.environ.get("RAG_SIMILARITY_THRESHOLD", "0.25"))

    # Keyword fallback helps short/direct questions like "types of leaves" and
    # casual wording when vector search scores are low.
    ENABLE_KEYWORD_FALLBACK: bool = os.environ.get("RAG_ENABLE_KEYWORD_FALLBACK", "true").lower() != "false"
    KEYWORD_TOP_K: int = int(os.environ.get("RAG_KEYWORD_TOP_K", "4"))
    MIN_KEYWORD_OVERLAP: int = int(os.environ.get("RAG_MIN_KEYWORD_OVERLAP", "1"))


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

def extract_segments_from_pdf(pdf_path: str) -> List[Tuple[str, str]]:
    """
    Like extract_text_from_pdf, but tables are pulled out as their own
    atomic segments (kind="table") instead of being left in the regular
    page-text flow (kind="prose").

    Why this matters: a table like the Inland Travel Allowance grid packs
    its column headers ("X/Y/Z Class cities", "Lodging", "TOTAL", ...) at
    the top and then just rows of bare numbers below. If that block gets
    cut by the normal sliding-window chunker, a chunk containing "Grade 1A"
    and its numbers can end up with *no* mention of which city class or
    which column those numbers belong to — the model then correctly can't
    answer, because the chunk genuinely doesn't say. Keeping the whole
    table together as one chunk keeps the labels attached to the numbers.

    Returns:
        List of (kind, text) tuples, kind in {"prose", "table"}.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise RAGInitializationError(f"Handbook PDF not found at '{path.resolve()}'.")

    segments: List[Tuple[str, str]] = []
    try:
        with fitz.open(path) as doc:
            for page in doc:
                page_text = page.get_text()
                if not page_text.strip():
                    continue

                try:
                    tables = page.find_tables().tables
                except Exception:  # noqa: BLE001 - table finding is best-effort
                    tables = []

                if not tables:
                    segments.append(("prose", page_text))
                    continue

                # Best-effort heading: first non-empty, non-footer line on
                # the page, e.g. "INLAND TRAVEL ALLOWANCE POLICY" — gives
                # the table chunk a topic label even if the table itself
                # doesn't repeat it on every row. Skip footer lines like
                # "27 | P a g e", which PyMuPDF sometimes returns first.
                heading = next(
                    (
                        line.strip()
                        for line in page_text.splitlines()
                        if line.strip() and not re.match(r"^\d+\s*\|\s*P\s*a\s*g\s*e\s*$", line.strip())
                    ),
                    "",
                )

                remainder = page_text
                for table in tables:
                    clip_text = page.get_text(clip=table.bbox).strip()
                    if not clip_text:
                        continue
                    table_chunk = f"[Section: {heading}]\n{clip_text}" if heading else clip_text
                    segments.append(("table", table_chunk))
                    # Remove the table's own text from the prose remainder
                    # so it isn't also chunked (and duplicated) as prose.
                    remainder = remainder.replace(clip_text, "")

                remainder = remainder.strip()
                if remainder:
                    segments.append(("prose", remainder))

    except RAGInitializationError:
        raise
    except Exception as exc:  # noqa: BLE001 - PyMuPDF can raise several error types
        raise RAGInitializationError(f"Failed to read PDF '{path}': {exc}") from exc

    if not segments:
        raise RAGInitializationError(
            f"No extractable text found in '{path}'. "
            "It may be a scanned/image-only PDF that needs OCR."
        )

    return segments


def chunk_segments(
    segments: List[Tuple[str, str]], chunk_size: int, overlap: int
) -> List[str]:
    """
    Turn (kind, text) segments into final chunks: "prose" segments go
    through the normal sliding-window chunker; "table" segments are kept
    whole, as a single chunk each, regardless of chunk_size.
    """
    chunks: List[str] = []
    for kind, text in segments:
        if kind == "table":
            text = text.strip()
            if text:
                chunks.append(text)
        else:
            chunks.extend(chunk_text(text, chunk_size, overlap))
    return chunks


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
# Query expansion & keyword fallback helpers
# --------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or",
    "our", "please", "policy", "the", "to", "we", "what", "when", "where", "which",
    "who", "why", "you", "your", "year", "per", "many", "much", "take", "get",
}

_QUERY_EXPANSIONS: Dict[str, str] = {
    "leave": (
        "leave leaves annual leave casual leave sick leave earned leave privilege leave "
        "paid leave unpaid leave holiday vacation time off absence leave entitlement "
        "leave balance types of leave number of leaves per year"
    ),
    "marriage": "marriage leave wedding leave special leave paid time off",
    "wedding": "marriage leave wedding leave special leave paid time off",
    "sick": "sick leave medical leave illness health doctor fever",
    "medical": "sick leave medical leave illness health doctor fever",
    "vacation": "annual leave earned leave privilege leave vacation holiday time off",
    "holiday": "annual leave holiday vacation time off leave",
    "resign": "resignation notice period exit policy termination separation",
    "quit": "resignation notice period exit policy termination separation",
    "notice": "notice period resignation exit policy",
    "late": "attendance working hours late arrival punctuality policy",
    "attendance": "attendance working hours late arrival punctuality policy",
    "salary": "payroll salary compensation wages payment",
    "benefit": "benefits insurance reimbursement allowance employee benefits",
    "benefits": "benefits insurance reimbursement allowance employee benefits",
}

_SYNONYM_TRIGGERS: Dict[str, str] = {
    "off": "leave",
    "absent": "leave",
    "absence": "leave",
    "leaves": "leave",
    "married": "marriage",
    "ill": "sick",
    "fever": "sick",
    "doctor": "medical",
    "trip": "vacation",
    "travel": "vacation",
    "leaving": "resign",
}


def _dedupe_overlapping_chunks(chunks: List[str]) -> List[str]:
    """
    Drop chunks that are near-duplicates of another chunk already kept
    (e.g. two adjacent chunks that both contain most of the same
    enumerated list because of chunk overlap). This keeps the context
    sent to the model from containing the same list twice in two
    different truncated forms, which was causing it to drop items.
    """
    kept: List[str] = []
    for chunk in chunks:
        chunk_norm = re.sub(r"\s+", " ", chunk).strip()
        is_duplicate = False
        for existing in kept:
            existing_norm = re.sub(r"\s+", " ", existing).strip()
            shorter, longer = sorted([chunk_norm, existing_norm], key=len)
            if not shorter:
                continue
            # If ~80%+ of the shorter chunk's words already appear
            # contiguously inside the longer one, treat it as a duplicate.
            if shorter in longer or (len(shorter) > 40 and shorter[:-40] in longer):
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(chunk)
    return kept


def _tokenize_for_keyword_search(text: str) -> List[str]:
    """Tokenize text into normalized keyword tokens for fallback search."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9']+", text.lower())
    normalized: List[str] = []
    for token in tokens:
        token = token.strip("'")
        if token.endswith("s") and len(token) > 3:
            token = token[:-1]
        if token and token not in _STOPWORDS and len(token) > 1:
            normalized.append(token)
    return normalized


def expand_query_for_retrieval(question: str) -> str:
    """
    Expand casual or indirect employee wording into handbook-style retrieval terms.
    This improves recall before vector search and keyword fallback.
    """
    question = (question or "").strip()
    lowered = question.lower()
    expansions: List[str] = []

    for trigger, expansion in _QUERY_EXPANSIONS.items():
        if re.search(rf"\b{re.escape(trigger)}\b", lowered):
            expansions.append(expansion)

    for trigger, canonical in _SYNONYM_TRIGGERS.items():
        if re.search(rf"\b{re.escape(trigger)}\b", lowered):
            expansion = _QUERY_EXPANSIONS.get(canonical)
            if expansion:
                expansions.append(expansion)

    if re.search(r"\b(disappear|away|not come|skip|miss work|few days)\b", lowered):
        expansions.append(_QUERY_EXPANSIONS["leave"])
    if re.search(r"\btypes?\b", lowered) and re.search(r"\bleaves?\b|\boff\b|\babsence\b", lowered):
        expansions.append(_QUERY_EXPANSIONS["leave"])

    if not expansions:
        return question

    seen = set()
    unique_expansions = []
    for expansion in expansions:
        if expansion not in seen:
            unique_expansions.append(expansion)
            seen.add(expansion)

    return "\n".join([question, *unique_expansions])


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
        self.enable_keyword_fallback = rag_settings.ENABLE_KEYWORD_FALLBACK
        self.keyword_top_k = rag_settings.KEYWORD_TOP_K
        self.min_keyword_overlap = rag_settings.MIN_KEYWORD_OVERLAP

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
            segments = extract_segments_from_pdf(self.pdf_path)

            logger.info("Building vector index...")
            chunks = chunk_segments(segments, self.chunk_size, self.chunk_overlap)
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

    def _top_matches(self, question: str, top_k: Optional[int] = None) -> List[Tuple[float, str]]:
        """Return (score, chunk_text) pairs for the top-k vector matches, best first."""
        assert self._index is not None  # guarded by is_ready check in search_context
        query_embedding = self._embed_query(question)
        k = top_k or self.top_k
        scores, indices = self._index.search(query_embedding, k)

        matches: List[Tuple[float, str]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS pads with -1 when there are fewer than top_k results
            matches.append((float(score), self._chunks[idx]))

        return matches

    def _keyword_matches(self, question: str) -> List[Tuple[int, str]]:
        """
        Return keyword-overlap matches as a fallback when vector search is too strict.
        This is intentionally simple and local: no external services, no extra index.
        """
        query_tokens = set(_tokenize_for_keyword_search(expand_query_for_retrieval(question)))
        if not query_tokens:
            return []

        scored: List[Tuple[int, str]] = []
        for chunk in self._chunks:
            chunk_tokens = set(_tokenize_for_keyword_search(chunk))
            overlap = len(query_tokens & chunk_tokens)
            if overlap >= self.min_keyword_overlap:
                scored.append((overlap, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[: self.keyword_top_k]

    def search_context(self, question: str) -> Optional[str]:
        """
        Embed the question, retrieve matching chunks, and return them combined
        into a single context string.

        This version is optimized for indirect/casual HR questions:
        1. Expand the query with handbook-style synonyms.
        2. Run vector search on both original and expanded variants.
        3. If vector scores are below threshold, use keyword fallback for common
           handbook terms such as leave, attendance, benefits, resignation, etc.
        """
        if not self.is_ready:
            raise RAGNotReadyError(
                "The vector index is not ready yet. Call initialize() at startup first."
            )

        question = (question or "").strip()
        if not question:
            return None

        expanded_question = expand_query_for_retrieval(question)
        query_variants = [question]
        if expanded_question != question:
            query_variants.append(expanded_question)

        vector_matches: List[Tuple[float, str]] = []
        seen_chunks = set()
        for query in query_variants:
            for score, chunk in self._top_matches(query, top_k=self.top_k):
                if chunk in seen_chunks:
                    continue
                vector_matches.append((score, chunk))
                seen_chunks.add(chunk)

        vector_matches.sort(key=lambda item: item[0], reverse=True)

        if vector_matches:
            best_score = vector_matches[0][0]
            logger.info(
                "Top vector score=%.4f (threshold=%.4f) for question=%r expanded=%r",
                best_score,
                self.similarity_threshold,
                question,
                expanded_question,
            )

            relevant_chunks = [
                chunk for score, chunk in vector_matches if score >= self.similarity_threshold
            ]
            if relevant_chunks:
                relevant_chunks = _dedupe_overlapping_chunks(relevant_chunks)
                return "\n\n---\n\n".join(relevant_chunks[: self.top_k])

            logger.info("Vector scores below threshold; checking keyword fallback.")
        else:
            logger.info("No vector matches returned; checking keyword fallback.")

        if self.enable_keyword_fallback:
            keyword_matches = self._keyword_matches(expanded_question)
            if keyword_matches:
                logger.info(
                    "Keyword fallback returned %d chunk(s); best overlap=%d for question=%r",
                    len(keyword_matches),
                    keyword_matches[0][0],
                    question,
                )
                return "\n\n---\n\n".join(chunk for _, chunk in keyword_matches)

        logger.info("No relevant handbook context found after vector and keyword retrieval.")
        return None


# Single shared instance used by the FastAPI app. Built once via
# rag_system.initialize() in main.py's startup/lifespan handler.
rag_system = RAGSystem()