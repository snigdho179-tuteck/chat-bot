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

import json
import logging
import os
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

    # Root folder containing one subfolder per department, e.g.
    #   data/hr/Employee_Handbook.pdf
    #   data/marketing/Marketing_Playbook.pdf
    #   data/finance/Finance_Policy.pdf
    #   data/sales/Sales_Handbook.pdf
    # Every PDF placed in a department's folder (by the admin, via File
    # Management) is indexed for that department. Override via .env if
    # your data lives elsewhere.
    DATA_ROOT: str = os.environ.get("RAG_DATA_ROOT", "../data")

    # Which department tabs exist. The frontend's chat tab list and the
    # admin's File Management tab should match these slugs. "hr" ships
    # first; add more here (or via .env, comma-separated) as new
    # department documents come online.
    DEPARTMENTS: List[str] = [
        d.strip() for d in os.environ.get("RAG_DEPARTMENTS", "hr,marketing,finance,sales").split(",")
        if d.strip()
    ]
    DEFAULT_DEPARTMENT: str = os.environ.get("RAG_DEFAULT_DEPARTMENT", "hr")

    # Backward-compat: the original single-file layout. If a department's
    # own folder (DATA_ROOT/<department>/) doesn't exist yet, the "hr"
    # department falls back to this single PDF so existing deployments
    # keep working unchanged.
    LEGACY_HR_PDF_PATH: str = os.environ.get("RAG_PDF_PATH", "../data/Employee_Handbook.pdf")

    # Larger than the original 500/100. Small chunk sizes were splitting
    # enumerated lists (e.g. the 6 types of leave) across two overlapping
    # chunks, so the model ended up seeing the same list twice in two
    # different truncated forms and would drop items when summarizing.
    CHUNK_SIZE: int = int(os.environ.get("RAG_CHUNK_SIZE", "900"))
    CHUNK_OVERLAP: int = int(os.environ.get("RAG_CHUNK_OVERLAP", "150"))

    # "semantic" (default): sentences are grouped by embedding similarity,
    # so a chunk boundary falls where the *topic* actually shifts instead
    # of at an arbitrary character offset. This keeps a policy's sentences
    # together even when it runs long, and splits earlier when the topic
    # changes even inside what would've been one fixed-size chunk.
    # "fixed": the original sliding-window character chunker, kept for
    # easy rollback / comparison.
    CHUNKING_STRATEGY: str = os.environ.get("RAG_CHUNKING_STRATEGY", "semantic").strip().lower()

    # How aggressively semantic chunking splits. This is a percentile over
    # the distribution of sentence-to-sentence semantic distances in a
    # document: a gap bigger than this percentile of gaps is treated as a
    # topic change and becomes a chunk boundary. Higher = fewer, larger
    # chunks (only splits on the starkest topic shifts); lower = more,
    # smaller chunks.
    SEMANTIC_BREAKPOINT_PERCENTILE: float = float(
        os.environ.get("RAG_SEMANTIC_BREAKPOINT_PERCENTILE", "90")
    )

    # Safety valve: even within one semantic "topic", force a split if the
    # accumulated chunk would otherwise grow past this many characters, so
    # a long uniform-topic section can't produce one giant chunk.
    SEMANTIC_MAX_CHUNK_SIZE: int = int(
        os.environ.get("RAG_SEMANTIC_MAX_CHUNK_SIZE", str(int(os.environ.get("RAG_CHUNK_SIZE", "900")) * 2))
    )

    # A group of sentences smaller than this (characters) gets merged into
    # a neighboring chunk rather than kept as its own tiny fragment.
    SEMANTIC_MIN_CHUNK_SIZE: int = int(os.environ.get("RAG_SEMANTIC_MIN_CHUNK_SIZE", "200"))

    EMBEDDING_MODEL_NAME: str = os.environ.get(
        "RAG_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"
    )

    # nomic-embed-text-v1.5 is a Matryoshka model: it natively outputs 768-dim
    # vectors but can be truncated to smaller sizes (512/256/128/64). We keep
    # the full 768 dims. FAISS's index dimension is still taken from the
    # actual embedding shape at build time, so this just tells the model
    # (via SentenceTransformer's truncate_dim) what to output.
    EMBEDDING_DIMENSION: int = int(os.environ.get("RAG_EMBEDDING_DIMENSION", "768"))

    # nomic-embed-text-v1.5 requires running its bundled modeling code.
    EMBEDDING_TRUST_REMOTE_CODE: bool = os.environ.get(
        "RAG_EMBEDDING_TRUST_REMOTE_CODE", "true"
    ).lower() != "false"

    # nomic's embed models are instruction-prefixed: text must be prefixed
    # with a task name ("search_document: ", "search_query: ", "clustering: ")
    # depending on how it's being embedded, or retrieval quality drops
    # sharply. These default to empty (no-op) for non-nomic models, and to
    # nomic's prefixes when EMBEDDING_MODEL_NAME is a nomic model, so this
    # never needs manual tuning when swapping the model via .env.
    _IS_NOMIC_EMBEDDING_MODEL: bool = "nomic" in EMBEDDING_MODEL_NAME.lower()
    EMBEDDING_DOCUMENT_PREFIX: str = os.environ.get(
        "RAG_EMBEDDING_DOCUMENT_PREFIX",
        "search_document: " if _IS_NOMIC_EMBEDDING_MODEL else "",
    )
    EMBEDDING_QUERY_PREFIX: str = os.environ.get(
        "RAG_EMBEDDING_QUERY_PREFIX",
        "search_query: " if _IS_NOMIC_EMBEDDING_MODEL else "",
    )
    EMBEDDING_CLUSTERING_PREFIX: str = os.environ.get(
        "RAG_EMBEDDING_CLUSTERING_PREFIX",
        "clustering: " if _IS_NOMIC_EMBEDDING_MODEL else "",
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


class RAGDepartmentNotFoundError(RAGError):
    """Raised when a chat/report request names a department that isn't configured."""


# --------------------------------------------------------------------------
# Text extraction & chunking
# --------------------------------------------------------------------------

_NUMERIC_CELL_RE = re.compile(r"^[\d,./%\-\s]*\d[\d,./%\-\s]*$")


def _looks_numeric(cell: str) -> bool:
    """True for cells that are purely numbers/currency-ish values (e.g.
    '4300', '10000 +', '1,200', '50%'), used to tell a header/label row
    apart from the first row of actual data."""
    cell = cell.strip()
    if not cell:
        return False
    return bool(_NUMERIC_CELL_RE.match(cell))


def _format_table_rows(rows: List[List[Optional[str]]]) -> str:
    """
    Turn PyMuPDF's table.extract() output (a list of rows, each a list of
    cell strings/None) into a markdown-style table, preserving row/column
    structure — instead of the old approach of dumping all the table's text
    via page.get_text(clip=bbox), which throws away row/column association
    entirely and interleaves every cell's text in raw reading order.

    Two things this handles that a naive "row 0 = header" approach doesn't:

    1. MULTI-ROW HEADERS. Tables like the Inland Travel Allowance grid have
       a grouping header ("X Class Cities" / "Y Class cities" / "Z Class
       cities") sitting ABOVE the real column labels ("Lodging", "Boarding",
       "TOTAL", "Out of pocket expenses"). Treating only row 0 as the header
       (the previous implementation) throws away which city-class each
       column group belongs to, so a Z-class total can get reported as if
       it were the X-class total for the same grade. All leading rows that
       contain no numeric cells are treated as header rows and combined
       into one compound label per column (grouping label first, then the
       specific column label), so every column keeps its full "which
       city-class / which measure" identity.

    2. MERGED/SPANNED CELLS. A cell that visually spans multiple rows (a
       shared grade label, a shared numeric value across two grade rows,
       etc.) comes back from PyMuPDF as "" on every row except the one
       where the label/value is actually printed. Each column is
       forward-filled independently, top-to-bottom, from the nearest
       non-empty cell above it in that SAME column. This only ever reuses
       a value for the row(s) directly below the row that actually has it —
       exactly what a rowspan means — and never invents a relationship
       between two rows that aren't actually part of the same merged cell.
       (An earlier version tried to detect "split across two physical
       rows" by merging whole rows whenever their non-empty columns didn't
       overlap; that guess is what glued unrelated records' numbers
       together — e.g. attributing Grade 1A's X-class total to its Z-class
       row. Per-column forward-fill replaces that guess entirely.)

    Purely empty rows (no non-empty cell anywhere) are dropped.
    """
    cleaned_rows: List[List[str]] = []
    for row in rows:
        cleaned_rows.append([(cell or "").replace("\n", " ").strip() for cell in row])

    if not cleaned_rows:
        return ""

    num_cols = max(len(r) for r in cleaned_rows)
    for row in cleaned_rows:
        row.extend([""] * (num_cols - len(row)))

    cleaned_rows = [row for row in cleaned_rows if any(cell for cell in row)]
    if not cleaned_rows:
        return ""

    # PyMuPDF's grid detector can over-segment a table with thin/invisible
    # gridlines and heavy row/col-spanning into extra phantom columns.
    # Two passes clean this up, both using only within-row evidence so
    # nothing gets fabricated:
    #
    # (a) Drop columns that are blank on every single row (header rows
    #     included) — these never carried a value in the first place.
    #
    # (b) Merge adjacent columns that are "mutually exclusive": if, on
    #     every row, at most one of two neighboring columns is non-empty,
    #     they're really one logical column whose word-wrapped text (or
    #     wrapped header label) landed in slightly different horizontal
    #     bins on different lines — e.g. "3T / AC" / "Chair" / "Car" each
    #     showing up as their own near-empty column across different rows
    #     of the same "CLASS OF TRAVEL" cell. Merging concatenates their
    #     text per row instead of leaving the real value scattered across
    #     several sparsely-populated columns.
    live_cols = [c for c in range(num_cols) if any(row[c] for row in cleaned_rows)]
    if live_cols and len(live_cols) < num_cols:
        cleaned_rows = [[row[c] for c in live_cols] for row in cleaned_rows]
        num_cols = len(live_cols)

    merged = True
    while merged and num_cols > 1:
        merged = False
        for c in range(num_cols - 1):
            if all(not (row[c] and row[c + 1]) for row in cleaned_rows):
                for row in cleaned_rows:
                    combined = " ".join(p for p in (row[c], row[c + 1]) if p)
                    row[c] = combined
                    del row[c + 1]
                num_cols -= 1
                merged = True
                break

    # --- Identify all leading header rows (not just row 0) -----------------
    # A row is still "header" as long as it has no numeric cells at all and
    # isn't the very last row in the table. Column labels that wrap onto
    # several lines (e.g. "Out of" / "pocket" / "expenses" as three separate
    # extracted rows) all land here too, so the cap is generous (8 rows) —
    # it's a safety valve against pathological tables with no data rows,
    # not a real limit on how many header lines a table can have.
    header_rows: List[List[str]] = [cleaned_rows[0]]
    idx = 1
    while (
        idx < len(cleaned_rows) - 1
        and len(header_rows) < 8
        and not any(_looks_numeric(c) for c in cleaned_rows[idx])
        and any(c for c in cleaned_rows[idx])
    ):
        header_rows.append(cleaned_rows[idx])
        idx += 1
    data_rows = cleaned_rows[idx:]

    # A header row whose only non-empty cell(s) all contain the exact same
    # text is a full-width caption (e.g. the table's overall title sitting
    # in its own row) rather than a per-column group label. Pull those out
    # so they don't get repeated inside every single column's label.
    caption_lines: List[str] = []
    group_header_rows: List[List[str]] = []
    for hrow in header_rows:
        distinct = {c for c in hrow if c}
        if len(distinct) == 1:
            caption_lines.append(next(iter(distinct)))
        else:
            group_header_rows.append(hrow)

    # Grouping header rows (e.g. "X Class Cities") only print their label
    # once, in the leftmost cell of the group of columns they cover, so
    # carry it rightward across the blanks it actually spans.
    for hrow in group_header_rows:
        last = ""
        for c in range(num_cols):
            if hrow[c]:
                last = hrow[c]
            elif last:
                hrow[c] = last

    # Combine the (non-caption) header rows into one compound label per
    # column, e.g. "X Class Cities Out of pocket expenses". A plain space
    # join reads naturally whether the pieces came from a real grouping
    # level ("X Class Cities" + "Lodging") or from a column label that
    # simply wrapped onto several extracted rows ("Out of" / "pocket" /
    # "expenses") — no need to tell those two cases apart.
    combined_header: List[str] = []
    for c in range(num_cols):
        pieces: List[str] = []
        seen = set()
        for hrow in group_header_rows:
            piece = hrow[c].strip()
            if piece and piece not in seen:
                pieces.append(piece)
                seen.add(piece)
        combined_header.append(" ".join(pieces) if pieces else f"Column {c + 1}")

    caption = " ".join(caption_lines).strip()

    # --- Forward-fill merged/spanned data cells, per column -----------------
    # Only ever carries a value DOWN into the row(s) immediately below the
    # row that actually has it — the correct semantics for a rowspan cell,
    # and never crosses between two rows that aren't actually merged.
    filled_rows: List[List[str]] = []
    last_seen = [""] * num_cols
    for row in data_rows:
        new_row = list(row)
        for c in range(num_cols):
            if new_row[c]:
                last_seen[c] = new_row[c]
            elif last_seen[c]:
                new_row[c] = last_seen[c]
        # A multi-line cell (e.g. a long "CLASS OF TRAVEL" description that
        # wraps across several lines) makes PyMuPDF emit one extra physical
        # row per wrapped line. After forward-fill those extra rows become
        # exact duplicates of the row above (same record, nothing new), so
        # collapsing consecutive duplicates removes the bloat without
        # losing any row that actually carries new information — this is
        # also what was making responses slow, since the same row could
        # otherwise repeat 5-8x in the context sent to the model.
        if filled_rows and new_row == filled_rows[-1]:
            continue
        # A row's own label can itself wrap onto a second physical line
        # (e.g. "Grade" then "10" as two separate extracted rows for
        # "Grade 10"). If this row is identical to the previous one in
        # every column except exactly one, and that one column is
        # non-empty in both, it's that same wrap pattern — concatenate the
        # differing column instead of keeping two rows for one record.
        if filled_rows:
            prev_row = filled_rows[-1]
            diff_cols = [c for c in range(num_cols) if new_row[c] != prev_row[c]]
            if len(diff_cols) == 1 and new_row[diff_cols[0]] and prev_row[diff_cols[0]]:
                c = diff_cols[0]
                prev_row[c] = f"{prev_row[c]} {new_row[c]}"
                continue
        filled_rows.append(new_row)

    lines = [f"{caption}\n"] if caption else []
    lines.append("| " + " | ".join(combined_header) + " |")
    lines.append("| " + " | ".join(["---"] * num_cols) + " |")
    for row in filled_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


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

                    try:
                        rows = table.extract()
                    except Exception:  # noqa: BLE001 - extraction is best-effort
                        rows = None

                    structured_text = _format_table_rows(rows) if rows else ""
                    # Prefer the structured (row/column-preserving) rendering;
                    # only fall back to the flat clipped text if structured
                    # extraction produced nothing usable.
                    body_text = structured_text or clip_text

                    table_chunk = f"[Section: {heading}]\n{body_text}" if heading else body_text
                    segments.append(("table", table_chunk))
                    # Remove the table's own text from the prose remainder
                    # so it isn't also chunked (and duplicated) as prose.
                    # Always matched against clip_text (the raw page text),
                    # regardless of which rendering was kept above.
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
    segments: List[Tuple[str, str]],
    chunk_size: int,
    overlap: int,
    embedding_model: Optional[SentenceTransformer] = None,
    strategy: Optional[str] = None,
) -> List[str]:
    """
    Turn (kind, text) segments into final chunks: "table" segments are
    always kept whole, as a single chunk each, regardless of chunk_size.

    "prose" segments go through one of two chunkers:
      - "semantic" (default, requires embedding_model): sentences are
        grouped by embedding similarity so boundaries fall at topic
        shifts rather than at a fixed character offset.
      - "fixed": the original overlapping sliding-window chunker.

    If strategy == "semantic" but no embedding_model is supplied, this
    falls back to "fixed" rather than failing, so callers that haven't
    wired up a model yet still get chunks.
    """
    strategy = (strategy or rag_settings.CHUNKING_STRATEGY).strip().lower()
    use_semantic = strategy == "semantic" and embedding_model is not None

    chunks: List[str] = []
    for kind, text in segments:
        if kind == "table":
            text = text.strip()
            if text:
                chunks.append(text)
        elif use_semantic:
            chunks.extend(
                semantic_chunk_text(
                    text,
                    embedding_model,
                    max_chunk_size=rag_settings.SEMANTIC_MAX_CHUNK_SIZE,
                    min_chunk_size=rag_settings.SEMANTIC_MIN_CHUNK_SIZE,
                    breakpoint_percentile=rag_settings.SEMANTIC_BREAKPOINT_PERCENTILE,
                    embed_prefix=rag_settings.EMBEDDING_CLUSTERING_PREFIX,
                )
            )
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


_SENTENCE_SPLIT_RE = re.compile(
    r"""
    (?<=[.!?])       # split after sentence-ending punctuation
    \s+              # followed by whitespace
    (?=[A-Z0-9"'\(\[])  # and the next sentence looks like it starts one
    """,
    re.VERBOSE,
)


def split_into_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter (no extra NLP dependency). Also treats
    blank lines / bullet starts as boundaries, since handbook prose is
    full of short list items that don't end in punctuation the regex
    above would catch.
    """
    text = text.strip()
    if not text:
        return []

    # First split on blank lines and bullet-like line starts, then run the
    # punctuation-based splitter on what's left of each piece.
    rough_pieces = re.split(r"\n\s*\n|\n(?=[•\-\*]\s|\d+[\.\)]\s)", text)

    sentences: List[str] = []
    for piece in rough_pieces:
        piece = piece.strip()
        if not piece:
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(piece):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)

    return sentences if sentences else [text]


def semantic_chunk_text(
    text: str,
    embedding_model: SentenceTransformer,
    max_chunk_size: int,
    min_chunk_size: int,
    breakpoint_percentile: float,
    embed_prefix: str = "",
) -> List[str]:
    """
    Split text into chunks along semantic (topic) boundaries instead of
    fixed character offsets.

    How it works:
      1. Split the text into sentences.
      2. Embed every sentence.
      3. Compute the cosine distance between each consecutive pair of
         sentence embeddings — a big jump means the topic just shifted.
      4. Treat any distance above the given percentile of all the
         document's own distances as a chunk boundary.
      5. Concatenate sentences between boundaries into chunks, further
         splitting anything that grows past max_chunk_size and merging
         anything smaller than min_chunk_size into its neighbor.

    This means a chunk boundary reflects an actual change in subject
    matter, so a policy that runs long stays in one chunk, while a short
    section can still get its own chunk if what follows it is unrelated —
    instead of both being cut at an arbitrary character count.
    """
    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        stripped = text.strip()
        return [stripped] if stripped else []

    embeddings = embedding_model.encode(
        [embed_prefix + s for s in sentences] if embed_prefix else sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Cosine distance between consecutive sentences (embeddings are
    # normalized, so dot product == cosine similarity).
    sims = np.einsum("ij,ij->i", embeddings[:-1], embeddings[1:])
    distances = 1.0 - sims

    if len(distances) == 0:
        stripped = text.strip()
        return [stripped] if stripped else []

    threshold = float(np.percentile(distances, breakpoint_percentile))

    # Group sentences into chunks: start a new chunk whenever the distance
    # to the next sentence exceeds the threshold.
    groups: List[List[str]] = [[sentences[0]]]
    for i, distance in enumerate(distances):
        if distance > threshold:
            groups.append([sentences[i + 1]])
        else:
            groups[-1].append(sentences[i + 1])

    # Merge tiny groups into a neighbor so we don't emit fragment chunks.
    merged: List[List[str]] = []
    for group in groups:
        group_len = sum(len(s) for s in group)
        if merged and group_len < min_chunk_size:
            merged[-1].extend(group)
        else:
            merged.append(list(group))

    # Split any still-oversized group on sentence boundaries so no single
    # chunk blows past max_chunk_size.
    chunks: List[str] = []
    for group in merged:
        current: List[str] = []
        current_len = 0
        for sentence in group:
            if current and current_len + len(sentence) + 1 > max_chunk_size:
                chunks.append(" ".join(current).strip())
                current = []
                current_len = 0
            current.append(sentence)
            current_len += len(sentence) + 1
        if current:
            chunks.append(" ".join(current).strip())

    return [c for c in chunks if c]


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


def _format_qa_chunk(question: str, answer: str) -> str:
    """
    Format an answered query as a single retrievable chunk. Keeping the
    question text in the chunk (not just the answer) is what lets future
    rephrasings of the same question still match it semantically.
    """
    return f"[Answered question]\nQ: {question}\nA: {answer}"


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
        department: str,
        pdf_dir: str,
        learned_qa_path: str,
        legacy_pdf_path: Optional[str] = None,
        chunk_size: int = rag_settings.CHUNK_SIZE,
        chunk_overlap: int = rag_settings.CHUNK_OVERLAP,
        embedding_model_name: str = rag_settings.EMBEDDING_MODEL_NAME,
        embedding_dimension: int = rag_settings.EMBEDDING_DIMENSION,
        top_k: int = rag_settings.TOP_K,
        similarity_threshold: float = rag_settings.SIMILARITY_THRESHOLD,
    ) -> None:
        self.department = department
        self.pdf_dir = pdf_dir
        self.learned_qa_path = learned_qa_path
        # Only "hr" uses this, and only if pdf_dir has nothing in it yet —
        # keeps pre-existing single-PDF deployments working unmodified.
        self.legacy_pdf_path = legacy_pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name
        self.embedding_dimension = embedding_dimension
        self.embedding_trust_remote_code = rag_settings.EMBEDDING_TRUST_REMOTE_CODE
        self.embedding_document_prefix = rag_settings.EMBEDDING_DOCUMENT_PREFIX
        self.embedding_query_prefix = rag_settings.EMBEDDING_QUERY_PREFIX
        self.embedding_clustering_prefix = rag_settings.EMBEDDING_CLUSTERING_PREFIX
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.enable_keyword_fallback = rag_settings.ENABLE_KEYWORD_FALLBACK
        self.keyword_top_k = rag_settings.KEYWORD_TOP_K
        self.min_keyword_overlap = rag_settings.MIN_KEYWORD_OVERLAP

        self._embedding_model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.Index] = None
        self._chunks: List[str] = []
        # Which chunk indices came from answered-query training rather than
        # an uploaded document — not currently used for special-casing
        # retrieval, but kept for introspection/debugging.
        self._qa_chunk_count: int = 0

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

    @property
    def qa_chunk_count(self) -> int:
        """How many of the current chunks came from answered queries."""
        return self._qa_chunk_count

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
            pdf_paths = self._discover_pdfs()
            if not pdf_paths:
                raise RAGInitializationError(
                    f"No PDFs found for department '{self.department}' "
                    f"(looked in '{self.pdf_dir}'). Upload a document via "
                    "File Management for this department first."
                )

            logger.info("Loading %d document(s) for department '%s'...", len(pdf_paths), self.department)
            segments: List[Tuple[str, str]] = []
            for pdf_path in pdf_paths:
                segments.extend(extract_segments_from_pdf(pdf_path))

            # Built before chunking (rather than after) because semantic
            # chunking needs the embedding model to score sentence-to-
            # sentence similarity while it decides where chunk boundaries
            # go. It's then reused below to embed the final chunks too.
            model = SentenceTransformer(
                self.embedding_model_name,
                trust_remote_code=self.embedding_trust_remote_code,
                truncate_dim=self.embedding_dimension,
            )

            logger.info(
                "Building vector index for department '%s' (chunking_strategy=%s)...",
                self.department,
                rag_settings.CHUNKING_STRATEGY,
            )
            chunks = chunk_segments(
                segments, self.chunk_size, self.chunk_overlap, embedding_model=model
            )

            # Fold in every previously-answered query for this department so
            # they get re-indexed on every restart, not just kept in memory.
            learned_chunks = [
                _format_qa_chunk(entry["question"], entry["answer"])
                for entry in self._load_learned_qa()
            ]
            chunks.extend(learned_chunks)

            if not chunks:
                raise RAGInitializationError(
                    f"Department '{self.department}' produced zero chunks — "
                    "check CHUNK_SIZE/CHUNK_OVERLAP or the source documents."
                )

            embeddings = model.encode(
                [self.embedding_document_prefix + c for c in chunks] if self.embedding_document_prefix else chunks,
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
            self._qa_chunk_count = len(learned_chunks)
            self._status = "loaded"
            self._last_error = None

            logger.info(
                "Vector index loaded for department '%s'. (%d chunks total, %d from answered queries, dim=%d)",
                self.department,
                len(chunks),
                len(learned_chunks),
                dimension,
            )

        except RAGInitializationError as exc:
            self._status = "error"
            self._last_error = str(exc)
            logger.error("RAG initialization failed for department '%s': %s", self.department, exc)
            raise
        except Exception as exc:  # noqa: BLE001 - surface anything unexpected clearly
            self._status = "error"
            self._last_error = str(exc)
            logger.exception("Unexpected error during RAG initialization for department '%s'", self.department)
            raise RAGInitializationError(f"Unexpected error building the vector index: {exc}") from exc

    def reload(self) -> None:
        """
        Force a full rebuild from whatever PDFs currently sit in pdf_dir —
        unlike initialize(), this does NOT no-op if already loaded. Called
        right after the admin uploads or deletes a department document so
        the change is live immediately, no restart required.
        """
        self._embedding_model = None
        self._index = None
        self._chunks = []
        self._qa_chunk_count = 0
        self._status = "not_loaded"
        self._last_error = None
        try:
            self.initialize()
        except RAGInitializationError:
            # No PDFs left (e.g. the last one was just deleted) — leave the
            # department registered but empty/"error" rather than crashing
            # the request; initialize() already recorded the reason.
            pass

    def list_documents(self) -> List[Dict[str, object]]:
        """Every PDF currently on disk for this department — used by the
        admin's File Management table."""
        dir_path = Path(self.pdf_dir)
        if not dir_path.is_dir():
            return []
        docs = []
        for p in sorted(dir_path.glob("*.pdf")):
            stat = p.stat()
            docs.append({
                "filename": p.name,
                "size_bytes": stat.st_size,
                "uploaded_at": stat.st_mtime,
            })
        return docs

    def delete_document(self, filename: str) -> bool:
        """Remove one uploaded PDF and re-index from what's left. Returns
        False if the file didn't exist."""
        target = Path(self.pdf_dir) / filename
        # Guard against path traversal — filename must resolve to a direct
        # child of this department's own folder.
        if target.parent.resolve() != Path(self.pdf_dir).resolve() or not target.is_file():
            return False
        target.unlink()
        self.reload()
        return True

    def _discover_pdfs(self) -> List[str]:
        """PDFs for this department: every *.pdf in pdf_dir, sorted for a
        stable chunk order; falls back to the single legacy file (hr only)
        if the department folder doesn't exist or is empty yet."""
        dir_path = Path(self.pdf_dir)
        if dir_path.is_dir():
            found = sorted(str(p) for p in dir_path.glob("*.pdf"))
            if found:
                return found

        if self.legacy_pdf_path and Path(self.legacy_pdf_path).is_file():
            return [self.legacy_pdf_path]

        return []

    # ---- learning from answered queries ----

    def _load_learned_qa(self) -> List[Dict[str, str]]:
        path = Path(self.learned_qa_path)
        if not path.is_file():
            return []
        entries: List[Dict[str, str]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed line in %s", path)
                    continue
                if entry.get("question") and entry.get("answer"):
                    entries.append(entry)
        return entries

    def add_qa_pair(self, question: str, answer: str) -> None:
        """
        "Train" this department's bot with a newly-answered query: embed
        the Q&A pair, append it to the live FAISS index immediately (so the
        very next question can match it, no restart needed), and persist
        it to disk so it survives restarts and gets folded back in by
        initialize().
        """
        question = (question or "").strip()
        answer = (answer or "").strip()
        if not question or not answer:
            return

        if not self.is_ready:
            # The index isn't built yet (e.g. this department has no PDF
            # yet) — still persist the answer so it's picked up the moment
            # the department is initialized.
            self._append_learned_qa(question, answer)
            return

        chunk = _format_qa_chunk(question, answer)
        assert self._embedding_model is not None and self._index is not None

        embedding = self._embedding_model.encode(
            [self.embedding_document_prefix + chunk] if self.embedding_document_prefix else [chunk],
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        ).astype("float32")

        self._index.add(embedding)
        self._chunks.append(chunk)
        self._qa_chunk_count += 1
        self._append_learned_qa(question, answer)

        logger.info(
            "Trained department '%s' with a newly-answered query (now %d QA chunks, %d total).",
            self.department, self._qa_chunk_count, len(self._chunks),
        )

    def _append_learned_qa(self, question: str, answer: str) -> None:
        path = Path(self.learned_qa_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"question": question, "answer": answer}) + "\n")

    # ---- retrieval ----

    def _embed_query(self, question: str) -> np.ndarray:
        assert self._embedding_model is not None  # guarded by is_ready check in search_context
        prefixed = self.embedding_query_prefix + question if self.embedding_query_prefix else question
        return self._embedding_model.encode(
            [prefixed],
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
                final_chunks = relevant_chunks[: self.top_k]
                logger.info(
                    "Sending %d chunk(s) to LLM for question=%r:\n%s",
                    len(final_chunks),
                    question,
                    "\n\n".join(f"--- CHUNK {i+1} ---\n{c}" for i, c in enumerate(final_chunks)),
                )
                return "\n\n---\n\n".join(final_chunks)

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
                fallback_chunks = [chunk for _, chunk in keyword_matches]
                logger.info(
                    "Sending %d chunk(s) to LLM via keyword fallback for question=%r:\n%s",
                    len(fallback_chunks),
                    question,
                    "\n\n".join(f"--- CHUNK {i+1} ---\n{c}" for i, c in enumerate(fallback_chunks)),
                )
                return "\n\n---\n\n".join(fallback_chunks)

        logger.info("No relevant handbook context found after vector and keyword retrieval.")
        return None


# --------------------------------------------------------------------------
# Multi-department manager
# --------------------------------------------------------------------------

class RAGManager:
    """
    Owns one RAGSystem per department/tab (hr, marketing, finance, sales, ...).
    Each department has its own folder of source PDFs under DATA_ROOT and
    its own learned-QA jsonl file, so uploading a document or answering a
    query for one department never touches another department's index.

    Usage:
        rag_manager.initialize_all()                       # once, at startup
        context = rag_manager.search_context(dept, q)       # per request
        rag_manager.train(dept, question, answer)           # per answered query
    """

    def __init__(self, departments: Optional[List[str]] = None) -> None:
        self.departments = departments or list(rag_settings.DEPARTMENTS)
        self.default_department = rag_settings.DEFAULT_DEPARTMENT
        self._systems: Dict[str, RAGSystem] = {}

        data_root = Path(rag_settings.DATA_ROOT)
        for dept in self.departments:
            self._systems[dept] = RAGSystem(
                department=dept,
                pdf_dir=str(data_root / dept),
                learned_qa_path=str(data_root / dept / "learned_qa.jsonl"),
                legacy_pdf_path=rag_settings.LEGACY_HR_PDF_PATH if dept == "hr" else None,
            )

    def initialize_all(self) -> None:
        """Build every department's index. A department whose document
        hasn't been uploaded yet fails independently (status='error') and
        does not block the others or app startup."""
        for dept, system in self._systems.items():
            try:
                system.initialize()
            except RAGInitializationError as exc:
                logger.error("Department '%s' not ready: %s", dept, exc)

    def get(self, department: Optional[str]) -> RAGSystem:
        dept = (department or self.default_department).strip().lower()
        system = self._systems.get(dept)
        if system is None:
            raise RAGDepartmentNotFoundError(
                f"Unknown department '{dept}'. Configured departments: {sorted(self._systems)}."
            )
        return system

    def ensure_department(self, department: str) -> RAGSystem:
        """
        Get a department's RAGSystem, creating (registering) it on the fly
        if this is a brand-new slug. This is what makes "upload a PDF for
        a department that doesn't exist yet" create the department/tab —
        no restart or .env edit required.
        """
        dept = (department or "").strip().lower()
        if not dept:
            raise RAGDepartmentNotFoundError("Department name cannot be empty.")
        if dept not in self._systems:
            data_root = Path(rag_settings.DATA_ROOT)
            self._systems[dept] = RAGSystem(
                department=dept,
                pdf_dir=str(data_root / dept),
                learned_qa_path=str(data_root / dept / "learned_qa.jsonl"),
            )
            self.departments.append(dept)
            logger.info("Registered new department '%s'.", dept)
        return self._systems[dept]

    def reload(self, department: str) -> RAGSystem:
        """Force a department to fully re-index from disk right now (after
        an upload or delete), rather than waiting for the next restart."""
        system = self.get(department)
        system.reload()
        return system

    def list_documents(self, department: str) -> List[Dict[str, object]]:
        return self.get(department).list_documents()

    def delete_document(self, department: str, filename: str) -> bool:
        return self.get(department).delete_document(filename)

    def remove_department(self, department: str) -> None:
        """Drop a department entirely (all its documents were removed).
        The chat UI simply stops showing a tab for it once it has zero
        ready documents; this only matters for the admin's own bookkeeping."""
        dept = (department or "").strip().lower()
        self._systems.pop(dept, None)
        self.departments = [d for d in self.departments if d != dept]

    def search_context(self, department: Optional[str], question: str) -> Optional[str]:
        return self.get(department).search_context(question)

    def train(self, department: Optional[str], question: str, answer: str) -> None:
        """Called once a reported query is answered — adds it to that
        department's live index immediately and persists it for reloads."""
        self.get(department).add_qa_pair(question, answer)

    def status_summary(self) -> Dict[str, str]:
        return {dept: system.status for dept, system in self._systems.items()}

    def list_departments_detail(self) -> List[Dict[str, object]]:
        """Slug + status + document/chunk counts for every registered
        department — the raw material for GET /departments."""
        details = []
        for dept, system in self._systems.items():
            details.append({
                "slug": dept,
                "status": system.status,
                "document_count": len(system.list_documents()),
                "chunk_count": system.chunk_count,
                "last_error": system.last_error,
            })
        return sorted(details, key=lambda d: d["slug"])

    @property
    def all_ready(self) -> bool:
        return all(system.is_ready for system in self._systems.values())


# Single shared instance used by the FastAPI app. Built once via
# rag_manager.initialize_all() in main.py's startup/lifespan handler.
rag_manager = RAGManager()

# Backward-compat alias: existing code that imports `rag_system` for the
# default department (hr) keeps working unchanged.
rag_system = rag_manager.get(rag_settings.DEFAULT_DEPARTMENT)