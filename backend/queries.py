"""
queries.py
----------
"Report to admin" / unanswered-queries module for the Employee Handbook
RAG Assistant.

Provides:
    - SQLite-backed storage (table: queries -> id, question, answer,
      asked_by, survey_title, chat_id, status, date, answered_at) in its
      own database file (unanswered.db by default).
    - POST   /queries      -> any logged-in user reports a chatbot answer
                               that needs a human follow-up (called from
                               the "Report to admin" button in index.html).
    - GET    /queries      -> list queries, optionally filtered by status
                               (called by admin_panel.html's "Unanswered
                               Queries" tab). Requires the "queries" tab.
    - PATCH  /queries/{id} -> submit/update an answer and status (called
                               from the "Respond to Query" panel). Requires
                               the "queries" tab.
    - DELETE /queries/{id} -> delete a query. Requires the "queries" tab.

Access to GET/PATCH/DELETE is restricted to accounts whose `tabs` include
"queries" (i.e. admins, and hr-employees who've been granted that tab),
mirroring how admin_panel.html itself decides which tabs to show.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from auth import get_current_user_row
from rag import rag_manager

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DB_PATH = os.environ.get("QUERIES_DB_PATH", "unanswered.db")

VALID_STATUSES = {"pending", "answered"}


# --------------------------------------------------------------------------
# Database setup
# --------------------------------------------------------------------------

def init_db() -> None:
    """Create the queries table if it doesn't already exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS queries (
                id            TEXT PRIMARY KEY,
                question      TEXT NOT NULL,
                answer        TEXT,
                asked_by      TEXT,
                survey_title  TEXT,
                department    TEXT NOT NULL DEFAULT 'hr',
                chat_id       TEXT,
                status        TEXT NOT NULL DEFAULT 'pending',
                date          TEXT NOT NULL,
                answered_at   TEXT
            )
            """
        )
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(queries)")}
        if "department" not in existing_cols:
            conn.execute("ALTER TABLE queries ADD COLUMN department TEXT NOT NULL DEFAULT 'hr'")
        conn.commit()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class ReportQueryRequest(BaseModel):
    question: str
    answer: Optional[str] = None
    surveyTitle: Optional[str] = None
    department: str = "hr"
    chatId: Optional[str] = None


class UpdateQueryRequest(BaseModel):
    answer: Optional[str] = None
    status: Optional[str] = None


class QueryOut(BaseModel):
    id: str
    question: str
    answer: Optional[str] = None
    askedBy: Optional[str] = None
    surveyTitle: Optional[str] = None
    department: str = "hr"
    chatId: Optional[str] = None
    status: str
    date: str
    answeredAt: Optional[str] = None


def _row_to_query_out(row: sqlite3.Row) -> QueryOut:
    return QueryOut(
        id=row["id"],
        question=row["question"],
        answer=row["answer"],
        askedBy=row["asked_by"],
        surveyTitle=row["survey_title"],
        department=row["department"] if "department" in row.keys() else "hr",
        chatId=row["chat_id"],
        status=row["status"],
        date=row["date"],
        answeredAt=row["answered_at"],
    )

def can_access_department(user, query_department):
    role = (user["role"] or "").lower()

    if role == "admin":
        return True

    if role == "hr-employee":
        return query_department.lower() == "hr"

    if role == "finance":
        return query_department.lower() == "finance"

    return False

# --------------------------------------------------------------------------
# Access control
# --------------------------------------------------------------------------

def require_queries_tab(user: sqlite3.Row = Depends(get_current_user_row)) -> sqlite3.Row:
    """Admins always pass; anyone else needs "queries" in their granted
    tabs (matches how admin_panel.html decides whether to show the tab)."""
    import json

    if user["role"] == "admin":
        return user
    tabs = json.loads(user["tabs"] or "[]")
    if "queries" not in tabs:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to the Unanswered Queries tab.",
        )
    return user


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

router = APIRouter(prefix="/queries", tags=["Queries"])


@router.post("", response_model=QueryOut, status_code=status.HTTP_201_CREATED)
async def report_query(
    request: ReportQueryRequest,
    user: sqlite3.Row = Depends(get_current_user_row),
) -> QueryOut:
    """
    Report a chatbot answer for human follow-up. Called by the
    "Report to admin" button under an assistant message in index.html.
    Any logged-in user can call this; the question always lands as
    'pending' for an admin/hr-employee to answer.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="question is required.")

    row = {
        "id": str(uuid.uuid4()),
        "question": question,
        "answer": (request.answer or "").strip() or None,
        "asked_by": user["email"],
        "survey_title": request.surveyTitle or "General",
        "department": (request.department or "hr").strip().lower(),
        "chat_id": request.chatId,
        "status": "pending",
        "date": datetime.now(timezone.utc).isoformat(),
        "answered_at": None,
    }

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO queries (id, question, answer, asked_by, survey_title, department, chat_id, status, date, answered_at)
            VALUES (:id, :question, :answer, :asked_by, :survey_title, :department, :chat_id, :status, :date, :answered_at)
            """,
            row,
        )
        conn.commit()

    return QueryOut(
        id=row["id"], question=row["question"], answer=row["answer"],
        askedBy=row["asked_by"], surveyTitle=row["survey_title"], department=row["department"],
        chatId=row["chat_id"], status=row["status"], date=row["date"], answeredAt=row["answered_at"],
    )


@router.get("", response_model=List[QueryOut])
async def list_queries(
    status_filter: Optional[str] = None,
    department: Optional[str] = None,
    user: sqlite3.Row = Depends(require_queries_tab),
) -> List[QueryOut]:
    """List reported queries, newest first.
    Optional ?status_filter=pending|answered and/or ?department=hr|marketing|..."""
    clauses = []
    params: List[str] = []
    if status_filter:
        if status_filter not in VALID_STATUSES:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid status filter.")
        clauses.append("status = ?")
        params.append(status_filter)
    if department:
        clauses.append("department = ?")
        params.append(department.strip().lower())

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with get_connection() as conn:
        rows = conn.execute(f"SELECT * FROM queries {where} ORDER BY date DESC", params).fetchall()

    if user["role"] == "admin":
        filtered_rows = rows

    elif user["role"] == "hr-employee":
        filtered_rows = [
            r for r in rows
            if (r["department"] or "").lower() == "hr"
        ]

    elif user["role"] == "finance":
        filtered_rows = [
            r for r in rows
            if (r["department"] or "").lower() == "finance"
        ]

    else:
        filtered_rows = []
    return [_row_to_query_out(r) for r in filtered_rows]


@router.patch("/{query_id}", response_model=QueryOut)
async def update_query(
    query_id: str,
    request: UpdateQueryRequest,
    user: sqlite3.Row = Depends(require_queries_tab),
) -> QueryOut:
    """
    Submit or edit an answer for a reported query. Called from the
    "Respond to Query" panel in admin_panel.html.
    """
    if request.status is not None and request.status not in VALID_STATUSES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid status.")

    with get_connection() as conn:
        existing = conn.execute("SELECT * FROM queries WHERE id = ?", (query_id,)).fetchone()
        if existing is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Query not found.")

        new_answer = request.answer if request.answer is not None else existing["answer"]
        new_status = request.status if request.status is not None else existing["status"]

        new_answered_at = existing["answered_at"]
        if new_status == "answered" and existing["status"] != "answered":
            new_answered_at = datetime.now(timezone.utc).isoformat()
        elif new_status == "pending":
            new_answered_at = None

        conn.execute(
            "UPDATE queries SET answer = ?, status = ?, answered_at = ? WHERE id = ?",
            (new_answer, new_status, new_answered_at, query_id),
        )
        conn.commit()

        updated = conn.execute("SELECT * FROM queries WHERE id = ?", (query_id,)).fetchone()
        if not can_access_department(
            user,
            existing["department"]
        ):
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to answer this query."
            )
    # The moment a query newly becomes "answered" (with an actual answer),
    # fold it straight into that department's live RAG index — this is the
    # "train the bot" step, so the very next similar question in this
    # department gets answered without needing another human reply.
    became_answered = new_status == "answered" and existing["status"] != "answered"
    if became_answered and new_answer:
        try:
            rag_manager.train(updated["department"], updated["question"], new_answer)
        except Exception as exc:  # noqa: BLE001 - training failure must not break the save
            import logging
            logging.getLogger("ai_assistant.queries").error(
                "Failed to train department '%s' with answered query %s: %s",
                updated["department"], query_id, exc,
            )

    return _row_to_query_out(updated)


@router.delete("/{query_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_query(
    query_id: str,
    user: sqlite3.Row = Depends(require_queries_tab),
) -> None:

    with get_connection() as conn:

        row = conn.execute(
            "SELECT * FROM queries WHERE id = ?",
            (query_id,)
        ).fetchone()

        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Query not found."
            )

        if not can_access_department(
            user,
            row["department"]
        ):
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to manage this query."
            )

        result = conn.execute(
            "DELETE FROM queries WHERE id = ?",
            (query_id,)
        )

        conn.commit()