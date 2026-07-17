"""
departments.py
---------------
Admin-only "File Management" API for department documents.

An admin uploads a PDF for a department (e.g. "Engineering"). If that
department doesn't exist yet, it's created on the spot; the PDF is saved
into its folder and the department's RAG index is (re)built immediately
— no restart needed. The chatbot then picks up the new department tab
the next time it calls GET /departments, and any question asked in that
tab is answered strictly from that department's uploaded document(s),
via rag_manager.search_context(department, question) in backend.py.

Endpoints:
    GET    /departments                          -> list all departments + status (any logged-in user)
    GET    /departments/{slug}/documents          -> list a department's PDFs (admin only)
    POST   /departments/upload                    -> upload a PDF, creating the department if new (admin only)
    DELETE /departments/{slug}/documents/{name}   -> remove one PDF, re-index (admin only)
    DELETE /departments/{slug}                    -> remove a department entirely (admin only)
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from auth import get_current_user_row, require_admin
from rag import (
    RAGDepartmentNotFoundError,
    rag_manager,
    rag_settings,
)

logger = logging.getLogger("ai_assistant.departments")

router = APIRouter(prefix="/departments", tags=["Departments"])

_LABELS_PATH = Path(rag_settings.DATA_ROOT) / "department_labels.json"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    if not slug:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Department name must contain at least one letter or number.")
    return slug


def _load_labels() -> Dict[str, str]:
    if not _LABELS_PATH.is_file():
        return {}
    try:
        return json.loads(_LABELS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_labels(labels: Dict[str, str]) -> None:
    _LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LABELS_PATH.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def _safe_filename(name: str) -> str:
    """Strip any path components and keep just a clean file name."""
    name = Path(name).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name or "document.pdf"


@router.get("")
async def list_departments(_user=Depends(get_current_user_row)) -> List[Dict[str, object]]:
    """
    Every registered department with its status and document count. Any
    logged-in user can call this — it's what the chatbot uses to build
    its department tabs (filtered client-side to status == 'loaded').
    """
    labels = _load_labels()
    details = rag_manager.list_departments_detail()
    for d in details:
        d["label"] = labels.get(d["slug"]) or d["slug"].replace("-", " ").title()
    return details


@router.get("/{slug}/documents")
async def list_department_documents(slug: str, _admin=Depends(require_admin)) -> List[Dict[str, object]]:
    """List the PDFs uploaded for one department. Admin only."""
    try:
        docs = rag_manager.list_documents(slug)
    except RAGDepartmentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return docs


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_department_document(
    department: str = Form(..., description="Department name, e.g. 'Engineering'. Created if it doesn't exist yet."),
    file: UploadFile = File(...),
    _admin=Depends(require_admin),
) -> Dict[str, object]:
    """
    Upload a PDF for a department, creating the department (and therefore
    its chatbot tab) if this is the first document for that name. The
    department's RAG index is rebuilt immediately so it's queryable as
    soon as this call returns.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are accepted.")

    slug = _slugify(department)
    system = rag_manager.ensure_department(slug)

    labels = _load_labels()
    labels.setdefault(slug, department.strip())
    _save_labels(labels)

    dest_dir = Path(system.pdf_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / _safe_filename(file.filename)
    if dest_path.exists():
        dest_path = dest_dir / f"{dest_path.stem}_{int(dest_path.stat().st_mtime)}{dest_path.suffix}"

    try:
        with dest_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    finally:
        await file.close()

    rag_manager.reload(slug)
    system = rag_manager.get(slug)

    logger.info("Uploaded '%s' for department '%s' (status=%s).", dest_path.name, slug, system.status)

    return {
        "slug": slug,
        "label": labels[slug],
        "status": system.status,
        "document_count": len(system.list_documents()),
        "chunk_count": system.chunk_count,
        "last_error": system.last_error,
    }


@router.delete("/{slug}/documents/{filename}")
async def delete_department_document(slug: str, filename: str, _admin=Depends(require_admin)) -> Dict[str, object]:
    """Remove one PDF from a department and re-index. Admin only."""
    try:
        deleted = rag_manager.delete_document(slug, filename)
    except RAGDepartmentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

    system = rag_manager.get(slug)
    return {"slug": slug, "status": system.status, "document_count": len(system.list_documents())}


@router.delete("/{slug}")
async def delete_department(slug: str, _admin=Depends(require_admin)) -> Dict[str, str]:
    """Delete a department entirely: all its PDFs, its learned Q&A, and its chatbot tab. Admin only."""
    try:
        system = rag_manager.get(slug)
    except RAGDepartmentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    pdf_dir = Path(system.pdf_dir)
    if pdf_dir.is_dir():
        shutil.rmtree(pdf_dir, ignore_errors=True)

    rag_manager.remove_department(slug)

    labels = _load_labels()
    if labels.pop(slug, None) is not None:
        _save_labels(labels)

    return {"slug": slug, "status": "deleted"}