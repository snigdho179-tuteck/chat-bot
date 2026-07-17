"""
auth.py
-------
Authentication module for the Employee Handbook RAG Assistant.

Provides:
    - SQLite-backed user storage (table: users -> id, email, password, role)
    - Signup endpoint  (POST /auth/signup)  -> email, password, role
    - Login endpoint   (POST /auth/login)   -> email, password
    - A JWT access token is returned on successful login/signup. The
      frontend stores it (e.g. in localStorage) and can send it back as
      `Authorization: Bearer <token>` on subsequent requests.

    NOTE ON PASSWORD STORAGE (per explicit project decision):
    Passwords are stored PLAINTEXT in the database, with NO hashing.
    This is insecure and should never be used in a real production
    system — anyone with DB access (or a DB leak) gets every password
    in the clear. If this project ever moves past prototyping, swap the
    equality check in `login_user` for something like
    `passlib.hash.bcrypt` (hash on signup, `verify()` on login) with a
    minimal code change, since all password handling is isolated here.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Iterator, List, Optional

import jwt
from fastapi import APIRouter, Depends, Header, HTTPException, status
import re

from pydantic import BaseModel, field_validator

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _validate_email(value: str) -> str:
    if not isinstance(value, str) or not _EMAIL_RE.match(value):
        raise ValueError("must be a valid email address")
    return value

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DB_PATH = os.environ.get("AUTH_DB_PATH", "users.db")

# In production, set JWT_SECRET_KEY via an environment variable instead of
# relying on this default.
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.environ.get("JWT_EXPIRE_HOURS", "24"))

VALID_ROLES = {"user", "hr-employee", "finance", "admin"}

# The admin panel has three tabs. Every hr-employee/admin user carries an
# explicit list of which ones they're allowed to see (stored as JSON in the
# `tabs` column) so an admin can grant/restrict access per person rather
# than only per role. Plain "user" accounts never get panel access at all
# and are never routed to the panel by the frontend.
ALL_TABS = ["users", "queries", "files"]


def default_tabs_for_role(role: str) -> List[str]:
    if role == "admin":
        return list(ALL_TABS)
    if role == "hr-employee":
        return ["queries"]
    # "finance" and plain "user" accounts get no panel tabs by default;
    # an admin can grant specific tabs (e.g. "queries", "files") manually.
    return []


# --------------------------------------------------------------------------
# Database setup
# --------------------------------------------------------------------------

def init_db() -> None:
    """Create the users table if it doesn't already exist, and migrate in
    the `status` / `tabs` columns for databases created before they existed."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id       TEXT PRIMARY KEY,
                email    TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                role     TEXT NOT NULL,
                status   TEXT NOT NULL DEFAULT 'active',
                tabs     TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
        if "status" not in existing_cols:
            conn.execute("ALTER TABLE users ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
        if "tabs" not in existing_cols:
            conn.execute("ALTER TABLE users ADD COLUMN tabs TEXT NOT NULL DEFAULT '[]'")
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

class SignupRequest(BaseModel):
    email: str
    password: str
    role: str

    @field_validator("email")
    @classmethod
    def email_must_be_valid(cls, value: str) -> str:
        return _validate_email(value)

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, value: str) -> str:
        if value not in VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(VALID_ROLES)}")
        return value

    @field_validator("password")
    @classmethod
    def password_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("password must not be empty")
        return value


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def email_must_be_valid(cls, value: str) -> str:
        return _validate_email(value)


class AuthResponse(BaseModel):
    token: str
    user_id: str
    email: str
    role: str
    status: str
    tabs: List[str]


class UserOut(BaseModel):
    id: str
    email: str
    role: str
    status: str
    tabs: List[str]


class AdminCreateUserRequest(BaseModel):
    email: str
    password: str
    role: str
    status: str = "active"
    tabs: Optional[List[str]] = None

    @field_validator("email")
    @classmethod
    def email_must_be_valid(cls, value: str) -> str:
        return _validate_email(value)

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, value: str) -> str:
        if value not in VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(VALID_ROLES)}")
        return value

    @field_validator("status")
    @classmethod
    def status_must_be_valid(cls, value: str) -> str:
        if value not in {"active", "inactive"}:
            raise ValueError("status must be 'active' or 'inactive'")
        return value


class AdminUpdateUserRequest(BaseModel):
    role: Optional[str] = None
    status: Optional[str] = None
    tabs: Optional[List[str]] = None
    password: Optional[str] = None

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(VALID_ROLES)}")
        return value

    @field_validator("status")
    @classmethod
    def status_must_be_valid(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in {"active", "inactive"}:
            raise ValueError("status must be 'active' or 'inactive'")
        return value


# --------------------------------------------------------------------------
# JWT helpers
# --------------------------------------------------------------------------

def create_access_token(user_id: str, email: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token.")


def get_current_user(authorization: Optional[str] = Header(default=None)) -> dict:
    """FastAPI dependency: reads `Authorization: Bearer <token>` and returns the payload."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header.",
        )
    token = authorization.split(" ", 1)[1].strip()
    return decode_access_token(token)


def get_current_user_row(payload: dict = Depends(get_current_user)) -> sqlite3.Row:
    """Re-reads the user from the DB (rather than trusting the JWT claim)
    so a role/status/tabs change takes effect immediately, not only after
    the old token expires."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, email, role, status, tabs FROM users WHERE id = ?", (payload["sub"],)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User no longer exists.")
    if row["status"] == "inactive":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This account is inactive.")
    return row


def require_admin(user: sqlite3.Row = Depends(get_current_user_row)) -> sqlite3.Row:
    if user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required.")
    return user


def _row_to_user_out(row: sqlite3.Row) -> UserOut:
    return UserOut(
        id=row["id"], email=row["email"], role=row["role"],
        status=row["status"], tabs=json.loads(row["tabs"] or "[]"),
    )


# --------------------------------------------------------------------------
# Core auth logic
# --------------------------------------------------------------------------

def signup_user(data: SignupRequest) -> AuthResponse:
    tabs = default_tabs_for_role(data.role)
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?", (data.email,)
        ).fetchone()
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email already exists.",
            )

        user_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO users (id, email, password, role, status, tabs) VALUES (?, ?, ?, ?, 'active', ?)",
            (user_id, data.email, data.password, data.role, json.dumps(tabs)),
        )
        conn.commit()

    token = create_access_token(user_id, data.email, data.role)
    return AuthResponse(token=token, user_id=user_id, email=data.email, role=data.role, status="active", tabs=tabs)


def login_user(data: LoginRequest) -> AuthResponse:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, email, password, role, status, tabs FROM users WHERE email = ?",
            (data.email,),
        ).fetchone()

    # Plaintext comparison, per the (insecure, explicitly requested) storage
    # scheme documented at the top of this file.
    if row is None or row["password"] != data.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    if row["status"] == "inactive":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This account is inactive.")

    token = create_access_token(row["id"], row["email"], row["role"])
    return AuthResponse(
        token=token, user_id=row["id"], email=row["email"], role=row["role"],
        status=row["status"], tabs=json.loads(row["tabs"] or "[]"),
    )


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(request: SignupRequest) -> AuthResponse:
    """Create a new user (email, password, role) and return a JWT access token."""
    return signup_user(request)


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest) -> AuthResponse:
    """Authenticate an existing user (email, password) and return a JWT access token."""
    return login_user(request)


@router.get("/me", response_model=UserOut)
async def me(user: sqlite3.Row = Depends(get_current_user_row)) -> UserOut:
    """
    Return the caller's current role/status/tabs straight from the DB.
    The frontend calls this on every admin_panel.html load (rather than
    trusting the JWT or localStorage) so a tab-permission change an admin
    just made takes effect immediately, without waiting for re-login.
    """
    return _row_to_user_out(user)


@router.get("/users", response_model=list[UserOut])
async def list_users(_admin: sqlite3.Row = Depends(require_admin)) -> list[UserOut]:
    """List every user. Admin only."""
    with get_connection() as conn:
        rows = conn.execute("SELECT id, email, role, status, tabs FROM users ORDER BY email").fetchall()
    return [_row_to_user_out(r) for r in rows]


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(
    request: AdminCreateUserRequest, _admin: sqlite3.Row = Depends(require_admin)
) -> UserOut:
    """Create a user with a chosen role, status, and panel-tab access. Admin only."""
    tabs = request.tabs if request.tabs is not None else default_tabs_for_role(request.role)
    tabs = [t for t in tabs if t in ALL_TABS]
    with get_connection() as conn:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (request.email,)).fetchone()
        if existing is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists.")
        user_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO users (id, email, password, role, status, tabs) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, request.email, request.password, request.role, request.status, json.dumps(tabs)),
        )
        conn.commit()
    return UserOut(id=user_id, email=request.email, role=request.role, status=request.status, tabs=tabs)


@router.put("/users/{user_id}", response_model=UserOut)
async def update_user(
    user_id: str, request: AdminUpdateUserRequest, _admin: sqlite3.Row = Depends(require_admin)
) -> UserOut:
    """Update a user's role, status, panel-tab access, and/or password. Admin only."""
    if user_id == _admin["id"] and request.status is not None and request.status == "inactive":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You can't deactivate your own account while logged in.",
        )

    with get_connection() as conn:
        row = conn.execute("SELECT id, email, role, status, tabs FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

        new_role = request.role if request.role is not None else row["role"]
        new_status = request.status if request.status is not None else row["status"]
        if request.tabs is not None:
            new_tabs = [t for t in request.tabs if t in ALL_TABS]
        else:
            new_tabs = json.loads(row["tabs"] or "[]")

        if request.password:
            conn.execute(
                "UPDATE users SET role = ?, status = ?, tabs = ?, password = ? WHERE id = ?",
                (new_role, new_status, json.dumps(new_tabs), request.password, user_id),
            )
        else:
            conn.execute(
                "UPDATE users SET role = ?, status = ?, tabs = ? WHERE id = ?",
                (new_role, new_status, json.dumps(new_tabs), user_id),
            )
        conn.commit()

    return UserOut(id=row["id"], email=row["email"], role=new_role, status=new_status, tabs=new_tabs)


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str, admin: sqlite3.Row = Depends(require_admin)) -> None:
    """Delete a user. Admin only; an admin cannot delete their own account this way."""
    if user_id == admin["id"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You can't delete your own account while logged in.")
    with get_connection() as conn:
        result = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        if result.rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")