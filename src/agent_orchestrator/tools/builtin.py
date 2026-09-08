"""Starter tool implementations. Each is a plain function returning a string,
registered against the shared `ToolRegistry` in `register_builtin_tools`.

Security note: `code_execution` and `file_read`/`file_write` are confined to
`settings.sandbox_workdir` and run with a timeout, but this is a starter-level
sandbox (subprocess + timeout), not a hard security boundary. Phase 5's
docker-compose setup should run these in an isolated container before this
system is ever exposed to untrusted input.
"""
from __future__ import annotations

import ipaddress
import socket
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field

from ..config import settings
from ..schemas import SpecialistType
from .registry import ToolRegistry

ALL_SPECIALISTS = {
    SpecialistType.RESEARCH,
    SpecialistType.DATA_ANALYSIS,
    SpecialistType.WRITING,
    SpecialistType.CODE_EXECUTION,
}


def _sandbox_dir() -> Path:
    path = Path(settings.sandbox_workdir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_in_sandbox(relative_path: str) -> Path:
    sandbox = _sandbox_dir()
    candidate = (sandbox / relative_path).resolve()
    if sandbox not in candidate.parents and candidate != sandbox:
        raise PermissionError(
            f"Path '{relative_path}' escapes the sandbox directory."
        )
    return candidate


def web_search(query: str, max_results: int = 5) -> str:
    """Web search backed by Tavily if TAVILY_API_KEY is set, otherwise a clear stub."""
    import os

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return (
            f"[web_search stub] No TAVILY_API_KEY configured; would have searched for: "
            f"'{query}' (max_results={max_results})."
        )
    response = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": max_results},
        timeout=15,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    lines = [f"- {r.get('title', '')}: {r.get('url', '')}" for r in results]
    return "\n".join(lines) or "No results found."


def file_read(path: str) -> str:
    target = _resolve_in_sandbox(path)
    if not target.is_file():
        raise FileNotFoundError(f"'{path}' does not exist in the sandbox.")
    return target.read_text()


def file_write(path: str, content: str) -> str:
    target = _resolve_in_sandbox(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} bytes to '{path}'."


def code_execution(code: str) -> str:
    """Runs Python code in a subprocess, inside the sandbox dir, with a timeout."""
    sandbox = _sandbox_dir()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=sandbox, delete=False
    ) as tmp:
        tmp.write(code)
        script_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=sandbox,
            capture_output=True,
            text=True,
            timeout=settings.code_execution_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"Code execution exceeded {settings.code_execution_timeout_seconds}s timeout."
        ) from exc
    finally:
        Path(script_path).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Code execution failed with no stderr.")
    return result.stdout.strip()


def db_query(sql: str) -> str:
    """Read-only queries against a sandboxed SQLite database (sandbox/app.db)."""
    normalized = sql.strip().lower()
    if not normalized.startswith("select"):
        raise PermissionError("db_query only allows SELECT statements.")

    db_path = _sandbox_dir() / "app.db"
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description] if cursor.description else []
    finally:
        conn.close()

    if not rows:
        return "No rows returned."
    header = " | ".join(columns)
    body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
    return f"{header}\n{body}"


_BLOCKED_HOSTS = {"localhost", "0.0.0.0", "metadata.google.internal"}


def _is_private_or_local(host: str) -> bool:
    if host in _BLOCKED_HOSTS:
        return True
    try:
        addr = socket.gethostbyname(host)
        ip = ipaddress.ip_address(addr)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except (socket.gaierror, ValueError):
        return False


def api_call(url: str, method: str = "GET", json_body: dict | None = None) -> str:
    """Generic HTTP call, blocked from targeting localhost/private-network addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("api_call only supports http/https URLs.")
    if _is_private_or_local(parsed.hostname or ""):
        raise PermissionError(f"api_call may not target private/local host '{parsed.hostname}'.")

    response = requests.request(method.upper(), url, json=json_body, timeout=15)
    return f"HTTP {response.status_code}\n{response.text[:2000]}"


class WebSearchInput(BaseModel):
    query: str
    max_results: int = 5


class FileReadInput(BaseModel):
    path: str


class FileWriteInput(BaseModel):
    path: str
    content: str


class CodeExecutionInput(BaseModel):
    code: str


class DbQueryInput(BaseModel):
    sql: str


class ApiCallInput(BaseModel):
    url: str
    method: str = "GET"
    json_body: dict | None = None


def register_builtin_tools(registry: ToolRegistry) -> None:
    registry.register(
        name="web_search",
        description="Search the web for a query, returns a list of titles + URLs.",
        func=web_search,
        allowed_specialists={SpecialistType.RESEARCH},
        args_schema=WebSearchInput,
        rate_limit_per_minute=20,
    )
    registry.register(
        name="file_read",
        description="Read a file's contents from the sandboxed workspace.",
        func=file_read,
        allowed_specialists=ALL_SPECIALISTS,
        args_schema=FileReadInput,
        rate_limit_per_minute=60,
    )
    registry.register(
        name="file_write",
        description="Write content to a file in the sandboxed workspace.",
        func=file_write,
        allowed_specialists=ALL_SPECIALISTS,
        args_schema=FileWriteInput,
        rate_limit_per_minute=60,
    )
    registry.register(
        name="code_execution",
        description="Execute a Python snippet in a sandboxed subprocess and return stdout.",
        func=code_execution,
        allowed_specialists={SpecialistType.CODE_EXECUTION, SpecialistType.DATA_ANALYSIS},
        args_schema=CodeExecutionInput,
        rate_limit_per_minute=15,
    )
    registry.register(
        name="db_query",
        description="Run a read-only SELECT query against the sandboxed SQLite database.",
        func=db_query,
        allowed_specialists={SpecialistType.DATA_ANALYSIS},
        args_schema=DbQueryInput,
        rate_limit_per_minute=30,
    )
    registry.register(
        name="api_call",
        description="Make an HTTP request to an external API (private/local hosts blocked).",
        func=api_call,
        allowed_specialists={SpecialistType.RESEARCH, SpecialistType.DATA_ANALYSIS},
        args_schema=ApiCallInput,
        rate_limit_per_minute=20,
    )
