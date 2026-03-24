from __future__ import annotations
import hashlib
import json
import logging
import re
import shutil
import sqlite3
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Optional

import requests

import config

REPO_CONTEXT_SCHEMA_VERSION = 2
_BUILD_META_KEY_PREFIX = "repo_context"

_MARKDOWN_SECTION_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_CONTRACT_RE = re.compile(r"^\s*(?:abstract\s+)?(contract|interface|library)\s+([A-Za-z_][A-Za-z0-9_]*)")
_FUNCTION_RE = re.compile(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)")
_VYPER_FUNCTION_RE = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)")
_EVENT_RE = re.compile(r"^\s*event\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)")
_ERROR_RE = re.compile(r"^\s*error\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)")
_STRUCT_RE = re.compile(r"^\s*struct\s+([A-Za-z_][A-Za-z0-9_]*)")
_JSON_ADDRESS_KEYWORDS = {"address", "vault", "router", "strategy", "token", "gauge"}


@dataclass(frozen=True)
class RepoSource:
    owner: str
    repo: str
    ref: Optional[str]
    product_tag: str
    authority_tag: str
    legacy: bool
    include_globs: tuple[str, ...]
    exclude_globs: tuple[str, ...]

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


@dataclass(frozen=True)
class RepoFile:
    repo_name: str
    repo_ref: str
    product_tag: str
    authority_tag: str
    legacy: bool
    path: str
    language: str
    content: str


@dataclass(frozen=True)
class RepoSearchResult:
    artifact_ref: str
    repo_name: str
    repo_ref: str
    path: str
    segment_type: str
    title: str
    snippet: str
    product_tag: str
    authority_tag: str
    legacy: bool
    score: float


def _manifest_payload(manifest_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(manifest_path or config.REPO_CONTEXT_MANIFEST_PATH)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_repo_manifest(payload: dict[str, Any]) -> None:
    repos = payload.get("repos")
    if not isinstance(repos, list) or not repos:
        raise ValueError("Repo manifest must include a non-empty 'repos' list.")

    seen: set[str] = set()
    default_owner = payload.get("default_owner", "yearn")
    for entry in repos:
        if not isinstance(entry, dict):
            raise ValueError("Each repo manifest entry must be an object.")

        missing = [
            key for key in ("repo", "product_tag", "authority_tag", "include")
            if key not in entry
        ]
        if missing:
            raise ValueError(f"Repo manifest entry missing required keys: {', '.join(missing)}")

        include_globs = entry.get("include")
        if not isinstance(include_globs, list) or not include_globs:
            raise ValueError(f"Repo '{entry.get('repo', '<unknown>')}' must define non-empty include globs.")

        repo_name = f"{entry.get('owner', default_owner)}/{entry['repo']}"
        if repo_name in seen:
            raise ValueError(f"Duplicate repo entry in manifest: {repo_name}")
        seen.add(repo_name)


def manifest_hash(manifest_path: Path | str | None = None) -> str:
    payload = _manifest_payload(manifest_path)
    validate_repo_manifest(payload)
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_repo_manifest(manifest_path: Path | str | None = None) -> list[RepoSource]:
    payload = _manifest_payload(manifest_path)
    validate_repo_manifest(payload)
    default_owner = payload.get("default_owner", "yearn")
    sources: list[RepoSource] = []

    for entry in payload.get("repos", []):
        owner = entry.get("owner", default_owner)
        repo = entry["repo"]
        sources.append(
            RepoSource(
                owner=owner,
                repo=repo,
                ref=entry.get("ref"),
                product_tag=entry["product_tag"],
                authority_tag=entry["authority_tag"],
                legacy=bool(entry.get("legacy", False)),
                include_globs=tuple(entry.get("include", [])),
                exclude_globs=tuple(entry.get("exclude", [])),
            )
        )
    return sources


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ysupport-repo-context",
    }
    if config.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"
    return headers


def _repo_cache_dir(cache_dir: Path, repo_name: str, repo_ref: str) -> Path:
    safe_repo = repo_name.replace("/", "__")
    safe_ref = repo_ref.replace("/", "__")
    return cache_dir / f"{safe_repo}@{safe_ref}"


def _cleanup_obsolete_repo_cache_dirs(cache_dir: Path, repo_name: str, keep_ref: str) -> None:
    safe_repo = repo_name.replace("/", "__")
    expected_dir = _repo_cache_dir(cache_dir, repo_name, keep_ref)
    for child in cache_dir.glob(f"{safe_repo}@*"):
        if child == expected_dir:
            continue
        if child.is_dir():
            shutil.rmtree(child)


def _detect_language(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".sol":
        return "solidity"
    if suffix == ".vy":
        return "vyper"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".json":
        return "json"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix in {".txt", ".rst"}:
        return "text"
    return "text"


def _path_matches(path: str, patterns: Iterable[str]) -> bool:
    pure_path = PurePosixPath(path)
    return any(pure_path.match(pattern) for pattern in patterns)


def _should_include_path(path: str, source: RepoSource) -> bool:
    normalized_path = path.lstrip("./")
    if source.exclude_globs and _path_matches(normalized_path, source.exclude_globs):
        return False
    return _path_matches(normalized_path, source.include_globs)


def _resolve_default_branch(session: requests.Session, source: RepoSource) -> str:
    if source.ref:
        return source.ref
    response = session.get(
        f"https://api.github.com/repos/{source.full_name}",
        headers=_github_headers(),
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    default_branch = payload.get("default_branch")
    if not default_branch:
        raise ValueError(f"Could not resolve default branch for {source.full_name}")
    return default_branch


def _download_repo_archive(session: requests.Session, source: RepoSource, repo_ref: str) -> bytes:
    archive_url = f"https://codeload.github.com/{source.owner}/{source.repo}/tar.gz/{repo_ref}"
    response = session.get(archive_url, headers=_github_headers(), timeout=60)
    response.raise_for_status()
    return response.content


def _extract_repo_files(
    archive_bytes: bytes,
    source: RepoSource,
    repo_ref: str,
    cache_dir: Path,
) -> list[RepoFile]:
    target_dir = _repo_cache_dir(cache_dir, source.full_name, repo_ref)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    repo_files: list[RepoFile] = []
    with tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue

            parts = PurePosixPath(member.name).parts
            if len(parts) < 2:
                continue
            relative_path = str(PurePosixPath(*parts[1:]))
            if not _should_include_path(relative_path, source):
                continue

            extracted = tar.extractfile(member)
            if extracted is None:
                continue

            raw_bytes = extracted.read()
            try:
                content = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                content = raw_bytes.decode("utf-8", errors="ignore")

            file_path = target_dir / relative_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

            repo_files.append(
                RepoFile(
                    repo_name=source.full_name,
                    repo_ref=repo_ref,
                    product_tag=source.product_tag,
                    authority_tag=source.authority_tag,
                    legacy=source.legacy,
                    path=relative_path,
                    language=_detect_language(relative_path),
                    content=content,
                )
            )
    return repo_files


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _collect_signature(lines: list[str], start_idx: int, *, max_lines: int = 12) -> str:
    parts: list[str] = []
    paren_balance = 0
    saw_open_paren = False

    for offset in range(max_lines):
        idx = start_idx + offset
        if idx >= len(lines):
            break
        line = lines[idx].strip()
        if not line:
            if parts:
                break
            continue

        parts.append(line)
        paren_balance += line.count("(") - line.count(")")
        if "(" in line:
            saw_open_paren = True
        if saw_open_paren and paren_balance <= 0:
            break

    return " ".join(parts)


def _iter_markdown_segments(path: str, content: str) -> Iterator[tuple[str, str, str]]:
    lines = content.splitlines()
    headings: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        match = _MARKDOWN_SECTION_RE.match(line)
        if match:
            headings.append((idx, match.group(2).strip()))

    if not headings:
        body = content.strip()
        if body:
            yield ("section", Path(path).name, body)
        return

    for index, (start_line, heading) in enumerate(headings):
        end_line = headings[index + 1][0] if index + 1 < len(headings) else len(lines)
        body = "\n".join(lines[start_line:end_line]).strip()
        if body:
            yield ("section", heading, body)


def _iter_code_segments(path: str, content: str, language: str) -> Iterator[tuple[str, str, str]]:
    lines = content.splitlines()
    if not lines:
        return

    contract_matches: list[tuple[int, str, str]] = []
    symbol_matches: list[tuple[int, str, str]] = []

    for idx, line in enumerate(lines):
        contract_match = _CONTRACT_RE.match(line)
        if contract_match:
            kind = contract_match.group(1)
            name = contract_match.group(2)
            contract_matches.append((idx, kind, name))
            continue

        function_signature = _collect_signature(lines, idx) if line.lstrip().startswith("function ") else line
        function_match = _FUNCTION_RE.match(function_signature)
        if function_match:
            signature = f"{function_match.group(1)}({function_match.group(2).strip()})"
            symbol_matches.append((idx, "function", signature))
            continue

        if language == "vyper":
            vyper_signature = _collect_signature(lines, idx) if line.lstrip().startswith("def ") else line
            vyper_function_match = _VYPER_FUNCTION_RE.match(vyper_signature)
            if vyper_function_match:
                signature = f"{vyper_function_match.group(1)}({vyper_function_match.group(2).strip()})"
                symbol_matches.append((idx, "function", signature))
                continue

        event_match = _EVENT_RE.match(line)
        if event_match:
            signature = f"{event_match.group(1)}({event_match.group(2).strip()})"
            symbol_matches.append((idx, "event", signature))
            continue

        error_match = _ERROR_RE.match(line)
        if error_match:
            signature = f"{error_match.group(1)}({error_match.group(2).strip()})"
            symbol_matches.append((idx, "error", signature))
            continue

        struct_match = _STRUCT_RE.match(line)
        if struct_match:
            symbol_matches.append((idx, "struct", struct_match.group(1)))

    if not contract_matches and not symbol_matches:
        snippet = "\n".join(lines[: min(len(lines), 180)]).strip()
        if snippet:
            yield ("file", Path(path).name, snippet)
        return

    for index, (start_line, kind, name) in enumerate(contract_matches):
        end_line = contract_matches[index + 1][0] if index + 1 < len(contract_matches) else len(lines)
        snippet = "\n".join(lines[start_line : min(end_line, start_line + 260)]).strip()
        if snippet:
            yield (kind, name, snippet)

    combined_boundaries = sorted(symbol_matches, key=lambda item: item[0])
    for index, (start_line, kind, title) in enumerate(combined_boundaries):
        end_line = combined_boundaries[index + 1][0] if index + 1 < len(combined_boundaries) else len(lines)
        snippet = "\n".join(lines[start_line : min(end_line, start_line + 120)]).strip()
        if snippet:
            yield (kind, title, snippet)


def _flatten_json(
    value: Any,
    prefix: str = "",
    facts: Optional[list[tuple[str, str]]] = None,
) -> list[tuple[str, str]]:
    if facts is None:
        facts = []

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_json(child, child_prefix, facts)
        return facts

    if isinstance(value, list):
        if all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
            facts.append((prefix, ", ".join(str(item) for item in value[:20])))
            return facts
        for idx, child in enumerate(value[:30]):
            child_prefix = f"{prefix}[{idx}]"
            _flatten_json(child, child_prefix, facts)
        return facts

    facts.append((prefix, str(value)))
    return facts


def _iter_json_segments(path: str, content: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        snippet = content[:2000].strip()
        return ([("json", Path(path).name, snippet)] if snippet else []), []

    facts = _flatten_json(payload)
    summary_lines = [f"{key}: {value}" for key, value in facts[:80] if value and value != "None"]
    summary = "\n".join(summary_lines).strip()
    segments = [("json", Path(path).name, summary)] if summary else []
    fact_rows: list[tuple[str, str, str]] = []
    for key, value in facts:
        if not key or not value:
            continue
        fact_type = "address" if any(token in key.lower() for token in _JSON_ADDRESS_KEYWORDS) else "metadata"
        fact_rows.append((key, value, fact_type))
    return segments, fact_rows


def _iter_repo_segments(repo_file: RepoFile) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    if repo_file.language == "markdown":
        return list(_iter_markdown_segments(repo_file.path, repo_file.content)), []
    if repo_file.language in {"solidity", "vyper"}:
        return list(_iter_code_segments(repo_file.path, repo_file.content, repo_file.language)), []
    if repo_file.language == "json":
        return _iter_json_segments(repo_file.path, repo_file.content)

    body = repo_file.content[:2000].strip()
    return ([("file", Path(repo_file.path).name, body)] if body else []), []


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS build_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_name TEXT NOT NULL,
            repo_ref TEXT NOT NULL,
            path TEXT NOT NULL,
            language TEXT NOT NULL,
            product_tag TEXT NOT NULL,
            authority_tag TEXT NOT NULL,
            legacy INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            content TEXT NOT NULL,
            UNIQUE(repo_name, repo_ref, path)
        );

        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            repo_name TEXT NOT NULL,
            repo_ref TEXT NOT NULL,
            path TEXT NOT NULL,
            language TEXT NOT NULL,
            product_tag TEXT NOT NULL,
            authority_tag TEXT NOT NULL,
            legacy INTEGER NOT NULL,
            segment_type TEXT NOT NULL,
            title TEXT NOT NULL,
            snippet TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            repo_name TEXT NOT NULL,
            repo_ref TEXT NOT NULL,
            path TEXT NOT NULL,
            product_tag TEXT NOT NULL,
            authority_tag TEXT NOT NULL,
            legacy INTEGER NOT NULL,
            fact_key TEXT NOT NULL,
            fact_value TEXT NOT NULL,
            fact_type TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
            repo_name,
            path,
            product_tag,
            authority_tag,
            segment_type,
            title,
            snippet
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
            repo_name,
            path,
            product_tag,
            authority_tag,
            fact_key,
            fact_value,
            fact_type
        );
        """
    )


def _set_build_meta(conn: sqlite3.Connection, key: str, value: Any) -> None:
    normalized_value = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    conn.execute(
        """
        INSERT INTO build_meta (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, normalized_value),
    )


def _get_build_meta(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT key, value FROM build_meta").fetchall()
    return {row["key"]: row["value"] for row in rows}


def _clear_repo_rows(conn: sqlite3.Connection, repo_name: str) -> None:
    file_rows = conn.execute("SELECT id FROM files WHERE repo_name = ?", (repo_name,)).fetchall()
    file_ids = [row[0] for row in file_rows]

    if file_ids:
        placeholders = ", ".join("?" for _ in file_ids)
        conn.execute(f"DELETE FROM segments WHERE file_id IN ({placeholders})", file_ids)
        conn.execute(f"DELETE FROM facts WHERE file_id IN ({placeholders})", file_ids)
    conn.execute("DELETE FROM files WHERE repo_name = ?", (repo_name,))
    conn.execute("DELETE FROM segments_fts WHERE repo_name = ?", (repo_name,))
    conn.execute("DELETE FROM facts_fts WHERE repo_name = ?", (repo_name,))


def _insert_repo_file(conn: sqlite3.Connection, repo_file: RepoFile) -> tuple[int, int, int]:
    cursor = conn.execute(
        """
        INSERT INTO files (
            repo_name, repo_ref, path, language, product_tag, authority_tag, legacy, content_hash, content
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            repo_file.repo_name,
            repo_file.repo_ref,
            repo_file.path,
            repo_file.language,
            repo_file.product_tag,
            repo_file.authority_tag,
            int(repo_file.legacy),
            _content_hash(repo_file.content),
            repo_file.content,
        ),
    )
    file_id = int(cursor.lastrowid)

    segments, facts = _iter_repo_segments(repo_file)
    segment_count = 0
    fact_count = 0

    for segment_type, title, snippet in segments:
        normalized_snippet = snippet.strip()
        if not normalized_snippet:
            continue

        segment_cursor = conn.execute(
            """
            INSERT INTO segments (
                file_id, repo_name, repo_ref, path, language, product_tag, authority_tag, legacy,
                segment_type, title, snippet
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                repo_file.repo_name,
                repo_file.repo_ref,
                repo_file.path,
                repo_file.language,
                repo_file.product_tag,
                repo_file.authority_tag,
                int(repo_file.legacy),
                segment_type,
                title,
                normalized_snippet,
            ),
        )
        conn.execute(
            """
            INSERT INTO segments_fts (
                rowid, repo_name, path, product_tag, authority_tag, segment_type, title, snippet
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(segment_cursor.lastrowid),
                repo_file.repo_name,
                repo_file.path,
                repo_file.product_tag,
                repo_file.authority_tag,
                segment_type,
                title,
                normalized_snippet,
            ),
        )
        segment_count += 1

    for fact_key, fact_value, fact_type in facts:
        normalized_value = _normalize_whitespace(fact_value)
        if not normalized_value:
            continue

        fact_cursor = conn.execute(
            """
            INSERT INTO facts (
                file_id, repo_name, repo_ref, path, product_tag, authority_tag, legacy, fact_key, fact_value, fact_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                repo_file.repo_name,
                repo_file.repo_ref,
                repo_file.path,
                repo_file.product_tag,
                repo_file.authority_tag,
                int(repo_file.legacy),
                fact_key,
                normalized_value,
                fact_type,
            ),
        )
        conn.execute(
            """
            INSERT INTO facts_fts (
                rowid, repo_name, path, product_tag, authority_tag, fact_key, fact_value, fact_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(fact_cursor.lastrowid),
                repo_file.repo_name,
                repo_file.path,
                repo_file.product_tag,
                repo_file.authority_tag,
                fact_key,
                normalized_value,
                fact_type,
            ),
        )
        fact_count += 1

    return file_id, segment_count, fact_count


def build_repo_context_index(
    manifest_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    repo_names: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    manifest_sources = load_repo_manifest(manifest_path)
    selected_repos = {name for name in repo_names or []}

    if selected_repos:
        sources = [
            source for source in manifest_sources
            if source.repo in selected_repos or source.full_name in selected_repos
        ]
        if not sources:
            requested = ", ".join(sorted(selected_repos))
            raise ValueError(f"No repos matched the requested filter: {requested}")
    else:
        sources = manifest_sources

    cache_root = Path(cache_dir or config.REPO_CONTEXT_CACHE_DIR)
    db_file = Path(db_path or config.REPO_CONTEXT_DB_PATH)
    cache_root.mkdir(parents=True, exist_ok=True)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    built_at = datetime.now(timezone.utc).isoformat()
    manifest_path_str = str(Path(manifest_path or config.REPO_CONTEXT_MANIFEST_PATH))
    current_manifest_hash = manifest_hash(manifest_path)
    repo_summaries: list[dict[str, Any]] = []
    total_files = 0
    total_segments = 0
    total_facts = 0

    with requests.Session() as session, sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_schema(conn)
        for source in sources:
            repo_ref = _resolve_default_branch(session, source)
            logging.info("Building repo context for %s @ %s", source.full_name, repo_ref)
            _cleanup_obsolete_repo_cache_dirs(cache_root, source.full_name, repo_ref)
            repo_files = _extract_repo_files(
                archive_bytes=_download_repo_archive(session, source, repo_ref),
                source=source,
                repo_ref=repo_ref,
                cache_dir=cache_root,
            )

            _clear_repo_rows(conn, source.full_name)

            repo_segment_count = 0
            repo_fact_count = 0
            for repo_file in repo_files:
                _, segment_count, fact_count = _insert_repo_file(conn, repo_file)
                repo_segment_count += segment_count
                repo_fact_count += fact_count

            repo_summaries.append(
                {
                    "repo_name": source.full_name,
                    "repo_ref": repo_ref,
                    "files_indexed": len(repo_files),
                    "segments_indexed": repo_segment_count,
                    "facts_indexed": repo_fact_count,
                }
            )
            total_files += len(repo_files)
            total_segments += repo_segment_count
            total_facts += repo_fact_count

        build_summary = {
            "repos_indexed": len(repo_summaries),
            "files_indexed": total_files,
            "segments_indexed": total_segments,
            "facts_indexed": total_facts,
            "repos": repo_summaries,
        }
        _set_build_meta(conn, f"{_BUILD_META_KEY_PREFIX}.schema_version", REPO_CONTEXT_SCHEMA_VERSION)
        _set_build_meta(conn, f"{_BUILD_META_KEY_PREFIX}.built_at", built_at)
        _set_build_meta(conn, f"{_BUILD_META_KEY_PREFIX}.manifest_hash", current_manifest_hash)
        _set_build_meta(conn, f"{_BUILD_META_KEY_PREFIX}.manifest_path", manifest_path_str)
        _set_build_meta(conn, f"{_BUILD_META_KEY_PREFIX}.summary", build_summary)
        conn.commit()
        conn.execute("VACUUM")

    return {
        "manifest_path": manifest_path_str,
        "cache_dir": str(cache_root),
        "db_path": str(db_file),
        "built_at": built_at,
        "manifest_hash": current_manifest_hash,
        "schema_version": REPO_CONTEXT_SCHEMA_VERSION,
        "repos_indexed": len(repo_summaries),
        "files_indexed": total_files,
        "segments_indexed": total_segments,
        "facts_indexed": total_facts,
        "repos": repo_summaries,
    }


def _connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path or config.REPO_CONTEXT_DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def get_repo_context_status(
    db_path: Path | str | None = None,
    *,
    enabled: Optional[bool] = None,
    require_fresh: Optional[bool] = None,
    max_age_hours: Optional[int] = None,
) -> dict[str, Any]:
    is_enabled = config.ENABLE_REPO_CONTEXT if enabled is None else enabled
    freshness_required = config.REPO_CONTEXT_REQUIRE_FRESH if require_fresh is None else require_fresh
    max_age = config.REPO_CONTEXT_MAX_AGE_HOURS if max_age_hours is None else max_age_hours
    path = Path(db_path or config.REPO_CONTEXT_DB_PATH)

    status: dict[str, Any] = {
        "enabled": is_enabled,
        "require_fresh": freshness_required,
        "max_age_hours": max_age,
        "db_path": str(path),
        "available": False,
        "fresh": False,
        "state": "disabled" if not is_enabled else "missing",
        "reason": "",
        "built_at": None,
        "age_hours": None,
        "schema_version": None,
        "manifest_hash": None,
        "summary": None,
    }

    if not is_enabled:
        status["reason"] = "Repo context is disabled by configuration."
        return status

    if not path.exists() or path.stat().st_size == 0:
        status["reason"] = "Repo context database is missing or empty."
        return status

    try:
        with _connect(path) as conn:
            _ensure_schema(conn)
            meta = _get_build_meta(conn)
    except Exception as exc:
        status["state"] = "error"
        status["reason"] = f"Failed to inspect repo context database: {exc}"
        return status

    built_at_raw = meta.get(f"{_BUILD_META_KEY_PREFIX}.built_at")
    manifest_hash_value = meta.get(f"{_BUILD_META_KEY_PREFIX}.manifest_hash")
    schema_version_raw = meta.get(f"{_BUILD_META_KEY_PREFIX}.schema_version")
    summary_raw = meta.get(f"{_BUILD_META_KEY_PREFIX}.summary")

    if not built_at_raw or not manifest_hash_value or not schema_version_raw:
        status["state"] = "stale"
        status["available"] = True
        status["reason"] = "Repo context database exists but build metadata is incomplete."
        return status

    status["available"] = True
    status["built_at"] = built_at_raw
    status["manifest_hash"] = manifest_hash_value
    try:
        status["schema_version"] = int(schema_version_raw)
    except ValueError:
        status["schema_version"] = schema_version_raw
    if summary_raw:
        try:
            status["summary"] = json.loads(summary_raw)
        except json.JSONDecodeError:
            status["summary"] = summary_raw

    try:
        built_at_dt = datetime.fromisoformat(built_at_raw)
        if built_at_dt.tzinfo is None:
            built_at_dt = built_at_dt.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - built_at_dt).total_seconds() / 3600
        status["age_hours"] = round(age_hours, 2)
        is_fresh = age_hours <= max_age
    except Exception:
        is_fresh = False
        status["reason"] = "Repo context build timestamp is invalid."

    status["fresh"] = is_fresh

    if manifest_hash_value != manifest_hash():
        status["state"] = "stale"
        status["reason"] = "Repo context database was built from a different manifest version."
        return status

    if status["schema_version"] != REPO_CONTEXT_SCHEMA_VERSION:
        status["state"] = "stale"
        status["reason"] = "Repo context database schema version does not match the current code."
        return status

    if freshness_required and not is_fresh:
        status["state"] = "stale"
        if not status["reason"]:
            status["reason"] = "Repo context database is older than the configured freshness limit."
        return status

    status["state"] = "ready" if is_fresh else "available"
    if not status["reason"]:
        status["reason"] = "Repo context database is ready."
    return status


def _fts_query(query: str) -> str:
    tokens = [token for token in re.findall(r"[A-Za-z0-9_]+", query.lower()) if len(token) > 1]
    if not tokens:
        return ""
    unique_tokens = list(dict.fromkeys(tokens[:10]))
    return " OR ".join(f"{token}*" for token in unique_tokens)


def _repo_scope(query: str) -> dict[str, Any]:
    lowered = query.lower()
    include_legacy = any(term in lowered for term in ["veyfi", "ve yfi", "legacy", "v1", "migrate", "migration"])
    wants_ui = any(
        term in lowered
        for term in ["button", "browser", "page", "page load", "site", "screen", "cta", "ui", "frontend", "web"]
    )
    preferred_products = {
        "styfi": ["styfi"],
        "vault": ["vaults"],
        "vaults": ["vaults"],
        "router": ["router"],
        "strategy": ["strategy"],
        "strategies": ["strategy"],
        "periphery": ["periphery"],
        "security": ["security"],
    }
    product_filters = [
        product_tag
        for keyword, product_tags in preferred_products.items()
        if keyword in lowered
        for product_tag in product_tags
    ]
    return {
        "include_legacy": include_legacy,
        "include_ui": wants_ui and config.REPO_CONTEXT_INCLUDE_UI,
        "product_filters": list(dict.fromkeys(product_filters)),
    }


def _query_tokens(query: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"[A-Za-z0-9_]+", query.lower())))


def _code_query_hints(query: str) -> dict[str, list[str] | bool]:
    file_hints = [match.lower() for match in re.findall(r"\b([A-Za-z0-9_-]+\.(?:vy|sol))\b", query)]
    raw_identifiers = re.findall(r"\b[_A-Za-z][_A-Za-z0-9]*\b", query)

    identifier_hints = [
        token.lower()
        for token in raw_identifiers
        if "_" in token or any(char.isupper() for char in token[1:])
    ]
    contract_hints = [
        token.lower()
        for token in raw_identifiers
        if token[:1].isupper() and any(char.isdigit() for char in token)
    ]
    for file_hint in file_hints:
        contract_hints.append(Path(file_hint).stem.lower())

    code_oriented = any(
        marker in query.lower()
        for marker in ["_", ".vy", ".sol", "function", "contract", "redeem", "withdraw", "balanceof", "strategy"]
    )
    return {
        "code_oriented": code_oriented,
        "file_hints": list(dict.fromkeys(file_hints)),
        "identifier_hints": list(dict.fromkeys(identifier_hints)),
        "contract_hints": list(dict.fromkeys(contract_hints)),
    }


def _rerank_repo_rows(rows: list[sqlite3.Row], query: str, *, limit: int) -> list[sqlite3.Row]:
    query_lower = query.lower()
    tokens = _query_tokens(query)
    hints = _code_query_hints(query)
    symbol_tokens = [token for token in tokens if "_" in token]
    file_tokens = {token for token in tokens if token.endswith(("vy", "sol")) or token.startswith("vault")}
    identifier_hints = hints["identifier_hints"]
    contract_hints = hints["contract_hints"]
    file_hints = hints["file_hints"]
    code_oriented = bool(hints["code_oriented"])
    markdown_names = {"security.md", "readme.md", "specification.md", "tech_spec.md"}

    def score(row: sqlite3.Row) -> tuple[float, float]:
        path = row["path"].lower()
        title = row["title"].lower()
        snippet = row["snippet"].lower()
        segment_type = row["segment_type"].lower()
        path_name = Path(path).name.lower()

        boost = 0.0
        for token in tokens:
            if token in title:
                boost += 5.0
            elif token in path:
                boost += 3.0
            elif token in snippet[:500]:
                boost += 1.0

        for file_hint in file_hints:
            if path.endswith(file_hint):
                boost += 15.0
            elif Path(path).stem.lower() == Path(file_hint).stem.lower():
                boost += 10.0

        for identifier in identifier_hints:
            if title.startswith(f"{identifier}("):
                boost += 18.0
            elif title == identifier:
                boost += 15.0
            elif identifier in title:
                boost += 10.0

        for contract_hint in contract_hints:
            if Path(path).stem.lower() == contract_hint:
                boost += 10.0
            elif contract_hint in path_name:
                boost += 6.0

        if any(token in title for token in symbol_tokens):
            boost += 8.0
        if any(token in path_name for token in file_tokens):
            boost += 4.0
        if "vaultv3.vy" in query_lower and "vaultv3.vy" in path:
            boost += 6.0
        if code_oriented and segment_type in {"function", "contract", "interface", "library"}:
            boost += 4.0
        if code_oriented and Path(path).suffix.lower() in {".vy", ".sol"}:
            boost += 3.0
        if code_oriented and path.startswith("contracts/"):
            boost += 2.0
        if code_oriented and path_name in markdown_names:
            boost -= 8.0
        if code_oriented and segment_type == "section":
            boost -= 3.0

        base_rank = -float(row["rank_score"])
        return (boost, base_rank)

    ranked_rows = sorted(rows, key=score, reverse=True)
    return ranked_rows[:limit]


def _supplement_repo_rows(
    conn: sqlite3.Connection,
    *,
    scope: dict[str, Any],
    hints: dict[str, list[str] | bool],
    result_limit: int,
) -> list[sqlite3.Row]:
    file_hints = [value for value in hints["file_hints"] if isinstance(value, str)]
    identifier_hints = [value for value in hints["identifier_hints"] if isinstance(value, str)]
    contract_hints = [value for value in hints["contract_hints"] if isinstance(value, str)]
    if not hints["code_oriented"] or (not file_hints and not identifier_hints and not contract_hints):
        return []

    filter_clauses = []
    params: list[Any] = []

    if not scope["include_legacy"]:
        filter_clauses.append("segments.legacy = 0")
    if not scope["include_ui"]:
        filter_clauses.append("segments.authority_tag != 'ui_flow'")
    product_filters = scope["product_filters"]
    if product_filters:
        placeholders = ", ".join("?" for _ in product_filters)
        filter_clauses.append(f"segments.product_tag IN ({placeholders})")
        params.extend(product_filters)
    filter_clauses.append("(lower(segments.path) LIKE '%.sol' OR lower(segments.path) LIKE '%.vy')")

    match_clauses = []
    for file_hint in file_hints:
        stem = Path(file_hint).stem.lower()
        match_clauses.append("lower(segments.path) LIKE ?")
        params.append(f"%{file_hint}%")
        if stem:
            match_clauses.append("lower(segments.path) LIKE ?")
            params.append(f"%{stem}%")

    for identifier in identifier_hints:
        match_clauses.append("lower(segments.title) LIKE ?")
        params.append(f"%{identifier}%")

    for contract_hint in contract_hints:
        match_clauses.append("lower(segments.path) LIKE ?")
        params.append(f"%{contract_hint}%")
        match_clauses.append("lower(segments.title) LIKE ?")
        params.append(f"%{contract_hint}%")

    if not match_clauses:
        return []

    candidate_limit = max(result_limit * 12, 80)
    sql = f"""
        SELECT
            segments.id AS artifact_id,
            segments.repo_name,
            segments.repo_ref,
            segments.path,
            segments.segment_type,
            segments.title,
            segments.snippet,
            segments.product_tag,
            segments.authority_tag,
            segments.legacy,
            0.0 AS rank_score
        FROM segments
        WHERE {' AND '.join(filter_clauses)} AND ({' OR '.join(match_clauses)})
        LIMIT ?
    """
    params.append(candidate_limit)
    return conn.execute(sql, params).fetchall()


def search_repo_context(
    query: str,
    limit: Optional[int] = None,
    include_legacy: Optional[bool] = None,
    include_ui: Optional[bool] = None,
    db_path: Path | str | None = None,
) -> list[RepoSearchResult]:
    status = get_repo_context_status(db_path=db_path, enabled=True)
    if status["state"] not in {"ready", "available"}:
        return []

    fts_query = _fts_query(query)
    if not fts_query:
        return []

    scope = _repo_scope(query)
    if include_legacy is not None:
        scope["include_legacy"] = include_legacy
    if include_ui is not None:
        scope["include_ui"] = include_ui and config.REPO_CONTEXT_INCLUDE_UI
    result_limit = limit or config.REPO_CONTEXT_TOP_K
    product_filters = scope["product_filters"]
    hints = _code_query_hints(query)

    filter_clauses = ["segments_fts MATCH ?"]
    params: list[Any] = [fts_query]

    if not scope["include_legacy"]:
        filter_clauses.append("segments.legacy = 0")
    if not scope["include_ui"]:
        filter_clauses.append("segments.authority_tag != 'ui_flow'")
    if product_filters:
        placeholders = ", ".join("?" for _ in product_filters)
        filter_clauses.append(f"segments.product_tag IN ({placeholders})")
        params.extend(product_filters)

    where_clause = f"WHERE {' AND '.join(filter_clauses)}"
    candidate_limit = max(result_limit * 6, result_limit)
    sql = f"""
        SELECT
            segments.id AS artifact_id,
            segments.repo_name,
            segments.repo_ref,
            segments.path,
            segments.segment_type,
            segments.title,
            segments.snippet,
            segments.product_tag,
            segments.authority_tag,
            segments.legacy,
            bm25(segments_fts) AS rank_score
        FROM segments_fts
        JOIN segments ON segments.id = segments_fts.rowid
        {where_clause}
        ORDER BY rank_score
        LIMIT ?
    """
    params.append(candidate_limit)

    with _connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        supplemental_rows = _supplement_repo_rows(conn, scope=scope, hints=hints, result_limit=result_limit)
        if supplemental_rows:
            merged_rows: dict[int, sqlite3.Row] = {int(row["artifact_id"]): row for row in rows}
            for row in supplemental_rows:
                merged_rows.setdefault(int(row["artifact_id"]), row)
            rows = list(merged_rows.values())

        if not rows:
            fact_filter_clauses = ["facts_fts MATCH ?"]
            fact_params: list[Any] = [fts_query]
            if not scope["include_legacy"]:
                fact_filter_clauses.append("facts.legacy = 0")
            if not scope["include_ui"]:
                fact_filter_clauses.append("facts.authority_tag != 'ui_flow'")
            if product_filters:
                placeholders = ", ".join("?" for _ in product_filters)
                fact_filter_clauses.append(f"facts.product_tag IN ({placeholders})")
                fact_params.extend(product_filters)
            fact_params.append(result_limit)

            fact_rows = conn.execute(
                f"""
                SELECT
                    facts.id AS artifact_id,
                    facts.repo_name,
                    facts.repo_ref,
                    facts.path,
                    facts.fact_type AS segment_type,
                    facts.fact_key AS title,
                    facts.fact_value AS snippet,
                    facts.product_tag,
                    facts.authority_tag,
                    facts.legacy,
                    bm25(facts_fts) AS rank_score
                FROM facts_fts
                JOIN facts ON facts.id = facts_fts.rowid
                WHERE {" AND ".join(fact_filter_clauses)}
                ORDER BY rank_score
                LIMIT ?
                """,
                fact_params,
            ).fetchall()
            rows = fact_rows

    rows = _rerank_repo_rows(rows, query, limit=result_limit)

    return [
        RepoSearchResult(
            artifact_ref=f"segment:{row['artifact_id']}" if "artifact_id" in row.keys() and row["segment_type"] != "metadata" and row["segment_type"] != "address" else f"fact:{row['artifact_id']}",
            repo_name=row["repo_name"],
            repo_ref=row["repo_ref"],
            path=row["path"],
            segment_type=row["segment_type"],
            title=row["title"],
            snippet=row["snippet"][: config.REPO_CONTEXT_MAX_SNIPPET_CHARS],
            product_tag=row["product_tag"],
            authority_tag=row["authority_tag"],
            legacy=bool(row["legacy"]),
            score=float(row["rank_score"]),
        )
        for row in rows
    ]


def format_repo_search_results(results: Iterable[RepoSearchResult]) -> str:
    parts = []
    artifact_refs: list[str] = []
    for result in results:
        legacy_text = " legacy=true" if result.legacy else ""
        artifact_refs.append(result.artifact_ref)
        parts.append(
            "\n".join(
                [
                    f"Artifact: {result.artifact_ref}",
                    f"Repo Source: {result.repo_name}@{result.repo_ref}",
                    f"Path: {result.path}",
                    f"Type: {result.segment_type} [{result.authority_tag}/{result.product_tag}{legacy_text}]",
                    f"Title: {result.title}",
                    f"Content:\n{result.snippet}",
                ]
            )
        )
    if not parts:
        return "No matching repo context was found."
    recommended_refs = ", ".join(artifact_refs[:3])
    guidance = (
        f"Recommended next step: call fetch_repo_artifacts_tool with one or more of these refs before searching again: {recommended_refs}."
        if recommended_refs
        else "Recommended next step: answer or escalate based on the retrieved evidence."
    )
    return "\n\n---\n\n".join(parts) + f"\n\n{guidance}"


def _parse_artifact_ref(artifact_ref: str) -> tuple[str, int]:
    if ":" not in artifact_ref:
        raise ValueError(f"Invalid artifact ref '{artifact_ref}'. Expected format '<kind>:<id>'.")
    kind, raw_id = artifact_ref.split(":", 1)
    if kind not in {"segment", "fact"}:
        raise ValueError(f"Unsupported artifact kind '{kind}'.")
    try:
        artifact_id = int(raw_id)
    except ValueError as exc:
        raise ValueError(f"Invalid artifact id '{raw_id}'.") from exc
    return kind, artifact_id


def _extract_file_excerpt(content: str, anchor: str, max_chars: int = 3200) -> str:
    normalized_content = content.strip()
    if not normalized_content:
        return ""
    if not anchor:
        return normalized_content[:max_chars]
    anchor_idx = normalized_content.find(anchor.strip())
    if anchor_idx == -1:
        return normalized_content[:max_chars]
    half_window = max_chars // 2
    start = max(anchor_idx - half_window, 0)
    end = min(anchor_idx + len(anchor) + half_window, len(normalized_content))
    excerpt = normalized_content[start:end]
    if start > 0:
        excerpt = "...\n" + excerpt
    if end < len(normalized_content):
        excerpt = excerpt + "\n..."
    return excerpt


def fetch_repo_artifacts(
    artifact_refs: Iterable[str],
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    status = get_repo_context_status(db_path=db_path, enabled=True)
    if status["state"] not in {"ready", "available"}:
        return []

    normalized_refs = []
    for artifact_ref in artifact_refs:
        stripped = artifact_ref.strip()
        if stripped and stripped not in normalized_refs:
            normalized_refs.append(stripped)

    artifacts: list[dict[str, Any]] = []
    with _connect(db_path) as conn:
        for artifact_ref in normalized_refs:
            kind, artifact_id = _parse_artifact_ref(artifact_ref)
            if kind == "segment":
                row = conn.execute(
                    """
                    SELECT
                        segments.id AS artifact_id,
                        segments.repo_name,
                        segments.repo_ref,
                        segments.path,
                        segments.segment_type,
                        segments.title,
                        segments.snippet,
                        segments.product_tag,
                        segments.authority_tag,
                        segments.legacy,
                        files.content
                    FROM segments
                    JOIN files ON files.id = segments.file_id
                    WHERE segments.id = ?
                    """,
                    (artifact_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT
                        facts.id AS artifact_id,
                        facts.repo_name,
                        facts.repo_ref,
                        facts.path,
                        facts.fact_type AS segment_type,
                        facts.fact_key AS title,
                        facts.fact_value AS snippet,
                        facts.product_tag,
                        facts.authority_tag,
                        facts.legacy,
                        files.content
                    FROM facts
                    JOIN files ON files.id = facts.file_id
                    WHERE facts.id = ?
                    """,
                    (artifact_id,),
                ).fetchone()

            if not row:
                continue

            artifacts.append(
                {
                    "artifact_ref": f"{kind}:{row['artifact_id']}",
                    "repo_name": row["repo_name"],
                    "repo_ref": row["repo_ref"],
                    "path": row["path"],
                    "segment_type": row["segment_type"],
                    "title": row["title"],
                    "snippet": row["snippet"][: config.REPO_CONTEXT_MAX_SNIPPET_CHARS],
                    "product_tag": row["product_tag"],
                    "authority_tag": row["authority_tag"],
                    "legacy": bool(row["legacy"]),
                    "file_excerpt": _extract_file_excerpt(row["content"] or "", row["snippet"]),
                }
            )
    return artifacts


def format_repo_artifacts(artifacts: Iterable[dict[str, Any]]) -> str:
    parts = []
    for artifact in artifacts:
        legacy_text = " legacy=true" if artifact["legacy"] else ""
        parts.append(
            "\n".join(
                [
                    f"Artifact: {artifact['artifact_ref']}",
                    f"Repo Source: {artifact['repo_name']}@{artifact['repo_ref']}",
                    f"Path: {artifact['path']}",
                    f"Type: {artifact['segment_type']} [{artifact['authority_tag']}/{artifact['product_tag']}{legacy_text}]",
                    f"Title: {artifact['title']}",
                    f"Matched Content:\n{artifact['snippet']}",
                    f"File Excerpt:\n{artifact['file_excerpt']}",
                ]
            )
        )
    if not parts:
        return "No repo artifacts were found for the requested references."
    return "\n\n---\n\n".join(parts)
