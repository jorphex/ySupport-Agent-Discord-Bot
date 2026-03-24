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
