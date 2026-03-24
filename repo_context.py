from __future__ import annotations
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import config
from repo_context_build import (
    REPO_CONTEXT_SCHEMA_VERSION,
    _BUILD_META_KEY_PREFIX,
    _ensure_schema,
    _get_build_meta,
    build_repo_context_index,
    load_repo_manifest,
    manifest_hash,
    validate_repo_manifest,
)


__all__ = [
    "RepoSearchResult",
    "build_repo_context_index",
    "load_repo_manifest",
    "manifest_hash",
    "validate_repo_manifest",
    "get_repo_context_status",
    "search_repo_context",
    "format_repo_search_results",
    "fetch_repo_artifacts",
    "format_repo_artifacts",
]


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
