from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4


@dataclass(frozen=True)
class CodexSupportHome:
    home_dir: Path
    config_path: Path
    auth_path: Path
    instructions_path: Path
    ysupport_mcp_enabled: bool


def prepare_codex_support_home(
    *,
    codex_home: str | Path,
    ysupport_mcp_url: str,
    mcp_server_api_key: str,
    auth_link_source: str | Path | None = None,
    web_search_mode: str = "live",
) -> CodexSupportHome:
    home_dir = Path(codex_home)
    home_dir.mkdir(parents=True, exist_ok=True)

    config_path = home_dir / "config.toml"
    auth_path = home_dir / "auth.json"
    instructions_path = home_dir / "ysupport_instructions.md"
    normalized_ysupport_mcp_url = ysupport_mcp_url.strip()
    normalized_mcp_server_api_key = mcp_server_api_key.strip()
    if bool(normalized_ysupport_mcp_url) != bool(normalized_mcp_server_api_key):
        raise ValueError("ySupport MCP requires both its HTTP URL and bearer key.")
    ysupport_mcp_enabled = bool(
        normalized_ysupport_mcp_url and normalized_mcp_server_api_key
    )
    if ysupport_mcp_enabled:
        _validate_http_mcp_url(normalized_ysupport_mcp_url)

    _atomic_write_text(
        instructions_path,
        _instructions_template_path().read_text(encoding="utf-8"),
    )
    _atomic_write_text(
        config_path,
        build_codex_support_config_toml(
            instructions_path=instructions_path,
            ysupport_mcp_url=normalized_ysupport_mcp_url,
            mcp_server_api_key=normalized_mcp_server_api_key
            if ysupport_mcp_enabled
            else "",
            web_search_mode=web_search_mode,
        ),
    )

    if auth_link_source:
        prepare_codex_auth_link(
            home_auth_path=auth_path,
            auth_link_source_path=Path(auth_link_source),
        )

    return CodexSupportHome(
        home_dir=home_dir,
        config_path=config_path,
        auth_path=auth_path,
        instructions_path=instructions_path,
        ysupport_mcp_enabled=ysupport_mcp_enabled,
    )


def prepare_codex_auth_link(
    *,
    home_auth_path: Path,
    auth_link_source_path: Path,
) -> Path:
    if not auth_link_source_path.is_file():
        raise FileNotFoundError(
            f"Codex auth link source is not a readable file: {auth_link_source_path}"
        )
    source_resolved = auth_link_source_path.resolve(strict=True)
    home_auth_path.parent.mkdir(parents=True, exist_ok=True)
    if home_auth_path.is_symlink():
        current_target = home_auth_path.resolve(strict=False)
        if current_target == source_resolved:
            return auth_link_source_path
    elif home_auth_path.exists():
        if home_auth_path.resolve(strict=False) == source_resolved:
            return auth_link_source_path
    temporary_link = home_auth_path.with_name(
        f".{home_auth_path.name}.{uuid4().hex}.tmp"
    )
    try:
        temporary_link.symlink_to(source_resolved)
        os.replace(temporary_link, home_auth_path)
    finally:
        temporary_link.unlink(missing_ok=True)
    return auth_link_source_path


def _atomic_write_text(path: Path, text: str) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(text, encoding="utf-8")
        temporary_path.chmod(0o600)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_codex_support_config_toml(
    *,
    instructions_path: str | Path,
    ysupport_mcp_url: str,
    mcp_server_api_key: str,
    web_search_mode: str = "live",
) -> str:
    normalized_ysupport_mcp_url = ysupport_mcp_url.strip()
    normalized_mcp_server_api_key = mcp_server_api_key.strip()
    normalized_web_search_mode = web_search_mode.strip()
    quoted_instructions_path = _toml_string(str(instructions_path))
    quoted_url = _toml_string(normalized_ysupport_mcp_url)
    quoted_api_key = _toml_string(f"Bearer {normalized_mcp_server_api_key}")
    quoted_web_search_mode = _toml_string(normalized_web_search_mode)
    return "\n".join(
        _codex_support_config_lines(
            quoted_instructions_path=quoted_instructions_path,
            quoted_web_search_mode=quoted_web_search_mode,
            quoted_url=quoted_url,
            quoted_api_key=quoted_api_key,
            ysupport_mcp_enabled=bool(
                normalized_ysupport_mcp_url and normalized_mcp_server_api_key
            ),
        )
    )


def _codex_support_config_lines(
    *,
    quoted_instructions_path: str,
    quoted_web_search_mode: str,
    quoted_url: str,
    quoted_api_key: str,
    ysupport_mcp_enabled: bool,
) -> list[str]:
    lines = [
        'approval_policy = "never"',
        'sandbox_mode = "danger-full-access"',
        "allow_login_shell = false",
        'cli_auth_credentials_store = "file"',
        f"model_instructions_file = {quoted_instructions_path}",
        f"web_search = {quoted_web_search_mode}",
        "",
        "[history]",
        'persistence = "none"',
        "",
        "[features]",
        "apps = false",
        "multi_agent = false",
        "shell_tool = true",
    ]
    if ysupport_mcp_enabled:
        lines.extend(
            [
                "",
                "[mcp_servers.ysupport]",
                "enabled = true",
                f"url = {quoted_url}",
                "",
                "[mcp_servers.ysupport.http_headers]",
                f"Authorization = {quoted_api_key}",
                "",
            ]
        )
    return lines


def _toml_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace("\b", "\\b")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\f", "\\f")
        .replace("\r", "\\r")
        .replace('"', '\\"')
    )
    return f'"{escaped}"'


def _instructions_template_path() -> Path:
    return Path(__file__).resolve().with_name("ysupport_codex_instructions.md")


def _validate_http_mcp_url(url: str) -> None:
    parsed = urlparse(url)
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("ySupport MCP URL has an invalid port.") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("ySupport MCP URL must be an absolute HTTP(S) URL.")
    if parsed.username or parsed.password:
        raise ValueError("ySupport MCP URL must not contain credentials.")
