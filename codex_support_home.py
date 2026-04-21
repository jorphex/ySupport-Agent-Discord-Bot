from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import shutil
import socket
import subprocess
from urllib.parse import urlparse
from typing import Any


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
    repo_root: str | Path | None = None,
    mcp_container_name: str | None = None,
    auth_source: str | Path | None = None,
    web_search_mode: str = "live",
) -> CodexSupportHome:
    home_dir = Path(codex_home)
    home_dir.mkdir(parents=True, exist_ok=True)

    config_path = home_dir / "config.toml"
    auth_path = home_dir / "auth.json"
    instructions_path = home_dir / "ysupport_instructions.md"
    shutil.copy2(_instructions_template_path(), instructions_path)
    repo_root_path = Path(repo_root) if repo_root else None
    normalized_ysupport_mcp_url = ysupport_mcp_url.strip()
    normalized_mcp_server_api_key = mcp_server_api_key.strip()
    use_http_mcp = bool(
        normalized_ysupport_mcp_url
        and normalized_mcp_server_api_key
        and _is_http_url_reachable(normalized_ysupport_mcp_url)
    )
    stdio_launcher = None
    if not use_http_mcp:
        stdio_launcher = _choose_ysupport_stdio_launcher(
            repo_root=repo_root_path,
            mcp_server_api_key=mcp_server_api_key,
            mcp_container_name=mcp_container_name,
        )
    ysupport_mcp_enabled = use_http_mcp or stdio_launcher is not None

    config_path.write_text(
        build_codex_support_config_toml(
            instructions_path=instructions_path,
            stdio_launcher=stdio_launcher,
            ysupport_mcp_url=normalized_ysupport_mcp_url if use_http_mcp else "",
            mcp_server_api_key=normalized_mcp_server_api_key if ysupport_mcp_enabled else "",
            web_search_mode=web_search_mode,
        ),
        encoding="utf-8",
    )

    if auth_source:
        source_path = Path(auth_source)
        if source_path.exists():
            shutil.copy2(source_path, auth_path)

    return CodexSupportHome(
        home_dir=home_dir,
        config_path=config_path,
        auth_path=auth_path,
        instructions_path=instructions_path,
        ysupport_mcp_enabled=ysupport_mcp_enabled,
    )


def build_codex_support_config_toml(
    *,
    instructions_path: str | Path,
    stdio_launcher: dict[str, Any] | None = None,
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
    use_stdio_mcp = stdio_launcher is not None
    return "\n".join(
        _codex_support_config_lines(
            quoted_instructions_path=quoted_instructions_path,
            quoted_web_search_mode=quoted_web_search_mode,
            stdio_launcher=stdio_launcher,
            quoted_url=quoted_url,
            quoted_api_key=quoted_api_key,
            ysupport_mcp_enabled=use_stdio_mcp or bool(
                normalized_ysupport_mcp_url and normalized_mcp_server_api_key
            ),
            use_stdio_mcp=use_stdio_mcp,
            quoted_mcp_server_api_key=_toml_string(normalized_mcp_server_api_key),
        )
    )


def _codex_support_config_lines(
    *,
    quoted_instructions_path: str,
    quoted_web_search_mode: str,
    stdio_launcher: dict[str, Any] | None,
    quoted_url: str,
    quoted_api_key: str,
    ysupport_mcp_enabled: bool,
    use_stdio_mcp: bool,
    quoted_mcp_server_api_key: str,
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
            "",
            "[tools]",
            "view_image = false",
        ]
    if ysupport_mcp_enabled:
        if use_stdio_mcp:
            assert stdio_launcher is not None
            lines.extend(
                [
                    "",
                    "[mcp_servers.ysupport]",
                    "enabled = true",
                    f'command = {_toml_string(str(stdio_launcher["command"]))}',
                    f'args = {_toml_string_array(stdio_launcher["args"])}',
                    "startup_timeout_sec = 20",
                    "tool_timeout_sec = 120",
                    "",
                ]
            )
            if stdio_launcher.get("cwd"):
                lines.append(f'cwd = {_toml_string(str(stdio_launcher["cwd"]))}')
            if stdio_launcher.get("env_vars"):
                lines.append(
                    f'env_vars = {_toml_string_array(stdio_launcher["env_vars"])}'
                )
            if stdio_launcher.get("env"):
                lines.extend(["", "[mcp_servers.ysupport.env]"])
                for key, value in stdio_launcher["env"].items():
                    lines.append(f"{key} = {_toml_string(str(value))}")
                lines.append("")
        else:
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
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_string_array(values: list[str]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


def _instructions_template_path() -> Path:
    return Path(__file__).resolve().with_name("ysupport_codex_instructions.md")


def _is_http_url_reachable(url: str, *, timeout_seconds: float = 1.0) -> bool:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return True
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def _choose_ysupport_stdio_launcher(
    *,
    repo_root: Path | None,
    mcp_server_api_key: str,
    mcp_container_name: str | None,
) -> dict[str, Any] | None:
    if repo_root is not None and (repo_root / "mcp_server.py").exists() and _local_python_has_mcp():
        return {
            "command": "python3",
            "args": ["mcp_server.py"],
            "cwd": str(repo_root),
            "env_vars": [
                "OPENAI_API_KEY",
                "PINECONE_API_KEY",
                "ALCHEMY_KEY",
                "PINECONE_INDEX_NAME",
                "YEARN_PINECONE_NAMESPACE",
            ],
            "env": {
                "MCP_TRANSPORT": "stdio",
                "MCP_SERVER_API_KEY": mcp_server_api_key,
            },
        }
    if mcp_container_name and _docker_container_running(mcp_container_name):
        return {
            "command": "docker",
            "args": [
                "exec",
                "-i",
                mcp_container_name,
                "env",
                "MCP_TRANSPORT=stdio",
                f"MCP_SERVER_API_KEY={mcp_server_api_key}",
                "python",
                "mcp_server.py",
            ],
            "env": {},
            "env_vars": [],
        }
    return None


def _local_python_has_mcp() -> bool:
    return importlib.util.find_spec("mcp") is not None


def _docker_container_running(container_name: str, *, timeout_seconds: float = 2.0) -> bool:
    try:
        completed = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{.State.Running}}",
                container_name,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and completed.stdout.strip() == "true"
