import json
import time
from typing import Any
from urllib import error, request

import config


DISCORD_API_BASE = "https://discord.com/api/v10"
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3
_DEFAULT_RETRY_DELAY_SECONDS = 1.0


class DiscordApiError(RuntimeError):
    def __init__(self, message: str, *, url: str, status_code: int | None = None) -> None:
        self.url = url
        self.status_code = status_code
        super().__init__(message)


def _read_json_body(response) -> Any:
    if response.status == 204:
        return None
    return json.load(response)


def _extract_retry_after_seconds(exc: error.HTTPError) -> float:
    try:
        payload = json.load(exc)
    except Exception:
        return _DEFAULT_RETRY_DELAY_SECONDS

    retry_after = payload.get("retry_after")
    if isinstance(retry_after, (int, float)) and retry_after > 0:
        return float(retry_after)
    return _DEFAULT_RETRY_DELAY_SECONDS


def _format_http_error(exc: error.HTTPError, *, url: str) -> DiscordApiError:
    try:
        payload = json.load(exc)
    except Exception:
        payload = None

    if isinstance(payload, dict):
        message = str(payload.get("message") or f"Discord API request failed with status {exc.code}.")
    else:
        message = f"Discord API request failed with status {exc.code}."
    return DiscordApiError(message, url=url, status_code=exc.code)


def _discord_request_json(method: str, url: str, payload: Any | None = None) -> Any:
    if not config.DISCORD_BOT_TOKEN:
        raise ValueError("DISCORD_BOT_TOKEN is required for Discord API access.")

    headers = {
        "Authorization": f"Bot {config.DISCORD_BOT_TOKEN}",
        "User-Agent": "ysupport-discord-api-client",
    }
    body: bytes | None = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")

    req = request.Request(url, data=body, headers=headers, method=method)
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with request.urlopen(req, timeout=30) as response:
                return _read_json_body(response)
        except error.HTTPError as exc:
            if exc.code in _RETRYABLE_STATUS_CODES and attempt < _MAX_ATTEMPTS:
                delay_seconds = (
                    _extract_retry_after_seconds(exc)
                    if exc.code == 429
                    else _DEFAULT_RETRY_DELAY_SECONDS
                )
                time.sleep(delay_seconds)
                continue
            raise _format_http_error(exc, url=url) from exc
        except error.URLError as exc:
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_DEFAULT_RETRY_DELAY_SECONDS)
                continue
            raise DiscordApiError(
                f"Discord API request failed: {exc.reason}",
                url=url,
            ) from exc


def discord_get_json(url: str) -> Any:
    return _discord_request_json("GET", url)


def discord_post_json(url: str, payload: Any) -> Any:
    return _discord_request_json("POST", url, payload)
