import json
from typing import Any
from urllib import request

import config


DISCORD_API_BASE = "https://discord.com/api/v10"


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
    with request.urlopen(req, timeout=30) as response:
        if response.status == 204:
            return None
        return json.load(response)


def discord_get_json(url: str) -> Any:
    return _discord_request_json("GET", url)


def discord_post_json(url: str, payload: Any) -> Any:
    return _discord_request_json("POST", url, payload)
