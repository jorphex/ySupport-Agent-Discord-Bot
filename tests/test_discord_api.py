import io
import json
import unittest
from urllib import error

import discord_api


class _FakeJsonResponse:
    def __init__(self, payload, *, status: int = 200) -> None:
        self.status = status
        self._body = io.StringIO(json.dumps(payload))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, *args, **kwargs):
        return self._body.read(*args, **kwargs)


class DiscordApiTests(unittest.TestCase):
    def test_discord_request_retries_after_rate_limit(self) -> None:
        rate_limited_body = io.BytesIO(json.dumps({"retry_after": 0.01}).encode("utf-8"))
        rate_limited_error = error.HTTPError(
            "https://discord.com/api/v10/test",
            429,
            "Too Many Requests",
            hdrs=None,
            fp=rate_limited_body,
        )
        success_response = _FakeJsonResponse({"ok": True})

        with (
            unittest.mock.patch.object(discord_api.config, "DISCORD_BOT_TOKEN", "test-token"),
            unittest.mock.patch(
                "discord_api.request.urlopen",
                side_effect=[rate_limited_error, success_response],
            ),
            unittest.mock.patch("discord_api.time.sleep") as mock_sleep,
        ):
            payload = discord_api.discord_get_json("https://discord.com/api/v10/test")

        self.assertEqual(payload, {"ok": True})
        mock_sleep.assert_called_once_with(0.01)

    def test_discord_request_retries_after_transient_url_error(self) -> None:
        success_response = _FakeJsonResponse({"ok": True})

        with (
            unittest.mock.patch.object(discord_api.config, "DISCORD_BOT_TOKEN", "test-token"),
            unittest.mock.patch(
                "discord_api.request.urlopen",
                side_effect=[error.URLError("temporary failure"), success_response],
            ),
            unittest.mock.patch("discord_api.time.sleep") as mock_sleep,
        ):
            payload = discord_api.discord_get_json("https://discord.com/api/v10/test")

        self.assertEqual(payload, {"ok": True})
        mock_sleep.assert_called_once_with(1.0)

    def test_discord_request_raises_normalized_error_after_final_http_failure(self) -> None:
        not_found_body = io.BytesIO(json.dumps({"message": "Unknown Channel"}).encode("utf-8"))
        not_found_error = error.HTTPError(
            "https://discord.com/api/v10/test",
            404,
            "Not Found",
            hdrs=None,
            fp=not_found_body,
        )

        with (
            unittest.mock.patch.object(discord_api.config, "DISCORD_BOT_TOKEN", "test-token"),
            unittest.mock.patch(
                "discord_api.request.urlopen",
                side_effect=not_found_error,
            ),
        ):
            with self.assertRaises(discord_api.DiscordApiError) as cm:
                discord_api.discord_get_json("https://discord.com/api/v10/test")

        self.assertEqual(cm.exception.status_code, 404)
        self.assertIn("Unknown Channel", str(cm.exception))
