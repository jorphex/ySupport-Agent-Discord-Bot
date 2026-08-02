from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import aiohttp

from codex_support_contract import SupportTurnRequest


_ALLOWED_DISCORD_ATTACHMENT_HOSTS = {
    "cdn.discordapp.com",
    "media.discordapp.net",
}
_MAX_ATTACHMENT_IMAGE_BYTES = 20 * 1024 * 1024


async def prepare_support_request_attachments(
    support_request: SupportTurnRequest,
    *,
    run_dir: str | Path,
) -> None:
    attachments = list(support_request.attachments)
    if not attachments:
        return
    attachments_dir = Path(run_dir) / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[dict[str, object]] = []
    for index, attachment in enumerate(attachments, start=1):
        item = dict(attachment)
        item["local_path"] = None
        if not attachment_is_image(item):
            prepared.append(item)
            continue
        try:
            local_path = await _download_attachment_image(
                attachment=item,
                attachments_dir=attachments_dir,
                index=index,
            )
        except Exception as exc:
            filename = str(item.get("filename") or f"attachment {index}")
            raise ValueError(
                f"Could not prepare image attachment {filename}: {exc}"
            ) from exc
        item["is_image"] = True
        item["local_path"] = str(local_path)
        prepared.append(item)
    support_request.attachments = prepared


def image_attachment_paths(support_request: SupportTurnRequest) -> list[Path]:
    image_paths: list[Path] = []
    for attachment in support_request.attachments:
        if not attachment_is_image(attachment):
            continue
        local_path = str(attachment.get("local_path") or "").strip()
        if not local_path:
            continue
        path = Path(local_path)
        if path.exists():
            image_paths.append(path)
    return image_paths


def attachment_is_image(attachment: dict[str, object]) -> bool:
    if bool(attachment.get("is_image")):
        return True
    content_type = str(attachment.get("content_type") or "").lower()
    if content_type.startswith("image/"):
        return True
    filename = str(attachment.get("filename") or "").lower()
    return filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"))


def _validate_attachment_image_source(attachment: dict[str, object]) -> str:
    url = str(attachment.get("url") or "").strip()
    if not url:
        raise ValueError("Attachment URL is required.")
    parsed = urlparse(url)
    if (
        parsed.scheme.lower() != "https"
        or (parsed.hostname or "").lower() not in _ALLOWED_DISCORD_ATTACHMENT_HOSTS
    ):
        raise ValueError("Attachment URL must use the Discord CDN.")
    declared_size = attachment.get("size")
    if isinstance(declared_size, int) and declared_size > _MAX_ATTACHMENT_IMAGE_BYTES:
        raise ValueError("Attachment image exceeds the 20 MiB safety limit.")
    return url


async def _read_attachment_image_body(response: aiohttp.ClientResponse) -> bytes:
    content_length = response.content_length
    if content_length is not None and content_length > _MAX_ATTACHMENT_IMAGE_BYTES:
        raise ValueError("Attachment image exceeds the 20 MiB safety limit.")
    body = bytearray()
    async for chunk in response.content.iter_chunked(64 * 1024):
        body.extend(chunk)
        if len(body) > _MAX_ATTACHMENT_IMAGE_BYTES:
            raise ValueError("Attachment image exceeds the 20 MiB safety limit.")
    return bytes(body)


async def _download_attachment_image(
    *,
    attachment: dict[str, object],
    attachments_dir: Path,
    index: int,
) -> Path:
    url = _validate_attachment_image_source(attachment)
    source_name = str(attachment.get("filename") or "").strip()
    suffix = Path(source_name).suffix or Path(urlparse(url).path).suffix or ".img"
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    target_path = attachments_dir / f"attachment_{index}{safe_suffix.lower()}"
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as response:
            response.raise_for_status()
            response_host = (response.url.host or "").lower()
            if response_host not in _ALLOWED_DISCORD_ATTACHMENT_HOSTS:
                raise ValueError("Attachment redirect left the Discord CDN.")
            body = await _read_attachment_image_body(response)
            content_type = (response.headers.get("Content-Type") or "").strip().lower()
    if not body:
        raise ValueError("Attachment download returned empty body.")
    if not content_type.startswith("image/"):
        raise ValueError("Attachment is not an image.")
    target_path.write_bytes(body)
    return target_path
