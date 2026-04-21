from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Any


_SESSION_ID_RE = re.compile(
    r"session id:\s*(?P<session_id>[0-9a-fA-F-]{36})",
    re.IGNORECASE,
)


@dataclass
class CodexSupportSessionRecord:
    conversation_key: str
    session_id: str
    created_at_utc: str
    updated_at_utc: str
    run_count: int
    last_artifact_dir: str | None = None
    last_error: str | None = None
    last_error_at_utc: str | None = None


class CodexSupportSessionManager:
    def __init__(
        self,
        root_dir: str | Path,
        *,
        max_age_hours: int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = (
            timedelta(hours=max_age_hours)
            if max_age_hours is not None and max_age_hours > 0
            else None
        )

    def load(self, conversation_key: str) -> CodexSupportSessionRecord | None:
        path = self._record_path(conversation_key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        record = CodexSupportSessionRecord(**payload)
        if self._is_expired(record):
            self.reset(conversation_key)
            return None
        return record

    def record_success(
        self,
        *,
        conversation_key: str,
        session_id: str,
        artifact_dir: str | None = None,
    ) -> CodexSupportSessionRecord:
        now = _utc_now_iso()
        existing = self.load(conversation_key)
        if existing is None or existing.session_id != session_id:
            record = CodexSupportSessionRecord(
                conversation_key=conversation_key,
                session_id=session_id,
                created_at_utc=now,
                updated_at_utc=now,
                run_count=1,
                last_artifact_dir=artifact_dir,
            )
        else:
            record = CodexSupportSessionRecord(
                conversation_key=existing.conversation_key,
                session_id=existing.session_id,
                created_at_utc=existing.created_at_utc,
                updated_at_utc=now,
                run_count=existing.run_count + 1,
                last_artifact_dir=artifact_dir or existing.last_artifact_dir,
                last_error=None,
                last_error_at_utc=None,
            )
        self._write(record)
        return record

    def record_failure(
        self,
        *,
        conversation_key: str,
        error_text: str,
        artifact_dir: str | None = None,
    ) -> CodexSupportSessionRecord | None:
        existing = self.load(conversation_key)
        if existing is None:
            return None
        failed = CodexSupportSessionRecord(
            conversation_key=existing.conversation_key,
            session_id=existing.session_id,
            created_at_utc=existing.created_at_utc,
            updated_at_utc=existing.updated_at_utc,
            run_count=existing.run_count,
            last_artifact_dir=artifact_dir or existing.last_artifact_dir,
            last_error=error_text,
            last_error_at_utc=_utc_now_iso(),
        )
        self._write(failed)
        return failed

    def reset(self, conversation_key: str) -> None:
        path = self._record_path(conversation_key)
        path.unlink(missing_ok=True)

    def inspect(self, conversation_key: str) -> dict[str, Any]:
        record = self.load(conversation_key)
        return {
            "conversation_key": conversation_key,
            "active": record is not None,
            "record": asdict(record) if record is not None else None,
        }

    def conversation_key_for_request(self, request) -> str:
        channel_type = "public" if request.run_context.get("is_public_trigger") else "ticket"
        channel_id = request.run_context.get("channel_id")
        if channel_id is not None:
            return f"{channel_type}:{channel_id}"
        return f"{channel_type}:{request.workflow_name}"

    def extract_session_id(self, stderr_text: str) -> str | None:
        match = _SESSION_ID_RE.search(stderr_text or "")
        if not match:
            return None
        return match.group("session_id")

    def _record_path(self, conversation_key: str) -> Path:
        safe_key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", conversation_key)
        return self.root_dir / f"{safe_key}.json"

    def _write(self, record: CodexSupportSessionRecord) -> None:
        self._record_path(record.conversation_key).write_text(
            json.dumps(asdict(record), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _is_expired(self, record: CodexSupportSessionRecord) -> bool:
        if self.max_age is None:
            return False
        try:
            updated_at = datetime.fromisoformat(record.updated_at_utc)
        except ValueError:
            return False
        return datetime.now(timezone.utc) - updated_at > self.max_age


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
