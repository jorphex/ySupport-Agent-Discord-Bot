"""Local production counters and UTC monthly reports."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
from pathlib import Path
import re
import sqlite3
from tempfile import NamedTemporaryFile

from ysupport import config

logger = logging.getLogger(__name__)

_REPLY_METRICS = ("ticket_auto_replies", "ticket_staff_replies", "public_replies")
_COUNT_METRICS = (
    "tickets_opened",
    "tickets_opted_in",
    "tickets_answered",
    *_REPLY_METRICS,
    "public_requests",
    *(
        f"{source}_handoff_{outcome}"
        for source in ("ticket", "public")
        for outcome in ("notices_sent", "notices_failed", "replies_delivered", "dismissed", "delivery_failed")
    ),
    "public_handoff_no_update",
)
_LATENCY_METRICS = (*_REPLY_METRICS, "codex_queue", "codex_execution")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS recording (
    id INTEGER PRIMARY KEY CHECK (id = 1), started_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS counts (
    month TEXT NOT NULL, metric TEXT NOT NULL, count INTEGER NOT NULL,
    PRIMARY KEY (month, metric)
);
CREATE TABLE IF NOT EXISTS latency (
    month TEXT NOT NULL, metric TEXT NOT NULL, milliseconds INTEGER NOT NULL,
    count INTEGER NOT NULL, PRIMARY KEY (month, metric, milliseconds)
);
CREATE TABLE IF NOT EXISTS ticket_seen (
    ticket_key BLOB NOT NULL, month TEXT NOT NULL, metric TEXT NOT NULL,
    PRIMARY KEY (ticket_key, month, metric)
);
CREATE TABLE IF NOT EXISTS handoff_seen (
    nonce TEXT PRIMARY KEY, channel_key BLOB NOT NULL
);
"""


def _directory() -> Path:
    return config.TICKET_EXECUTION_STATE_ROOT / "support-stats"


def _ticket_key(ticket_id: int) -> bytes:
    return sha256(str(ticket_id).encode("ascii")).digest()


def _increment(db: sqlite3.Connection, month: str, metric: str, ticket_id: int | None) -> None:
    if ticket_id is not None:
        inserted = db.execute(
            "INSERT INTO ticket_seen VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
            (_ticket_key(ticket_id), month, metric),
        )
        if not inserted.rowcount:
            return
    db.execute(
        "INSERT INTO counts VALUES (?, ?, 1) ON CONFLICT DO UPDATE SET count = count + 1",
        (month, metric),
    )


def _sample(db: sqlite3.Connection, month: str, metric: str, milliseconds: int) -> None:
    db.execute(
        "INSERT INTO latency VALUES (?, ?, ?, 1) ON CONFLICT DO UPDATE SET count = count + 1",
        (month, metric, milliseconds),
    )


class _Recorder:
    def __init__(self, directory: Path):
        self.directory = directory
        self.database = directory / "stats.sqlite3"
        self._month = None
        self._pending_ticket_deletions: set[bytes] = set()

    def _deletion_path(self, key: bytes) -> Path:
        return self.directory / f"deleted-ticket-{key.hex()}"

    def restore_deletions(self) -> None:
        try:
            for path in self.directory.glob("deleted-ticket-*"):
                if match := re.fullmatch(r"deleted-ticket-([0-9a-f]{64})", path.name):
                    self._pending_ticket_deletions.add(bytes.fromhex(match[1]))
        except OSError:
            logger.exception("Could not read pending ySupport statistics deletions.")

    def forget_ticket(self, ticket_id: int, now: datetime) -> None:
        key = _ticket_key(ticket_id)
        self._pending_ticket_deletions.add(key)
        # A busy SQLite database must not prevent saving the deletion intent.
        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._deletion_path(key).touch(mode=0o600, exist_ok=True)
        except OSError:
            logger.exception("Could not save a ySupport statistics deletion for retry after restart.")
        self.write(now)

    def maintain(self, now: datetime) -> None:
        if self._pending_ticket_deletions or self._month != now.strftime("%Y-%m"):
            self.write(now)

    def write(
        self,
        now: datetime,
        *,
        counts: tuple[tuple[str, int | None], ...] = (),
        samples: tuple[tuple[str, int], ...] = (),
        handoff: tuple[str, str, int] | None = None,
        settled_handoff: str | None = None,
    ) -> None:
        month = now.strftime("%Y-%m")
        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            # A concurrent report or storage fault must not hold up support.
            with closing(sqlite3.connect(self.database, timeout=0.05)) as db:
                db.execute("PRAGMA secure_delete = ON")
                if self._month is None:
                    db.executescript(_SCHEMA)
                with db:
                    if self._month is None:
                        db.execute(
                            "INSERT INTO recording VALUES (1, ?) ON CONFLICT DO NOTHING",
                            (now.isoformat(timespec="seconds"),),
                        )
                    if self._month != month:
                        db.execute("DELETE FROM ticket_seen WHERE month < ?", (month,))
                    if handoff is not None:
                        metric, nonce, channel_id = handoff
                        inserted = db.execute(
                            "INSERT INTO handoff_seen VALUES (?, ?) ON CONFLICT DO NOTHING",
                            (nonce, _ticket_key(channel_id)),
                        )
                        if inserted.rowcount:
                            _increment(db, month, metric, None)
                    if settled_handoff is not None:
                        db.execute("DELETE FROM handoff_seen WHERE nonce = ?", (settled_handoff,))
                    for metric, ticket_id in counts:
                        _increment(db, month, metric, ticket_id)
                    for metric, milliseconds in samples:
                        _sample(db, month, metric, milliseconds)
                    for key in self._pending_ticket_deletions:
                        db.execute("DELETE FROM ticket_seen WHERE ticket_key = ?", (key,))
                        db.execute("DELETE FROM handoff_seen WHERE channel_key = ?", (key,))
                self._month = month
        except (OSError, sqlite3.Error):
            logger.exception("Could not write ySupport statistics.")
            return
        for key in tuple(self._pending_ticket_deletions):
            try:
                self._deletion_path(key).unlink(missing_ok=True)
            except OSError:
                logger.exception("Could not clear a completed ySupport statistics deletion marker.")
            else:
                self._pending_ticket_deletions.remove(key)


_recorder: _Recorder | None = None


def enable_recording() -> None:
    global _recorder
    _recorder = _Recorder(_directory())
    _recorder.restore_deletions()
    _recorder.write(_now())


def _now() -> datetime:
    return datetime.now(timezone.utc)


def record(metric: str, *, ticket_id: int | None = None) -> None:
    if _recorder is None:
        return
    _recorder.write(_now(), counts=((metric, ticket_id),))


def forget_ticket(ticket_id: int) -> None:
    if _recorder is not None:
        _recorder.forget_ticket(ticket_id, _now())


def maintain() -> None:
    if _recorder is not None:
        _recorder.maintain(_now())


def record_handoff_outcome(metric: str, nonce: str, channel_id: int) -> None:
    # The nonce survives a retry from posting until terminal state is durable.
    if _recorder is not None:
        _recorder.write(_now(), handoff=(metric, nonce, channel_id))


def settle_handoff_outcome(nonce: str) -> None:
    if _recorder is not None:
        _recorder.write(_now(), settled_handoff=nonce)


def record_reply(metric: str, started_at: datetime | None, *, ticket_id: int | None = None) -> None:
    if _recorder is None:
        return
    now = _now()
    counts = ((metric, None),)
    if ticket_id is not None:
        counts += (("tickets_answered", ticket_id),)
    samples = ()
    if started_at is not None and started_at <= now:
        samples = ((metric, round((now - started_at).total_seconds() * 1000)),)
    _recorder.write(now, counts=counts, samples=samples)


def record_codex_timing(queue_ms: int, execution_ms: int) -> None:
    if _recorder is None:
        return
    _recorder.write(_now(), samples=(("codex_queue", queue_ms), ("codex_execution", execution_ms)))


def _summarize_latency(rows: list[tuple[int, int]]) -> dict:
    count = sum(frequency for _, frequency in rows)
    if not count:
        return {"samples": 0, "median": None, "p90": None}
    ranks = ((count + 1) // 2, count // 2 + 1, (9 * count + 9) // 10)
    values = []
    cumulative = 0
    for milliseconds, frequency in rows:
        cumulative += frequency
        while len(values) < len(ranks) and cumulative >= ranks[len(values)]:
            values.append(milliseconds)
    return {"samples": count, "median": (values[0] + values[1]) / 2, "p90": values[2]}


def monthly_report(database: Path, month: str) -> dict:
    uri = database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.execute("BEGIN")
        recording = db.execute("SELECT started_at_utc FROM recording WHERE id = 1").fetchone()
        counts = dict(db.execute("SELECT metric, count FROM counts WHERE month = ?", (month,)))
        latency_rows = {
            metric: db.execute(
                "SELECT milliseconds, count FROM latency WHERE month = ? AND metric = ? ORDER BY milliseconds",
                (month, metric),
            ).fetchall()
            for metric in _LATENCY_METRICS
        }
    return {
        "month_utc": month,
        "recording_started_at_utc": recording[0] if recording else None,
        "counts": {metric: counts.get(metric, 0) for metric in _COUNT_METRICS},
        "latency_ms": {metric: _summarize_latency(rows) for metric, rows in latency_rows.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a UTC month of ySupport statistics as JSON.")
    parser.add_argument("month", nargs="?", default=_now().strftime("%Y-%m"), help="UTC month in YYYY-MM")
    parser.add_argument("--write", action="store_true", help="save YYYY-MM.json beside the database")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", args.month) or args.month.startswith("0000"):
        parser.error("month must use YYYY-MM")
    directory = _directory()
    try:
        report = monthly_report(directory / "stats.sqlite3", args.month)
        output = json.dumps(report, indent=2) + "\n"
        if args.write:
            path = directory / f"{args.month}.json"
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as handle:
                temporary = Path(handle.name)
                try:
                    handle.write(output)
                    handle.close()
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            print(path)
        else:
            print(output, end="")
    except (OSError, sqlite3.Error) as exc:
        parser.exit(1, f"Could not export ySupport statistics: {exc}\n")


if __name__ == "__main__":
    main()
