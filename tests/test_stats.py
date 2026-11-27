from contextlib import closing, ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ysupport import config, stats


class MonthlyStatsTests(unittest.TestCase):
    def setUp(self):
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        root = Path(contexts.enter_context(TemporaryDirectory()))
        self.database = root / "support-stats/stats.sqlite3"
        self.day = datetime(2026, 11, 30, 23, 59, 59, tzinfo=timezone.utc)
        contexts.enter_context(patch.object(config, "TICKET_EXECUTION_STATE_ROOT", root))
        contexts.enter_context(patch.object(stats, "_now", lambda: self.day))
        contexts.enter_context(patch.object(stats, "_recorder", None))

    def test_month_rollover_restart_and_deletion_preserve_anonymous_totals(self):
        ticket_id = 1471293489343987654
        stats.record("tickets_opened", ticket_id=ticket_id)
        self.assertFalse(self.database.exists())
        stats.enable_recording()
        stats.record("tickets_opened", ticket_id=ticket_id)
        stats.record("tickets_opted_in", ticket_id=ticket_id)
        stats.record_reply("ticket_auto_replies", self.day - timedelta(seconds=7), ticket_id=ticket_id)

        stats.enable_recording()
        stats.record("tickets_opened", ticket_id=ticket_id)
        stats.record("tickets_opted_in", ticket_id=ticket_id)
        stats.record_reply("ticket_auto_replies", self.day - timedelta(seconds=3), ticket_id=ticket_id)
        stats.record_reply("ticket_staff_replies", None, ticket_id=ticket_id)
        stats.record_codex_timing(19, 4300)
        november = stats.monthly_report(self.database, "2026-11")
        self.assertEqual(november["recording_started_at_utc"], "2026-11-30T23:59:59+00:00")
        self.assertEqual(
            {metric: value for metric, value in november["counts"].items() if value},
            {
                "tickets_opened": 1, "tickets_opted_in": 1, "tickets_answered": 1,
                "ticket_auto_replies": 2, "ticket_staff_replies": 1,
            },
        )
        self.assertEqual(november["latency_ms"]["ticket_auto_replies"], {"samples": 2, "median": 5000, "p90": 7000})
        self.assertEqual(november["latency_ms"]["ticket_staff_replies"]["samples"], 0)
        self.assertEqual(november["latency_ms"]["codex_queue"]["p90"], 19)
        self.assertEqual(november["latency_ms"]["codex_execution"]["p90"], 4300)
        self.assertNotIn(str(ticket_id).encode(), self.database.read_bytes())

        self.day += timedelta(seconds=1)
        stats.record_reply("ticket_staff_replies", self.day - timedelta(seconds=10), ticket_id=ticket_id)
        december = stats.monthly_report(self.database, "2026-12")
        self.assertEqual(december["counts"]["tickets_answered"], 1)
        self.assertEqual(december["counts"]["tickets_opened"], 0)
        with closing(sqlite3.connect(self.database)) as db:
            self.assertEqual(db.execute("SELECT month FROM ticket_seen").fetchall(), [("2026-12",)])
            stats.forget_ticket(ticket_id)
            self.assertEqual(db.execute("SELECT month FROM ticket_seen").fetchall(), [])
        self.assertEqual(stats.monthly_report(self.database, "2026-11"), november)
        self.assertEqual(stats.monthly_report(self.database, "2026-12"), december)

    def test_storage_failure_rolls_back_the_whole_reply_and_allows_later_writes(self):
        stats.enable_recording()
        with closing(sqlite3.connect(self.database)) as db:
            db.executescript("""
                CREATE TRIGGER fail_sample BEFORE INSERT ON latency
                BEGIN SELECT RAISE(ABORT, 'simulated latency write failure'); END;
            """)
            with self.assertLogs("ysupport.stats", level="ERROR"):
                stats.record_reply("ticket_staff_replies", self.day, ticket_id=1234)
            self.assertFalse(any(stats.monthly_report(self.database, "2026-11")["counts"].values()))
            db.execute("DROP TRIGGER fail_sample")

        stats.record_reply("ticket_staff_replies", self.day, ticket_id=1234)
        report = stats.monthly_report(self.database, "2026-11")
        self.assertEqual(report["counts"]["tickets_answered"], 1)
        self.assertEqual(report["counts"]["ticket_staff_replies"], 1)
        self.assertEqual(report["latency_ms"]["ticket_staff_replies"]["samples"], 1)

    def test_locked_fingerprint_deletion_survives_restart(self):
        stats.enable_recording()
        stats.record("tickets_opened", ticket_id=1234)
        stats.record_handoff_outcome("ticket_handoff_replies_delivered", "test-nonce", 1234)

        with closing(sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True)) as export:
            export.execute("BEGIN")
            export.execute("SELECT COUNT(*) FROM counts").fetchone()
            with self.assertLogs("ysupport.stats", level="ERROR"):
                stats.forget_ticket(1234)
            self.assertEqual(export.execute("SELECT COUNT(*) FROM ticket_seen").fetchone()[0], 1)

        stats.enable_recording()
        with closing(sqlite3.connect(self.database)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM ticket_seen").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM handoff_seen").fetchone()[0], 0)
        report = stats.monthly_report(self.database, "2026-11")
        self.assertEqual(report["counts"]["tickets_opened"], 1)
        self.assertEqual(report["counts"]["ticket_handoff_replies_delivered"], 1)

    def test_handoff_retry_across_utc_rollover_counts_the_original_delivery(self):
        stats.enable_recording()
        metric = "public_handoff_replies_delivered"
        stats.record_handoff_outcome(metric, "test-nonce", 1234)
        self.day += timedelta(seconds=1)
        stats.enable_recording()
        stats.record_handoff_outcome(metric, "test-nonce", 1234)
        stats.settle_handoff_outcome("test-nonce")

        self.assertEqual(stats.monthly_report(self.database, "2026-11")["counts"][metric], 1)
        self.assertEqual(stats.monthly_report(self.database, "2026-12")["counts"][metric], 0)
        with closing(sqlite3.connect(self.database)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM handoff_seen").fetchone()[0], 0)
