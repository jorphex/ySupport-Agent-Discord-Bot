"""Run isolated support/ingestion workflows and write repeatable JSON evidence.

Discord, Telegram, GitHub and model responses are controlled external fixtures.
Production controllers, native execution, persistence, Git and SQLite run normally.
No live messages, credentials or production data are used.
"""

# Runtime imports must follow credential/state isolation.
# ruff: noqa: E402

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import gc
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import tests

for _key in list(os.environ):
    if _key.startswith("TICKET_EXECUTION_") or _key in {
        "OPENAI_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "GITHUB_TOKEN",
        "PINECONE_API_KEY",
        "MCP_SERVER_API_KEY",
    }:
        os.environ.pop(_key)
os.environ["TICKET_EXECUTION_STATE_ROOT"] = tests.TEST_STATE_ROOT
os.environ["TICKET_EXECUTION_ENDPOINT"] = "codex_support_exec"
os.environ["MCP_SERVER_API_KEY"] = "offline-test-key"
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

import config
import docs_repo_tools
import repo_context
import repo_context_build
import state
from codex_support_home import prune_codex_support_caches
from telegram_handoff_controller import TelegramHandoffController
from ticket_investigation.codex_support_endpoint import (
    CodexSupportTicketExecutionJsonEndpoint,
)
from ticket_investigation.executor import TransportTicketInvestigationExecutor
from ticket_investigation.json_endpoint import JsonEndpointTicketExecutionTransport
from tests.test_ticket_intake import (
    _FakeDiscordChannel,
    _FakeOriginalMessage,
    _FakePublicChannel,
    _FakeTriggerMessage,
)
from views import (
    InitialInquiryView,
    stop_ticket_for_manual_support,
    take_over_ticket_for_staff,
)
from yearn_rag.fetch_yearn_docs import SOURCE_URL, refresh_yearn_docs
from ysupport import TicketBot


async def eventually(predicate, *, timeout=8):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("Workflow did not reach its expected state.")
        await asyncio.sleep(0.01)


def verify_cache_retention(root):
    home = root / "home"
    paths = [
        home / ".tmp/plugins/old",
        home / "cache/remote_plugin_catalog/old",
        home / "plugins/.remote-plugin-install-staging/old/file",
        home / "plugins/cache/installed/file",
        home / "sessions/active.jsonl",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("retain unless reproducible cache")
    outside = root / "auth.json"
    outside.write_text("fixture credential sentinel")
    (home / "auth.json").symlink_to(outside)
    old = (datetime.now(timezone.utc) - timedelta(days=9)).timestamp()
    for path in sorted(home.rglob("*"), reverse=True):
        if not path.is_symlink():
            os.utime(path, (old, old))
    recent = home / ".tmp/plugins/recent"
    recent.write_text("in use recently")
    os.utime(recent, (datetime.now(timezone.utc).timestamp(), old))
    os.utime(recent.parent, (old, old))
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    assert prune_codex_support_caches(home, cutoff=cutoff) == 2
    assert recent.exists() and paths[3].exists() and paths[4].exists()
    assert outside.read_text() == "fixture credential sentinel"
    (home / "cache/remote_plugin_catalog").symlink_to(root, target_is_directory=True)
    assert prune_codex_support_caches(home, cutoff=cutoff) == 0
    os.utime(recent, (old, old))
    os.utime(recent.parent, (old, old))
    assert prune_codex_support_caches(home, cutoff=cutoff) == 1
    return {
        "stale_cache_trees_removed": 3,
        "auth_plugins_sessions_recent_files_and_external_symlinks_preserved": True,
    }


def verify_docs_checkout(root):
    from yearn_rag.embed_and_store import source_fingerprint
    from yearn_rag.process_docs import process_markdown_files

    source = root / "origin"
    source.mkdir(parents=True)

    def git(*args):
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    git("init", "-b", "master", str(source))
    git("-C", str(source), "config", "user.email", "fixture@example.invalid")
    git("-C", str(source), "config", "user.name", "E2E fixture")
    (source / "docs").mkdir()
    (source / "static").mkdir()
    (source / "docs/vaults.md").write_text(
        "# Vaults\nWithdrawals return the underlying asset.\n"
    )
    (source / "static/unused.bin").write_bytes(b"unused" * 10000)
    git("-C", str(source), "add", ".")
    git("-C", str(source), "commit", "-m", "initial")
    (source / "README.md").write_text("fixture")
    git("-C", str(source), "add", ".")
    git("-C", str(source), "commit", "-m", "second")
    target = root / "checkout"
    git_env = {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": f"url.{source.as_uri()}.insteadOf",
        "GIT_CONFIG_VALUE_0": SOURCE_URL,
    }
    with patch.dict(os.environ, git_env):
        git("clone", SOURCE_URL, str(target))
        git(
            "-C",
            str(target),
            "config",
            "remote.origin.url",
            SOURCE_URL.removesuffix(".git"),
        )
        original = target / "docs/vaults.md"
        timestamp = original.stat().st_mtime_ns
        fingerprint = source_fingerprint(process_markdown_files(target / "docs", []))
        assert refresh_yearn_docs(target)
        assert original.stat().st_mtime_ns == timestamp
        assert (
            source_fingerprint(process_markdown_files(target / "docs", []))
            == fingerprint
        )
        assert not (target / "static").exists()
        assert git("-C", str(target), "rev-parse", "--is-shallow-repository") == "true"
        assert not refresh_yearn_docs(target)
        (source / "docs/new.md").write_text("# New page\nNew supported behavior.\n")
        git("-C", str(source), "add", ".")
        git("-C", str(source), "commit", "-m", "new docs")
        assert refresh_yearn_docs(target)
        assert original.stat().st_mtime_ns == timestamp
        assert (target / "docs/new.md").exists()
        previous_head = git("-C", str(target), "rev-parse", "HEAD")
        (source / "docs/new.md").write_text("# New page\nUpdated supported behavior.\n")
        git("-C", str(source), "add", ".")
        git("-C", str(source), "commit", "-m", "updated docs")
        rename = Path.rename

        def fail_swap(path, destination):
            if path.name == "checkout" and path.parent.name.startswith(".yearn-docs-"):
                raise OSError("fixture publication failure")
            return rename(path, destination)

        with patch.object(Path, "rename", new=fail_swap):
            try:
                refresh_yearn_docs(target)
            except OSError:
                pass
            else:
                raise AssertionError("Injected swap failure was ignored.")
        assert git("-C", str(target), "rev-parse", "HEAD") == previous_head
        assert refresh_yearn_docs(target)
        previous_head = git("-C", str(target), "rev-parse", "HEAD")
        with patch.dict(
            os.environ,
            {"GIT_CONFIG_KEY_0": f"url.{(root / 'missing').as_uri()}.insteadOf"},
        ):
            try:
                refresh_yearn_docs(target)
            except subprocess.CalledProcessError:
                pass
            else:
                raise AssertionError("Unreachable source replaced the checkout.")
        assert git("-C", str(target), "rev-parse", "HEAD") == previous_head
        original_content = original.read_text()
        original.write_text("local source edit")
        try:
            refresh_yearn_docs(target)
        except RuntimeError as exc:
            assert "local changes" in str(exc)
        else:
            raise AssertionError("Dirty docs checkout was replaced.")
        original.write_text(original_content)
        git("-C", str(target), "config", "user.email", "fixture@example.invalid")
        git("-C", str(target), "config", "user.name", "E2E fixture")
        git("-C", str(target), "commit", "--allow-empty", "-m", "local commit")
        try:
            refresh_yearn_docs(target)
        except RuntimeError as exc:
            assert "local commits" in str(exc)
        else:
            raise AssertionError("Unpublished local commit was discarded.")
    return {
        "shallow_sparse_checkout": True,
        "unchanged_file_timestamps_preserved": True,
        "unchanged_embedding_fingerprint": True,
        "remote_changes_fetched": True,
        "fetch_and_publication_failures_preserved_checkout": True,
        "local_edits_and_commits_protected": True,
    }


def verify_repo_refresh(root):
    root.mkdir(parents=True)
    revisions = {"one": "1" * 40, "two": "2" * 40}
    archives = []
    fail = set()
    documents = {}
    extra_files = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            parts = self.path.split("/")
            name = parts[3] if self.path.startswith("/repos/") else parts[2]
            if name in fail:
                self.send_error(503)
                return
            if self.path.startswith("/repos/"):
                body = json.dumps({"sha": revisions[name]}).encode()
            else:
                revision = parts[-1]
                archives.append((name, revision))
                buffer = io.BytesIO()
                with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
                    content = documents.get(name, f"# Vault {name}\nVerified withdrawal revision {revision}.\n").encode()
                    info = tarfile.TarInfo(f"{name}-{revision}/README.md")
                    info.size = len(content)
                    tar.addfile(info, io.BytesIO(content))
                    if name in {"two", "risk"}:
                        metadata = json.dumps({**{f"field{i}": "ordinary" for i in range(80)},
                                               "metric": "raremetric" if name == "two" else "riskmetric"}).encode()
                        info = tarfile.TarInfo(f"{name}-{revision}/data.json")
                        info.size = len(metadata)
                        tar.addfile(info, io.BytesIO(metadata))
                    for path, text in extra_files.get(name, {}).items():
                        data = text.encode()
                        info = tarfile.TarInfo(f"{name}-{revision}/{path}")
                        info.size = len(data)
                        tar.addfile(info, io.BytesIO(data))
                body = buffer.getvalue()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    manifest = root / "manifest.json"
    payload = {
        "schema_version": 1,
        "repos": [
            {
                "repo": name,
                "ref": "main",
                "product_tag": "metrics" if name == "two" else "vaults",
                "authority_tag": "support_data" if name == "two" else "contract_code",
                "legacy": False,
                "include": ["README.md", "data.json"],
            }
            for name in revisions
        ],
    }
    manifest.write_text(json.dumps(payload))
    db = root / "context.sqlite3"
    get = repo_context_build.requests.Session.get

    def redirect(session, url, *args, **kwargs):
        from urllib.parse import urlsplit

        assert urlsplit(url).hostname in {"api.github.com", "codeload.github.com"}
        return get(
            session,
            f"http://127.0.0.1:{server.server_port}{urlsplit(url).path}",
            *args,
            **kwargs,
        )

    def build():
        return repo_context_build.build_repo_context_index(
            manifest_path=manifest, cache_dir=root / "cache", db_path=db
        )

    try:
        with (
            patch.object(repo_context_build.requests.Session, "get", new=redirect),
            patch.object(config, "REPO_CONTEXT_MANIFEST_PATH", manifest),
        ):
            first = build()
            assert first["repos_reused"] == 0 and len(archives) == 2
            results = repo_context.search_repo_context("withdrawal", db_path=db)
            refs = {r.artifact_ref for r in results}
            assert refs and repo_context.fetch_repo_artifacts(
                [next(iter(refs))], db_path=db
            )
            scoped_support = {}
            for query in ("vault withdrawal", "strategy withdrawal", "strategy raremetric"):
                matches = repo_context.search_repo_context(query, db_path=db)
                assert any(r.repo_name == "yearn/two" for r in matches), query
                if query.startswith("strategy"):
                    assert all(r.repo_name == "yearn/two" for r in matches)
                fetched = repo_context.fetch_repo_artifacts([matches[0].artifact_ref], db_path=db)
                assert fetched and fetched[0]["authority_tag"] == matches[0].authority_tag
                if "raremetric" in query:
                    assert any(r.artifact_ref.startswith("fact:") for r in matches)
                scoped_support[query] = [r.artifact_ref for r in matches]
            second = build()
            assert second["repos_reused"] == 2 and len(archives) == 2
            revisions["one"] = "3" * 40
            third = build()
            assert third["repos_reused"] == 1 and len(archives) == 3
            assert refs == {
                r.artifact_ref
                for r in repo_context.search_repo_context("withdrawal", db_path=db)
            }
            (root / "cache/yearn__one@main/README.md").write_text("corrupt cache")
            assert build()["repos_reused"] == 1 and len(archives) == 4
            downloads = len(archives)
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "UPDATE build_meta SET value = 'previous-parser' WHERE key = 'repo_context.pipeline_hash'"
                )
            assert build()["repos_reused"] == 0 and len(archives) == downloads + 2
            previous_status = repo_context.get_repo_context_status(
                db_path=db, enabled=True
            )
            payload["repos"][0]["include"] = ["missing.md"]
            manifest.write_text(json.dumps(payload))
            try:
                build()
            except RuntimeError as exc:
                assert "matched no files" in str(exc)
            else:
                raise AssertionError("Empty source was published.")
            assert (
                repo_context.get_repo_context_status(db_path=db, enabled=True)[
                    "built_at"
                ]
                == previous_status["built_at"]
            )
            payload["repos"][0]["include"] = ["README.md"]
            payload["repos"][0]["authority_tag"] = "documentation"
            manifest.write_text(json.dumps(payload))
            assert build()["repos_reused"] == 1
            old_status = repo_context.get_repo_context_status(db_path=db, enabled=True)
            revisions["one"] = "4" * 40
            fail.add("two")
            try:
                build()
            except repo_context_build.requests.HTTPError:
                pass
            else:
                raise AssertionError("Failed source was published.")
            assert (
                repo_context.get_repo_context_status(db_path=db, enabled=True)[
                    "built_at"
                ]
                == old_status["built_at"]
            )
            fail.clear()
            payload["repos"] = payload["repos"][:1]
            manifest.write_text(json.dumps(payload))
            assert build()["repos_indexed"] == 1
            assert not (root / "cache/yearn__two@main").exists()
            revisions.update(security="5" * 40, risk="6" * 40, indexing="7" * 40)
            documents.update(
                security="# Yearn security disclosure\nOfficial private report process.\n",
                risk="".join(f"# Security disclosure private report {i}\nThird-party protocol due diligence.\n" for i in range(8)),
                indexing="# Indexer event coverage\nNormalized strategy events are not a complete timeline.\n",
            )
            payload["repos"].extend({
                "repo": name, "ref": "main", "product_tag": name,
                "authority_tag": "security" if name == "security" else "support_data",
                "legacy": False, "include": ["README.md", "data.json"],
            } for name in ("security", "risk", "indexing"))
            manifest.write_text(json.dumps(payload))
            assert build()["repos_indexed"] == 4
            security = repo_context.search_repo_context("security disclosure private report", db_path=db)
            assert security and security[0].repo_name == "yearn/security", [r.repo_name for r in security]
            assert all(r.repo_name != "yearn/risk" for r in security)
            assert repo_context.fetch_repo_artifacts([security[0].artifact_ref], db_path=db)
            assert not repo_context.search_repo_context("router riskmetric", db_path=db)
            risk = repo_context.search_repo_context("due diligence", db_path=db)
            assert risk and risk[0].repo_name == "yearn/risk"
            indexing = repo_context.search_repo_context("strategy indexer event coverage", db_path=db)
            assert indexing and indexing[0].repo_name == "yearn/indexing"
            assert repo_context.fetch_repo_artifacts([indexing[0].artifact_ref], db_path=db)
            text_files = {}
            text_markers = {}
            for ext, prefix, template in (
                ("py", "#", "{marker} = True\n"),
                ("ts", "//", "export const {marker} = true;\n"),
                ("graphql", "#", "type {marker} {{ value: Boolean }}\n"),
                ("yaml", "#", "{marker}: true\n"),
            ):
                path = f"src/long.{ext}"
                markers = [f"{ext}{place}marker" for place in ("head", "middle", "tail")]
                padding = (f"{prefix} repeated source padding with context\n" * 160)
                text_files[path] = template.format(marker=markers[0]) + padding + template.format(marker=markers[1]) + padding + template.format(marker=markers[2])
                text_markers[path] = markers
            bridge = "bridgeleftmarker bridgerightmarker"
            text_files["src/oneline.ts"] = "x" * 1589 + " " + bridge + " " + "x" * 14999 + " onelinetailmarker\n"
            text_markers["src/oneline.ts"] = ["onelinetailmarker"]
            text_files["contracts/Example.sol"] = "contract Example {\nfunction solmethodmarker() external {}\n}\n"
            text_files["contracts/Example.vy"] = "@external\ndef vymethodmarker():\n    pass\n"
            text_files["src/fees.py"] = "state = 'strategy known zero unavailable'\n" * 300
            text_files["docs/fee-accounting.md"] = "# Definitions\nA known zero is verified; unavailable means missing evidence. These states differ for strategy fees.\n"
            extra_files["indexing"] = text_files
            payload["repos"][-1]["include"].extend(text_files)
            manifest.write_text(json.dumps(payload))
            build()
            definitions = repo_context.search_repo_context("strategy fees known zero unavailable", db_path=db)
            assert definitions and definitions[0].path == "docs/fee-accounting.md", [r.path for r in definitions]
            explicit = repo_context.search_repo_context("fees.py known zero", db_path=db)
            assert explicit and explicit[0].path == "src/fees.py"
            text_refs = {}
            for path, markers in text_markers.items():
                for marker in markers:
                    rows = repo_context.search_repo_context(marker, db_path=db)
                    matching = [r for r in rows if r.path == path and marker in r.snippet]
                    assert matching, (path, marker)
                    fetched = repo_context.fetch_repo_artifacts([matching[0].artifact_ref], db_path=db)[0]
                    assert marker in fetched["file_excerpt"] and fetched["path"] == path
                    assert len(fetched["snippet"]) <= config.REPO_CONTEXT_MAX_SNIPPET_CHARS
                    assert len(fetched["file_excerpt"]) <= 4808
                    text_refs[marker] = matching[0].artifact_ref
            rows = repo_context.search_repo_context("bridgeleftmarker bridgerightmarker", db_path=db)
            assert any(bridge in r.snippet for r in rows)
            assert len({r.artifact_ref for r in rows}) == len(rows)
            for marker in ("solmethodmarker", "vymethodmarker"):
                rows = repo_context.search_repo_context(marker, db_path=db)
                assert rows and rows[0].segment_type == "function"
                text_refs[marker] = rows[0].artifact_ref
            with sqlite3.connect(db) as conn:
                chunks = conn.execute("SELECT id,path,segment_type,title,snippet FROM segments WHERE repo_name='yearn/indexing' AND path != 'README.md' ORDER BY id").fetchall()
            for row_id, path, kind, title, snippet in chunks:
                if path not in text_markers:
                    continue
                assert len(snippet) <= 1600
                if kind == "chunk":
                    match = re.search(r"lines (\d+)-(\d+), chars (\d+)-(\d+)$", title)
                    assert match, title
                    first, last, start, end = map(int, match.groups())
                    original = text_files[path]
                    assert original[start:end].strip() == snippet
                    assert first == original.count("\n", 0, start) + 1
                    assert last == original.count("\n", 0, end - 1) + 1
                    fetched = repo_context.fetch_repo_artifacts([f"segment:{row_id}"], db_path=db)[0]
                    assert fetched["file_excerpt"] == snippet
            with patch.dict(os.environ, {"DOCS_INGESTION_FORCE_REFRESH": "1"}):
                build()
            with sqlite3.connect(db) as conn:
                assert chunks == conn.execute("SELECT id,path,segment_type,title,snippet FROM segments WHERE repo_name='yearn/indexing' AND path != 'README.md' ORDER BY id").fetchall()
            assert all(repo_context.fetch_repo_artifacts([ref], db_path=db) for ref in text_refs.values())
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    return {
        "unchanged_sources_reused": 2,
        "changed_and_corrupt_sources_rebuilt": True,
        "stable_search_fetch_refs": True,
        "failed_build_did_not_publish_freshness": True,
        "removed_sources_pruned": True,
        "parser_and_manifest_changes_invalidated_reuse": True,
        "empty_source_did_not_publish_freshness": True,
        "support_data_across_product_filters": scoped_support,
        "security_priority_and_risk_scope": True,
        "risk_metadata_cannot_bypass_scope": True,
        "unscoped_risk_and_scoped_indexer_retrieval": True,
        "full_text_search_fetch": text_refs,
        "chunk_boundaries_overlap_and_budgets": True,
        "unchanged_rebuild_preserves_refs": True,
        "explanatory_docs_and_explicit_source_queries": True,
    }


async def verify_docs_requests():
    calls = {"stats": 0}
    clock = [0.0]

    class Index:
        def describe_index_stats(self):
            calls["stats"] += 1
            if calls["stats"] == 3:
                raise RuntimeError("fixture provider failure")
            return {"namespaces": {"yearn-docs": {}, "flex-docs": {}}}

        def query(self, **kwargs):
            return {
                "matches": [
                    {
                        "id": kwargs["namespace"],
                        "score": 0.9,
                        "metadata": {
                            "text": "Flex is a fixed-rate lending market with lender vaults.",
                            "doc_title": "Vaults",
                            "source_url": "https://docs.yearn.fi/vaults",
                        },
                    }
                ]
            }

    index = Index()
    client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(data=[SimpleNamespace(embedding=[1.0])])
        ),
        inference=SimpleNamespace(
            rerank=lambda **kw: SimpleNamespace(
                data=[SimpleNamespace(index=i) for i in range(len(kw["documents"]))]
            )
        ),
    )
    await docs_repo_tools.close_shared_clients()
    with (
        patch.object(docs_repo_tools, "_get_pinecone_index", return_value=index),
        patch.object(docs_repo_tools, "_get_pinecone_client", return_value=client),
        patch.object(docs_repo_tools, "_get_openai_sync_client", return_value=client),
        patch.object(docs_repo_tools, "monotonic", side_effect=lambda: clock[0]),
    ):
        replies = await asyncio.gather(
            *(
                docs_repo_tools.core_search_docs_context("What is Flex?")
                for _ in range(6)
            )
        )
        assert all("https://docs.yearn.fi/vaults" in reply for reply in replies)
        assert calls["stats"] == 1
        clock[0] += 1000
        await docs_repo_tools.core_search_docs_context("What is Flex?")
        clock[0] += 1000
        await docs_repo_tools.core_search_docs_context("What is Flex?")
        await docs_repo_tools.core_search_docs_context("What is Flex?")
        assert calls["stats"] == 4
    await docs_repo_tools.close_shared_clients()
    return {
        "concurrent_lookups": 6,
        "initial_metadata_requests": 1,
        "expired_and_failed_metadata_refreshed": True,
    }


async def verify_support_flow(root):
    root.mkdir(parents=True)
    child = root / "native_fixture.py"
    child.write_text("""import json,os,sys,time
from pathlib import Path
root=Path(sys.argv[1]); request=json.loads((Path(os.environ["TICKET_EXECUTION_RUN_DIR"])/"support_request.json").read_text()); request["_fixture_pid"]=os.getpid()
channel=str(request["channel_id"]); (root/(channel+".started")).write_text(json.dumps(request))
if channel in {"501","504"}:
 deadline=time.monotonic()+20
 while not (root/(channel+".release")).exists():
  if time.monotonic()>deadline: raise RuntimeError("fixture release timed out")
  time.sleep(.01)
result={"answer":"The withdrawal is queued.","requires_human_handoff":False,"handoff_reason":None,"handoff_kind":None,"evidence_summary":"The internal team confirmed this status.","used_tools":[]}
print(json.dumps({"type":"item.completed","item":{"type":"agent_message","text":json.dumps(result)}}))
""")
    bot = TicketBot(intents=__import__("discord").Intents.none())
    endpoint = CodexSupportTicketExecutionJsonEndpoint(
        codex_command=[sys.executable, str(child), str(root)],
        codex_home=root / "home",
        session_dir=root / "sessions",
        session_max_age_hours=168,
        env={},
        timeout_seconds=20,
    )
    bot.investigation_executor = TransportTicketInvestigationExecutor(
        JsonEndpointTicketExecutionTransport(endpoint)
    )
    channels = {i: _FakeDiscordChannel(i) for i in (501, 502, 503, 504, 505)}
    for channel in channels.values():
        channel.category = SimpleNamespace(id=1)
    bot.get_channel = channels.get
    updates = asyncio.Queue()
    callbacks = []
    pruning = None

    async def api(method, payload):
        if method == "getUpdates":
            try:
                return {
                    "ok": True,
                    "result": await asyncio.wait_for(updates.get(), timeout=0.05),
                }
            except asyncio.TimeoutError:
                return {"ok": True, "result": []}
        if method == "answerCallbackQuery":
            callbacks.append(payload["callback_query_id"])
        return {"ok": True, "result": True}

    def notice(channel):
        state.remember_team_handoff_notice(
            channel,
            state.TeamHandoffNotice(
                telegram_chat_id="123",
                telegram_message_id=channel,
                reason="Fixture team update",
            ),
        )
        state.conversation_threads[channel] = [
            {"role": "user", "content": "What is the withdrawal status?"}
        ]
        state.persist_ticket_state(channel)

    def reply(channel, update_id):
        return {
            "update_id": update_id,
            "message": {
                "message_id": update_id,
                "chat": {"id": "123"},
                "text": "Tell the user the withdrawal is queued.",
                "reply_to_message": {"message_id": channel},
            },
        }

    try:
        with (
            patch.object(config, "TELEGRAM_BOT_TOKEN", "fixture"),
            patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"),
            patch.dict(os.environ, {"YSUPPORT_ALLOW_TEST_TELEGRAM": "1"}),
            patch("handoff._telegram_api_call", new=api),
        ):
            for channel in (501, 502, 503):
                notice(channel)
            bot.telegram_handoffs.start()
            await updates.put([reply(501, 1)])
            await eventually(lambda: (root / "501.started").exists())
            old_cache = root / "home/.tmp/plugins/old"
            old_cache.parent.mkdir(parents=True)
            old_cache.write_text("expired staging fixture")
            old = (datetime.now(timezone.utc) - timedelta(days=9)).timestamp()
            os.utime(old_cache, (old, old))
            os.utime(old_cache.parent, (old, old))
            pruning = asyncio.create_task(endpoint.prune_expired_sessions())
            await asyncio.sleep(0)
            assert not pruning.done() and old_cache.exists()
            await updates.put(
                [
                    {
                        "update_id": 2,
                        "callback_query": {
                            "id": "dismiss-other",
                            "data": "dismiss_handoff",
                            "message": {"chat": {"id": "123"}, "message_id": 502},
                        },
                    },
                    reply(501, 3),
                    reply(503, 4),
                ]
            )
            await eventually(
                lambda: (
                    "dismiss-other" in callbacks
                    and bot.telegram_handoffs._telegram_update_offset == 5
                )
            )
            assert not channels[501].sent_messages
            assert 502 not in state.team_handoff_notice_by_channel
            await take_over_ticket_for_staff(501)
            await stop_ticket_for_manual_support(503)
            (root / "501.release").touch()
            await eventually(lambda: 501 not in state.team_handoff_notice_by_channel)
            await pruning
            assert not old_cache.exists()
            assert channels[501].sent_messages == ["The withdrawal is queued."]
            assert state.stop_reasons_by_channel[501] == "staff_takeover"
            assert not (root / "503.started").exists()
            failed_sends = []

            async def fail_send(*args, **kwargs):
                failed_sends.append(True)
                raise RuntimeError("Expected E2E Discord delivery failure")

            channels[505].send = fail_send
            notice(505)
            await updates.put([reply(505, 5)])
            await eventually(lambda: len(failed_sends) == 1)
            await updates.put([{"update_id": 6, "message": {"text": "noop"}}])
            await eventually(lambda: bot.telegram_handoffs._telegram_update_offset == 7)
            await asyncio.sleep(0.1)
            assert len(failed_sends) == 1
            await stop_ticket_for_manual_support(505)
            notice(504)
            await updates.put([reply(504, 7)])
            await eventually(lambda: (root / "504.started").exists())
            cancelled_pid = json.loads((root / "504.started").read_text())[
                "_fixture_pid"
            ]
            await bot.telegram_handoffs.close()
            assert not Path(f"/proc/{cancelled_pid}").exists()
            assert not channels[504].sent_messages
            persisted = json.loads((state._TICKET_STATE_DIR / "504.json").read_text())
            assert persisted["team_handoff_notice"]["status"] == "pending_delivery"
            for mapping in (
                state.conversation_threads,
                state.ticket_investigation_jobs,
                state.team_handoff_notice_by_channel,
            ):
                mapping.pop(504, None)
            state.hydrate_ticket_state(504)
            assert (
                state.team_handoff_notice_by_channel[504].status == "pending_delivery"
            )
            (root / "504.release").touch()
            bot.telegram_handoffs = TelegramHandoffController(bot)
            bot.telegram_handoffs.start()
            await eventually(lambda: 504 not in state.team_handoff_notice_by_channel)
            assert channels[504].sent_messages == ["The withdrawal is queued."]
            await bot.telegram_handoffs.close()

        original = _FakeOriginalMessage(8001, "Where can I find the withdrawal docs?")
        channel = _FakePublicChannel(801, original)
        with patch(
            "ysupport._outer_support_boundary_result",
            new=AsyncMock(return_value={"tripwire_triggered": False}),
        ):
            await asyncio.gather(
                *(
                    bot._handle_public_trigger_message(
                        _FakeTriggerMessage(channel, reference_message_id=1), "y"
                    )
                    for _ in range(2)
                )
            )
        assert channel.sent_messages == ["The withdrawal is queued."] * 2
        last_request = json.loads((root / "801.started").read_text())
        assert any(
            item.get("role") == "assistant"
            for item in last_request["recent_transcript"]
        )
        gc.collect()
        assert len(bot._public_turn_locks) == 0
        state.remember_ticket_owner_user_id(802, 42)
        business_channel = _FakeDiscordChannel(802)
        interaction = SimpleNamespace(
            channel=business_channel,
            channel_id=802,
            user=SimpleNamespace(id=42),
            message=SimpleNamespace(components=[], id=42),
            edit_original_response=AsyncMock(),
            response=SimpleNamespace(is_done=lambda: False, defer=AsyncMock()),
        )
        view = InitialInquiryView()
        assert await view.interaction_check(interaction)
        await next(
            button
            for button in view.children
            if button.custom_id == "initial_bd_partner"
        ).callback(interaction)
        assert len(business_channel.sent_messages) == 1
        assert (
            "No proposal is necessary for listing." in business_channel.sent_messages[0]
        )
        assert (
            "5 sentences" in business_channel.sent_messages[0]
            and "corn" not in business_channel.sent_messages[0]
        )
    finally:
        await bot.close()
        if pruning is not None:
            if not pruning.done():
                pruning.cancel()
            await asyncio.gather(pruning, return_exceptions=True)
        for channel in (*channels, 802):
            state.clear_ticket_channel_state(channel, delete_persisted=True)
        state.clear_public_conversation(8001)
    return {
        "dismissal_processed_during_slow_native_execution": True,
        "duplicate_reply_ignored": True,
        "staff_takeover_remained_stopped": True,
        "explicit_stop_prevented_queued_delivery": True,
        "failed_delivery_attempted_once_until_manual_intervention": True,
        "cache_maintenance_waited_for_active_native_turn": True,
        "shutdown_killed_child_and_restart_recovered_durable_reply": True,
        "public_locks_reclaimed": True,
        "listing_policy_and_marketing_instructions_shown_without_extra_button": True,
    }


async def verify_native_evaluation(root):
    """Real controllers, JSON subprocess and HTTP I/O; only model outputs are fixtures.

    Failure probes: skipped lookup, wrong wallet/chain/hash, unwanted handoff,
    stale discovery coverage, and a controller that keeps running after a handoff.
    """
    from tests import native_support_e2e as evaluation

    root.mkdir(parents=True)
    child = root / "evaluation_model.py"
    child.write_text(r'''import asyncio,json,os,re,sys,subprocess
from pathlib import Path
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
request=json.loads((Path(os.environ["TICKET_EXECUTION_RUN_DIR"])/"support_request.json").read_text())
mode=sys.argv[1];text=request["current_user_message"];blob=json.dumps(request)
handoff=mode.startswith("handoff");used=[]
if mode=="handoff_error" and "solidity" in text:raise RuntimeError("Expected fixture failure after bypassed handoff")
answer="yvUSDC 12.34; approval reverted because allowance was too low."
async def tools():
 async with streamablehttp_client(os.environ["YSUPPORT_EVAL_MCP_URL"],headers={"Authorization":"Bearer "+os.environ["YSUPPORT_EVAL_MCP_TOKEN"]}) as (r,w,_):
  async with ClientSession(r,w) as session:
   await session.initialize()
   if mode.startswith("coverage"):
    args={"coverage_empty":{"market":"eth"},"coverage_zero":{"token_symbol":"USDC"},"coverage_page":{}}[mode]
    result=await session.call_tool("support_dashboard_discover",{"chain_id":1,"limit":1,**args})
    content="\n".join(item.text for item in result.content if hasattr(item,"text"))
    data=json.loads(content[content.index("{"):])
    return json.dumps({"returned_rows":len(data["rows"]),"total":data["pagination"]["total"],**data["coverage"]})
   if mode.startswith("apy"):
    await session.call_tool("search_vaults",{"query":"usdc"})
    return "yvUSDC APY: "+("14.2%" if mode=="apy_wrong" else "4.2%")
   if mode.startswith("docs_apy"):
    await session.call_tool("search_documentation",{"query":"vault APY price per share weekly window"})
    correct="Vault APY annualizes backward-looking PPS performance."
    if "Thanks." in text:return correct if mode=="docs_apy_wrong" else "Yes, the same methodology applies."
    return "APY is only the wallet balance." if mode=="docs_apy_wrong" else correct
   if "Liquid Locker" in text or "LiquidLocker" in text or "solidity" in text:
    await session.call_tool("search_repo_context",{"query":"LiquidLockerDepositor unstake"})
    await session.call_tool("fetch_repo_artifacts",{"artifact_refs_text":"segment:1"})
    return "LiquidLockerDepositor: unstake resets block.timestamp and maxRedeem. yearn-security policy requires private review."
   query="styfi veyfi" if "yfi" in text.lower() else "getting started deposit"
   result=await session.call_tool("search_documentation",{"query":query})
   return "\n".join(item.text for item in result.content if hasattr(item,"text"))
if mode.startswith(("apy","docs_apy","coverage")):
 answer=asyncio.run(tools());used=["ysupport_mcp"]
elif handoff:
 answer=asyncio.run(tools())+" A maintainer decision is needed."
 used=["ysupport_mcp"]
elif mode=="scan_docs":
 subprocess.run(request["support_state"]["internal_rpc"]["command"],input=json.dumps({"operation":"check_all_deposits","arguments":{"user_address_or_ens":"0x"+"11"*20}}),text=True,capture_output=True,timeout=10)
 used=["shell"]
elif mode!="skip":
 if "tx hash" in text.lower():
  tx=re.search(r"0x[0-9a-fA-F]{64}",text).group();chain="ethereum"
  query={"chain":"katana" if mode=="wrong_chain" else chain,"tx_hash":"0x"+"0"*64 if mode=="wrong_hash" else tx,"mode":"receipt"};operation="inspect_onchain"
 else:
  wallet=re.search(r"0x[0-9a-fA-F]{40}",text).group()
  query={"user_address_or_ens":"0x"+"0"*40 if mode=="wrong_wallet" else wallet};operation="check_all_deposits"
 subprocess.run(request["support_state"]["internal_rpc"]["command"],input=json.dumps({"operation":operation,"arguments":query}),text=True,capture_output=True,timeout=10)
 used=["shell"]
result={"answer":answer,"requires_human_handoff":handoff,"handoff_kind":"human_decision" if handoff else None,"handoff_reason":"Maintainer decision required" if handoff else None,"evidence_summary":"Controlled external model fixture","used_tools":used}
print(json.dumps({"type":"item.completed","item":{"type":"agent_message","text":json.dumps(result)}}))
''')
    cases = json.loads(evaluation.CASE_PATH.read_text())
    by_id = {case["id"]: case for case in cases}
    import copy
    import ysupport

    probes, expected = [], []

    def probe(mode, case_id, error=None):
        case = copy.deepcopy(by_id[case_id])
        if mode.startswith("handoff") and "poc_followup" in case_id:
            case["turns"][0]["expect_handoff"] = True
        case["id"] = f"{len(probes)}:{mode}:{case_id}"
        case["_fixture_mode"] = mode
        probes.append(case)
        expected.append(error)

    deposit = "deposit_button_intent_calls_check_all_deposits_tool"
    transaction = "tx_hash_bug_report_uses_onchain_inspection_without_wallet_prompt"
    probe("correct", deposit)
    probe("skip", deposit, "Missing provider lookup")
    probe("wrong_wallet", deposit, "Missing provider lookup")
    probe("wrong_chain", transaction, "Missing provider lookup")
    probe("wrong_hash", transaction, "Missing provider lookup")
    probe("scan_docs", "apy_mechanics_question_routes_to_docs_not_vault_search", "Unwanted call: positions")
    for name in (
        "styfi_contract_address_question_routes_to_docs_and_answers_directly",
        "veyfi_deposit_question_points_to_current_supported_styfi_path",
        "onboarding_question_uses_docs_backed_getting_started_answer_without_yfi_detour",
    ):
        probe("handoff", name, "Expected handoff=False")
    probe("apy", "combined_data_button_routes_vault_query_to_search_tool")
    probe("apy_wrong", "combined_data_button_routes_vault_query_to_search_tool", "Missing answer fact")
    probe("docs_apy", "apy_mechanics_question_routes_to_docs_not_vault_search")
    probe("docs_apy_wrong", "apy_mechanics_question_routes_to_docs_not_vault_search", "Missing answer concept")
    coverage_expected = {
        "coverage_empty": {"returned_rows": 0, "total": 0, "visible_vaults": 0,
                           "with_realized_apy": 0, "without_realized_apy": 0, "coverage_ratio": None},
        "coverage_zero": {"returned_rows": 1, "total": 1, "visible_vaults": 1,
                          "with_realized_apy": 1, "without_realized_apy": 0, "coverage_ratio": 1.0},
        "coverage_page": {"returned_rows": 1, "total": 2, "visible_vaults": 2,
                          "with_realized_apy": 1, "without_realized_apy": 1, "coverage_ratio": 0.5},
    }
    for mode in coverage_expected:
        probe(mode, "recommendation_question_avoids_single_strategy_ys_vaults_by_default")
        case = probes[-1]
        case["checks"] = {"tools": ["support_dashboard_discover"], "handoff": False}
        payload = case["providers"]["support_dashboard_discover"]
        payload["rows"][0]["realized_apy_30d"] = 0.0
        payload["coverage"] = {"visible_vaults": 2, "with_realized_apy": 1,
                               "without_realized_apy": 1, "coverage_ratio": 0.5}
    followup = "sdyfi_pasted_poc_followup_stays_repo_grounded_without_browser_questions"
    probe("handoff", followup)
    probe("handoff", followup, "Expected delivery=queued_for_team")
    probe("handoff_error", followup, "Expected delivery=queued_for_team")
    original_waiting = ysupport.is_ticket_waiting_for_team

    def simulate_broken_guard(channel_id):
        return False if channel_id >= 970000 + len(probes) - 2 else original_waiting(channel_id)

    async def classifier(_self, **_kwargs):
        return SimpleNamespace(final_output_as=lambda schema: schema(
            classification="yearn_support", business_subtype=None, reasoning="Controlled model fixture"))

    output = root / "native-evaluation.json"
    with (
        patch("support_boundary.Runner.run", new=classifier),
        patch("ysupport.is_ticket_waiting_for_team", new=simulate_broken_guard),
    ):
        status = await evaluation.run(
            SimpleNamespace(output=output, case=[]),
            command_override=lambda case: [sys.executable, str(child), case["_fixture_mode"]],
            cases_override=probes,
        )
    report = json.loads(output.read_text())
    assert status == 1 and report["complete"] and report["model"] == "external-fixture"
    coverage_observed = {}
    for case, error, result in zip(probes, expected, report["cases"]):
        failures = result["failures"]
        assert (any(error in failure for failure in failures) if error else not failures), result
        mode = case["_fixture_mode"]
        if mode in coverage_expected:
            coverage_observed[mode] = json.loads(result["turns"][0]["answer"])
            assert coverage_observed[mode] == coverage_expected[mode], (mode, coverage_observed[mode])
    failed_attempt = report["cases"][-1]["turns"][-1]
    assert failed_attempt["native_calls"] == 1 and failed_attempt["delivery"] == "error"
    return {"cases": report["cases"], "discovery_coverage": coverage_observed,
            "all_expected_regressions_rejected": True}


async def main(artifact):
    report = {
        "ok": False,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "scenarios": {},
    }
    report["source_hashes"] = {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in (
            "telegram_handoff_controller.py",
            "ysupport.py",
            "views.py",
            "bot_behavior.py",
            "docs_repo_tools.py",
            "repo_context.py",
            "repo_context_build.py",
            "codex_support_home.py",
            "ticket_investigation/codex_support_endpoint.py",
            "ticket_investigation/internal_rpc.py",
            "chain_access.py",
            "onchain_tools.py",
            "vault_position_tools.py",
            "yearn_rag/fetch_yearn_docs.py",
            "yearn_rag/update_docs.sh",
            "tests/optimization_e2e.py",
            "tests/native_support_e2e.py",
            "tests/native_support_cases.json",
        )
    }
    logging.getLogger().setLevel(logging.WARNING)
    try:
        with tempfile.TemporaryDirectory(prefix="ysupport-optimization-e2e-") as temp:
            root = Path(temp)
            report["scenarios"]["cache_retention"] = verify_cache_retention(
                root / "caches"
            )
            report["scenarios"]["docs_checkout"] = verify_docs_checkout(root / "docs")
            report["scenarios"]["repo_refresh"] = verify_repo_refresh(root / "repos")
            report["scenarios"]["docs_retrieval"] = await verify_docs_requests()
            report["scenarios"]["support_ui_and_delivery"] = await verify_support_flow(
                root / "support"
            )
            report["scenarios"]["native_evaluation"] = await verify_native_evaluation(
                root / "native-evaluation"
            )
        report["ok"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(report, indent=2) + "\n")
        print(f"E2E evidence: {artifact}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(".cache/verification/optimization-e2e.json"),
    )
    asyncio.run(main(parser.parse_args().artifact))
