"""Verify package origins, resources and real CLI/script entry points offline."""

import argparse
import asyncio
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import tests

for key in list(os.environ):
    if key.startswith(("TICKET_EXECUTION_", "REPO_CONTEXT_")) or key in {
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "ALCHEMY_KEY",
        "GITHUB_TOKEN",
        "DISCORD_BOT_TOKEN",
        "TELEGRAM_BOT_TOKEN",
        "MCP_SERVER_API_KEY",
    }:
        os.environ.pop(key)
os.environ.update(
    PYTHON_DOTENV_DISABLED="1", TICKET_EXECUTION_STATE_ROOT=tests.TEST_STATE_ROOT
)
ROOT = Path(__file__).resolve().parents[1]


def verify_package():
    assert not list(ROOT.glob("*.py")), "Loose root Python modules remain"
    for name in ("bot", "execution", "integrations", "tools", "retrieval", "ingestion"):
        assert (ROOT / "ysupport" / name / "__init__.py").is_file(), name
    modules = [
        "ysupport.config",
        "ysupport.bot.client",
        "ysupport.bot.state",
        "ysupport.bot.views",
        "ysupport.integrations.mcp_server",
        "ysupport.integrations.discord_api",
        "ysupport.integrations.ticket_transcript_fetch",
        "ysupport.execution.internal_rpc",
        "ysupport.execution.json_endpoint",
        "ysupport.tools.vault_position_tools",
        "ysupport.tools.vault_search_tools",
        "ysupport.retrieval.docs_repo_tools",
        "ysupport.retrieval.repo_context_build",
        "ysupport.ingestion.fetch_yearn_docs",
        "ysupport.ingestion.fetch_flex_docs",
        "ysupport.ingestion.process_docs",
        "ysupport.ingestion.embed_and_store",
    ]
    for name in modules:
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(ROOT / "ysupport"), name
    return {"module_count": len(modules), "one_package_identity": True}


def verify_resources():
    from ysupport import config
    from ysupport.execution.codex_support_home import _instructions_template_path
    from ysupport.ingestion import embed_and_store, fetch_flex_docs, process_docs

    assert config.BASE_DIR == ROOT
    assert config.REPO_CONTEXT_DB_PATH == ROOT / ".cache/repo_context/repo_context.sqlite3"
    assert (
        config.REPO_CONTEXT_MANIFEST_PATH
        == ROOT / "ysupport/retrieval/repo_sources.json"
    )
    assert json.loads(config.REPO_CONTEXT_MANIFEST_PATH.read_text())["repos"]
    assert _instructions_template_path().is_file()
    assert json.loads((ROOT / "ysupport/tools/v1_vaults.json").read_text())
    data = ROOT / ".cache/docs_sources"
    assert fetch_flex_docs.OUTPUT_DIR == data / "flex-docs"
    assert (
        Path(process_docs.DOC_SOURCES["yearn"]["source_dir"])
        == data / "yearn-devdocs/docs"
    )
    assert (
        Path(embed_and_store.EMBEDDING_SOURCES["yearn_docs"]["input_json"])
        == data / "cleaned_yearn_docs.json"
    )
    with tempfile.TemporaryDirectory() as temp:
        code = "import sys;sys.path.insert(0,sys.argv[1]);from ysupport import config;from ysupport.execution.codex_support_home import _instructions_template_path;assert str(config.BASE_DIR)==sys.argv[1];assert _instructions_template_path().is_file()"
        subprocess.run(
            [sys.executable, "-B", "-c", code, str(ROOT)],
            cwd=temp,
            env=dict(os.environ),
            check=True,
            capture_output=True,
            text=True,
        )
    return {
        "resources_found": True,
        "foreign_cwd": True,
        "generated_data_outside_source": True,
    }


def verify_cli():
    modules = [
        "ysupport.execution.status",
        "ysupport.execution.sessions",
        "ysupport.execution.replay_cli",
        "ysupport.execution.replay_capture",
        "ysupport.integrations.ticket_transcript_fetch",
        "ysupport.ingestion.fetch_yearn_docs",
        "ysupport.ingestion.build_repo_context",
        "ysupport.ingestion.verify_repo_context",
    ]
    for module in modules:
        result = subprocess.run(
            [sys.executable, "-B", "-m", module, "--help"],
            cwd=ROOT,
            env=dict(os.environ),
            text=True,
            capture_output=True,
            timeout=15,
        )
        assert result.returncode == 0 and "usage:" in result.stdout.lower(), (
            module,
            result.stderr,
        )
    return {"cli_modules": len(modules), "no_live_calls": True}


def verify_host_script():
    with tempfile.TemporaryDirectory() as temp:
        fixture = Path(temp)
        (fixture / "scripts").mkdir()
        (fixture / ".venv/bin").mkdir(parents=True)
        script = fixture / "scripts/run_ysupport_host.sh"
        script.write_bytes((ROOT / "scripts/run_ysupport_host.sh").read_bytes())
        python = fixture / ".venv/bin/python"
        python.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$@" > "$LAYOUT_PROBE"\n')
        python.chmod(0o755)
        probe = fixture / "probe"
        env = {
            **os.environ,
            "LAYOUT_PROBE": str(probe),
            "TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE": str(
                fixture / "fixture-auth.json"
            ),
        }
        subprocess.run(
            ["bash", str(script)],
            cwd="/tmp",
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert probe.read_text().splitlines() == [
            str(fixture),
            "-m",
            "ysupport.bot.client",
        ]
    subprocess.run(
        ["bash", "-n", str(ROOT / "scripts/update_docs.sh")],
        check=True,
        capture_output=True,
    )
    return {"host_entrypoint": True, "script_cwd": True, "docs_script_syntax": True}


async def main(artifact):
    report = {"ok": False, "scenarios": {}}
    for name, check in (
        ("package", verify_package),
        ("resources", verify_resources),
        ("cli", verify_cli),
        ("host_script", verify_host_script),
    ):
        try:
            report["scenarios"][name] = {"ok": True, **check()}
        except Exception as exc:
            report["scenarios"][name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    report["ok"] = all(value["ok"] for value in report["scenarios"].values())
    report["source_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*(ROOT / "ysupport").rglob("*.py"), Path(__file__)]
        if ".venv" not in p.parts
        and (p.is_relative_to(ROOT / "ysupport") or p == Path(__file__))
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["scenarios"], indent=2))
    return int(not report["ok"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path, default=ROOT / ".cache/verification/layout-e2e.json"
    )
    raise SystemExit(asyncio.run(main(parser.parse_args().artifact)))
