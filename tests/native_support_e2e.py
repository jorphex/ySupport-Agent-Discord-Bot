"""Evaluate real ticket controllers and native execution with controlled data I/O.

RUN_NATIVE_SUPPORT_E2E=1 python -B -m tests.native_support_e2e --output /private/eval.json
Run as an unprivileged service user with explicit bot auth and classifier API key.
Discord/Telegram and data providers are fixtures; native models are real.
"""

import argparse
import asyncio
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import secrets
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch


SOURCE_ROOT = Path(__file__).resolve().parent.parent
CASE_PATH = Path(__file__).with_name("native_support_cases.json")


def failures(case, calls, turns):
    """Grade delivered native answers, controller transitions and observed I/O."""
    checks = case["checks"]
    answered = [turn for turn in turns if turn["delivery"] in {"answered", "boundary"}]
    errors = []
    if not answered:
        return ["No support answer was delivered"]
    index = checks.get("answer_turn", -1)
    if not -len(answered) <= index < len(answered):
        return ["Expected answer turn was not delivered"]
    graded_turn = answered[index]
    text = re.sub(r"(?<=\d),(?=\d)", "", graded_turn["answer"].lower())
    transcript = "\n".join(turn["answer"].lower() for turn in turns)

    def includes(word):
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?%", word):
            expected = Decimal(word[:-1])
            return any(Decimal(value) == expected for value in re.findall(r"(?<![\d.+-])([+-]?\d+(?:\.\d+)?)\s*%", text))
        return word.lower() in text

    for words in checks.get("transcript_any", []):
        if not any(word.lower() in transcript for word in words):
            errors.append(f"Missing transcript evidence: {words}")
    for word in checks.get("transcript_none", []):
        if word.lower() in transcript:
            errors.append(f"Unwanted transcript request: {word}")
    for word in checks.get("all", []):
        if not includes(word):
            errors.append(f"Missing answer fact: {word}")
    for words in checks.get("any", []):
        if not any(includes(word) for word in words):
            errors.append(f"Missing answer concept: {words}")
    for word in checks.get("none", []):
        if includes(word):
            errors.append(f"Unwanted answer: {word}")
    if "handoff" in checks and graded_turn["handoff"] != checks["handoff"]:
        errors.append(f"Expected handoff={checks['handoff']}, got {graded_turn['handoff']}")
    if "boundary" in checks and graded_turn["boundary"] != checks["boundary"]:
        errors.append(f"Expected boundary={checks['boundary']}, got {graded_turn['boundary']}")
    used = {call["tool"] for call in calls if call["fixture_matched"]}
    for name in checks.get("tools", []):
        if name not in used:
            errors.append(f"Missing actual MCP call: {name}")
    for names in checks.get("tools_any", []):
        if not used.intersection(names):
            errors.append(f"Missing actual MCP call from: {names}")
    for name in checks.get("no_tools", []):
        if any(call["tool"] == name for call in calls):
            errors.append(f"Unwanted call: {name}")
    for lookup in case.get("lookups", []):
        if not lookup.get("required", True):
            continue
        if not any(call["tool"] == lookup["operation"] and call["fixture_matched"]
                   and call.get("transport") == ("http" if lookup["operation"] == "report" else "internal_rpc")
                   and {k: v.lower() for k, v in call.get("query", {}).items()}
                   == {k: v.lower() for k, v in lookup["query"].items()} for call in calls):
            errors.append(f"Missing provider lookup: {lookup['operation']} {lookup['query']}")
    for index, (expected, actual) in enumerate(zip(case["turns"], turns)):
        delivery = expected.get("expect_delivery", "answered")
        if delivery == "queued_if_handoff":
            delivery = "queued_for_team" if index and turns[index - 1]["handoff"] else "answered"
        valid = actual["delivery"] in {"answered", "boundary"} if delivery == "answered" else actual["delivery"] == delivery
        if not valid:
            errors.append(f"Expected delivery={delivery}, got {actual['delivery']}")
        if "expect_handoff" in expected and actual["handoff"] != expected["expect_handoff"]:
            errors.append(f"Expected turn handoff={expected['expect_handoff']}, got {actual['handoff']}")
        if actual["delivery"] == "queued_for_team":
            if actual["native_calls"] or not actual["followup_persisted"]:
                errors.append("Queued follow-up ran the model or was not persisted")
        elif not actual["answer"].strip():
            errors.append("Empty answer")
        if actual["handoff"] and not (actual["waiting_for_team"] and actual["handoff_persisted"]):
            errors.append("Requested handoff was not persisted")
        if actual.get("runtime_error"):
            errors.append("Ticket controller stopped for a runtime error")
    if len(turns) != len(case["turns"]):
        errors.append("Not every user turn was processed")
    return errors


class DiscordChannel:
    """Only the external Discord object is simulated; controller code is unchanged."""
    def __init__(self, channel_id):
        self.id = channel_id
        self.category = SimpleNamespace(id=1)
        self.guild = SimpleNamespace(id=1)
        self.messages = []

    async def send(self, content, **_kwargs):
        message = SimpleNamespace(content=content, deleted=False)

        async def edit(*, content):
            message.content = content

        async def delete():
            message.deleted = True

        message.edit, message.delete = edit, delete
        self.messages.append(message)
        return message

    async def history(self, **_kwargs):
        for message in []:
            yield message

    def typing(self):
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def context():
            yield

        return context()


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


async def run(args, *, command_override=None, cases_override=None):
    # Root, cwd, environment and state isolation precede every runtime import.
    runtime_env = dict(os.environ)
    if command_override is None:
        if os.geteuid() == 0:
            raise ValueError("Run native evaluations as an unprivileged service user.")
        auth = Path(runtime_env.get("TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE", ""))
        if not auth.is_file() or not runtime_env.get("OPENAI_API_KEY"):
            raise ValueError("Provide explicit bot auth and OPENAI_API_KEY for the classifier.")
        metadata = auth.stat()
        if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
            raise ValueError("Bot auth must be service-owned and mode 0600.")
    else:
        auth = None
    output = args.output.absolute()
    old_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="ysupport-native-e2e-") as temp, ExitStack() as patches:
        root = Path(temp)
        os.chdir(root)
        patches.callback(os.chdir, old_cwd)
        patches.enter_context(patch.dict(os.environ, {
            "PYTHON_DOTENV_DISABLED": "1", "YSUPPORT_ALLOW_TEST_TELEGRAM": "1",
            "YSUPPORT_ALLOW_TEST_LLM": "0", "TICKET_EXECUTION_STATE_ROOT": str(root / "state"),
            "TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE": "", "MCP_SERVER_API_KEY": "fixture",
            "DISCORD_BOT_TOKEN": "", "TELEGRAM_BOT_TOKEN": "", "PINECONE_API_KEY": "",
            "ALCHEMY_KEY": "", "GITHUB_TOKEN": "",
        }))
        with patch("dotenv.load_dotenv", return_value=False):
            from ysupport import config
        from ysupport.tools import chain_access
        import discord
        from ysupport.retrieval import docs_repo_tools
        from ysupport.integrations import handoff
        from ysupport.integrations import mcp_server
        from ysupport.tools import onchain_tools
        from ysupport.bot import state
        from ysupport.tools import support_dashboard_tools
        import uvicorn
        from ysupport.tools import position_evidence
        from ysupport.tools import vault_search_tools
        from ysupport.tools import vault_position_tools
        from ysupport.tools import yearn_targets
        from eth_utils.abi import abi_to_signature, collapse_if_tuple
        from web3 import Web3
        from ysupport.bot import client as ysupport
        from starlette.responses import JSONResponse
        from ysupport.bot.support_boundary import evaluate_support_boundary
        from ysupport.execution.codex_support_endpoint import CodexSupportTicketExecutionJsonEndpoint
        logging.getLogger().setLevel(logging.WARNING)

        cases = cases_override if cases_override is not None else json.loads(CASE_PATH.read_text())
        if args.case:
            cases = [case for case in cases if set([case["id"], *case.get("aliases", [])]).intersection(args.case)]
            if not set(args.case).issubset({name for case in cases for name in [case["id"], *case.get("aliases", [])]}):
                raise ValueError("Unknown scenario ID")
        report = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "case_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
            "source_sha256": {name: hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest() for name in (
                'ysupport/bot/client.py', 'ysupport/bot/ticket_intake.py', 'ysupport/bot/state.py', 'ysupport/bot/support_boundary.py', 'ysupport/integrations/mcp_server.py', 'ysupport/bot/bot_behavior.py', 'ysupport/execution/codex_support_contract.py',
                'ysupport/tools/support_dashboard_tools.py', 'ysupport/tools/vault_search_tools.py', 'ysupport/tools/kong_vaults.py',
                'ysupport/execution/ysupport_codex_instructions.md', 'ysupport/execution/codex_support_endpoint.py',
                'ysupport/execution/internal_rpc.py', 'ysupport/execution/json_endpoint.py', 'ysupport/tools/position_evidence.py', 'ysupport/tools/chain_access.py', 'ysupport/tools/onchain_tools.py', 'ysupport/tools/vault_position_tools.py',
                "tests/native_support_e2e.py", "tests/native_support_cases.json",
            )},
            "model": "external-fixture" if command_override else config.TICKET_EXECUTION_CODEX_MODEL,
            "reasoning_effort": config.TICKET_EXECUTION_CODEX_REASONING_EFFORT,
            "boundary_model": config.LLM_GUARDRAIL_MODEL,
            "selected_case_ids": [case["id"] for case in cases],
            "legacy_scenario_ids": [name for case in cases if case.get("legacy_suite") for name in [case["id"], *case.get("aliases", [])]],
            "total": len(cases),
            "legacy_scenarios_covered": sum(1 + len(case.get("aliases", [])) for case in cases if case.get("legacy_suite")),
            "complete": False, "cases": [],
        }
        write_report(output, report)
        current, calls, native_results, boundaries = {}, [], [], []
        native_calls = 0
        providers = {
            "search_documentation": (docs_repo_tools, "core_search_docs_context"),
            "search_vaults": (vault_search_tools, "core_search_vaults"),
            "search_repo_context": (docs_repo_tools, "core_search_repo_context"),
            "fetch_repo_artifacts": (docs_repo_tools, "core_fetch_repo_artifacts"),
            "repo_context_status": (docs_repo_tools, "core_repo_context_status"),
            **{name: (support_dashboard_tools, f"core_{name}") for name in (
                "support_dashboard_discover", "support_dashboard_reports", "support_dashboard_status",
                "support_dashboard_changes", "support_dashboard_styfi",
                "support_dashboard_flex",
            )},
        }

        def provider(name):
            async def invoke(*positional, **keywords):
                fixtures = current["providers"]
                terms = current.get("provider_query_terms", {}).get(name, [])
                query = json.dumps([positional, keywords]).lower()
                matched = name in fixtures and (not terms or any(term.lower() in query for term in terms))
                value = fixtures.get(name)
                record = {"tool": name, "args": list(positional), "kwargs": keywords, "fixture_matched": matched}
                if name == "search_vaults" and matched:
                    value = deepcopy(value)
                    search = (positional[0] if positional else keywords["query"]).strip().lower()
                    chain = positional[1] if len(positional) > 1 else keywords.get("chain")
                    rows = [row for row in value["rows"] if (
                        not chain or row["chain_id"] == config.CHAIN_NAME_TO_ID.get(chain.lower())
                    ) and vault_search_tools._matches_query({
                        "address": row["vault_address"], "name": row.get("name"),
                        "symbol": row.get("symbol"), "asset": row.get("asset", {}),
                    }, search)]
                    sort = positional[2] if len(positional) > 2 else keywords.get("sort_by")
                    field = "estimated_apy" if sort in {"highest_apy", "lowest_apy", "highest_apr", "lowest_apr"} else "tvl_usd"
                    rows.sort(key=lambda row: (row.get(field) is None,
                        (row.get(field) or 0) * (1 if sort in {"lowest_apy", "lowest_apr"} else -1)))
                    offset, limit = keywords.get("offset", 0), keywords.get("limit", 5)
                    value["rows"] = rows[offset:offset+limit]
                    value["pagination"] = {"total": len(rows), "offset": offset, "limit": limit,
                        "returned": len(value["rows"]), "next_offset": offset+limit if offset+limit < len(rows) else None}
                    record["fixture_matched"] = bool(value["rows"])
                    record["returned_vaults"] = [(row["chain_id"], row["vault_address"]) for row in value["rows"]]
                calls.append(record)
                if matched:
                    return value
                error = "No matching evidence in this controlled evaluation fixture."
                return {"error": error} if name == "search_vaults" else error
            return invoke

        async def discover_data(path, params):
            assert path == "/api/discover", f"Unexpected dashboard I/O: {path}"
            payload = current["providers"].get("support_dashboard_discover")
            terms = current.get("provider_query_terms", {}).get("support_dashboard_discover", [])
            matched = isinstance(payload, dict) and (not terms or any(term.lower() in json.dumps(params).lower() for term in terms))
            calls.append({"tool": "support_dashboard_discover", "args": [], "kwargs": params, "fixture_matched": matched})
            payload = deepcopy(payload) if matched else {"rows": [], "filters": {}}
            rows = [row for row in payload["rows"]
                    if (params.get("chain_id") is None or row["chain_id"] == params["chain_id"])
                    and (not params.get("token_symbol") or row["token_symbol"].lower() == params["token_symbol"].lower())
                    and (params.get("min_tvl_usd") is None or row.get("tvl_usd") is not None and row["tvl_usd"] >= params["min_tvl_usd"])
                    and (params.get("market", "all") == "all" or row["market"] == params["market"])]
            field = {"tvl": "tvl_usd", "est_apy": "est_apy", "apy_30d": "realized_apy_30d", "momentum": "momentum_7d_30d"}[params.get("sort_by", "tvl")]
            rows.sort(key=lambda row: (row.get(field) is None,
                (row.get(field) or 0) * (1 if params.get("direction") == "asc" else -1)))
            with_realized_apy = sum(row.get("realized_apy_30d") is not None for row in rows)
            payload["coverage"] = {
                "visible_vaults": len(rows),
                "with_realized_apy": with_realized_apy,
                "without_realized_apy": len(rows) - with_realized_apy,
                "coverage_ratio": with_realized_apy / len(rows) if rows else None,
            }
            payload["rows"] = rows[:params.get("limit", 10)]
            payload["filters"].update(params)
            payload["pagination"] = {"limit": params.get("limit", 10), "total": len(rows)}
            return payload

        def observation(operation, query, *, transport, nonce=None):
            matched = next((entry for entry in current.get("lookups", []) if entry["operation"] == operation
                            and {k: v.lower() for k, v in entry["query"].items()} == {k: v.lower() for k, v in query.items()}), None)
            valid = (nonce is None or nonce == current.get("nonce")) and matched is not None
            calls.append({"tool": operation, "query": query, "fixture_matched": valid, "transport": transport})
            return matched if valid else None

        async def lookup(request):
            matched = observation(request.path_params["operation"], dict(request.query_params), transport="http", nonce=request.path_params["nonce"])
            valid = matched is not None
            return JSONResponse({"observation": matched["result"]} if valid else {"error": "No matching observation"}, status_code=200 if valid else 404)

        def rpc_observation(operation, query):
            matched = observation(operation, query, transport="internal_rpc")
            return (json.dumps(matched["result"]) if isinstance(matched["result"], (dict, list)) else str(matched["result"])) if matched else "No matching observation in the controlled provider fixture."

        async def inspect_onchain(**args):
            return rpc_observation("transaction", {"chain": args["chain"], "tx_hash": args.get("tx_hash", "")})

        async def positions(**args):
            return rpc_observation("positions", {"wallet": args["user_address_or_ens"]})

        async def withdrawal(**args):
            return rpc_observation("withdrawal", {"wallet": args.get("user_address_or_ens") or "", "chain": args["chain"], "vault": args["vault_address"]})

        async def wallet_history(**args):
            query = {"wallet": args["wallet_address"], "chain": args["chain"], "token": args["token_address"]}
            if args.get("tx_hashes_json"):
                query["transactions"] = ",".join(sorted(value.lower() for value in json.loads(args["tx_hashes_json"])))
            return rpc_observation("wallet_history", query)

        async def contract_batch(**args):
            batch = json.loads(args["calls_json"])
            vaults = {entry.get("to_address", "").lower() for entry in batch}
            addresses = [value for entry in batch for value in json.loads(entry.get("args_json", "[]")) if isinstance(value, str) and re.fullmatch(r"0x[0-9a-fA-F]{40}", value)]
            if len(vaults) != 1 or not addresses:
                return "No matching observation in the controlled provider fixture."
            matched = observation("contract_batch", {"chain": args["chain"], "tx_hash": args.get("tx_hash", ""), "vault": next(iter(vaults)), "wallet": addresses[0]}, transport="internal_rpc")
            record = calls[-1]
            record.update(fixture_matched=False, requests=batch)
            if matched is None:
                return "No matching observation in the controlled provider fixture."
            value = json.loads(matched["result"]) if isinstance(matched["result"], str) else deepcopy(matched["result"])
            selected = []
            for entry in batch:
                try:
                    abi = onchain_tools._build_function_abi(
                        function_signature=entry.get("function_signature"),
                        function_abi_json=entry.get("function_abi_json"),
                        output_types_json=entry.get("output_types_json"),
                    )
                    inputs = json.loads(entry.get("args_json") or "[]")
                    if not isinstance(inputs, list) or len(inputs) != len(abi["inputs"]):
                        raise ValueError("ABI argument count does not match its inputs.")
                    inputs = [onchain_tools._coerce_abi_argument(field, arg) for field, arg in zip(abi["inputs"], inputs)]
                    contract = Web3().eth.contract(address=Web3.to_checksum_address(entry["to_address"]), abi=[abi])
                    getattr(contract.functions, abi["name"])(*inputs)._encode_transaction_data()
                    signature = abi_to_signature(abi)
                    outputs = [collapse_if_tuple(field) for field in abi["outputs"]]
                    # Only this verified V3 overload defaults to zero loss.
                    lookup_signature = "maxWithdraw(address,uint256)" if signature == "maxWithdraw(address)" else signature
                    lookup_inputs = [*inputs, 0] if signature == "maxWithdraw(address)" else inputs
                    match = next((row for row in value["results"]
                                  if row["function"] == lookup_signature and row["output_types"] == outputs
                                  and [str(v).lower() for v in row["args"]] == [str(v).lower() for v in lookup_inputs]), None)
                    if match is None:
                        raise ValueError("No matching contract field observation.")
                    selected.append({**{k: v for k, v in match.items() if k != "output_types"}, "function": signature, "args": inputs})
                except Exception as exc:
                    selected.append({"ok": False, "error": str(exc)})
            record["fixture_matched"] = any(row["ok"] for row in selected)
            value["results"] = selected
            value["complete"] = all(row["ok"] for row in selected)
            return json.dumps(value)

        async def chain_state(**args):
            return rpc_observation("read_chain_state", args)

        async def boundary(message, **context):
            result = await evaluate_support_boundary(message, **context)
            if result.get("error"):
                raise RuntimeError("Boundary classifier failed")
            boundaries.append(result)
            return result

        async def telegram(method, payload):
            if method != "sendMessage":
                raise AssertionError(f"Unexpected external Telegram operation: {method}")
            return {"ok": True, "result": {"message_id": len(native_results), "chat": {"id": "evaluation"}}}

        async def resolve_target(address, *, chain_hint=None):
            is_wallet = any(entry["query"].get("wallet", "").lower() == address.lower() for entry in current.get("lookups", []))
            return yearn_targets.ResolvedYearnAddressTarget(address=address, kind="eoa_or_missing_code" if is_wallet else "unknown", chain=chain_hint)

        for name, (module, attribute) in providers.items():
            if name != "support_dashboard_discover":
                patches.enter_context(patch.object(module, attribute, new=provider(name)))
        patches.enter_context(patch.object(support_dashboard_tools, "_fetch_dashboard_json", new=discover_data))
        patches.enter_context(patch.object(ysupport, "_outer_support_boundary_result", new=boundary))
        patches.enter_context(patch.object(handoff, "_telegram_api_call", new=telegram))
        patches.enter_context(patch.object(yearn_targets, "resolve_yearn_address_target", new=resolve_target))
        patches.enter_context(patch.object(discord, "TextChannel", DiscordChannel))
        patches.enter_context(patch.object(onchain_tools, "core_inspect_onchain", new=inspect_onchain))
        patches.enter_context(patch.object(vault_position_tools, "core_check_all_deposits", new=positions))
        patches.enter_context(patch.object(vault_position_tools, "core_get_withdrawal_instructions", new=withdrawal))
        patches.enter_context(patch.object(chain_access, "core_read_chain_state", new=chain_state))
        patches.enter_context(patch.object(position_evidence, "core_get_wallet_history", new=wallet_history))
        patches.enter_context(patch.object(onchain_tools, "core_read_contracts", new=contract_batch))
        patches.enter_context(patch.multiple(config, CATEGORY_CONTEXT_MAP={1: "yearn"}, COOLDOWN_SECONDS=0,
                                            BUG_REPORT_COOLDOWN_SECONDS=0, TELEGRAM_BOT_TOKEN="fixture",
                                            TELEGRAM_YSUPPORT_CHAT="evaluation", TICKET_EXECUTION_CODEX_SESSION_DIR=str(root / "sessions"),
                                            RPC_URLS={name: "http://127.0.0.1:1/disabled" for name in config.CHAIN_NAME_TO_ID}))
        for attribute, path in {
            "_DISCORD_STATE_ROOT": root / "state", "_TICKET_STATE_DIR": root / "state/tickets",
            "_PUBLIC_STATE_DIR": root / "state/public", "_TELEGRAM_STATE_PATH": root / "state/telegram.json",
        }.items():
            patches.enter_context(patch.object(state, attribute, path))
        import socket
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        token = secrets.token_hex(24)
        server = mcp_server._build_mcp_server(host="127.0.0.1", port=port, api_key=token)
        declared = {tool.name: tool for tool in await mcp_server.mcp.list_tools()}
        assert set(declared) == set(providers), "Every production MCP tool needs controlled evaluation I/O"
        for name in providers:
            server.add_tool(getattr(mcp_server, name), annotations=declared[name].annotations)
        app = server.streamable_http_app()
        app.add_route("/evidence/{nonce}/{operation}", lookup, methods=["GET"])
        http = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
        http_task = asyncio.create_task(http.serve(sockets=[sock]))
        bot = None
        try:
            for _ in range(200):
                if http.started:
                    break
                if http_task.done():
                    await http_task
                    raise RuntimeError("Fixture MCP failed to start")
                await asyncio.sleep(0.01)
            if not http.started:
                raise RuntimeError("Fixture MCP startup timed out")
            child_env = {key: value for key, value in runtime_env.items() if key in {"PATH", "LANG", "SSL_CERT_FILE", "SSL_CERT_DIR"}}
            child_env.update(CODEX_HOME=str(root / "home"), HOME=str(root))
            if command_override:
                child_env.update(YSUPPORT_EVAL_MCP_URL=f"http://127.0.0.1:{port}/mcp", YSUPPORT_EVAL_MCP_TOKEN=token)
            command = command_override(cases[0]) if callable(command_override) else command_override
            endpoint = CodexSupportTicketExecutionJsonEndpoint(
                codex_command=command or config.TICKET_EXECUTION_CODEX_COMMAND or None,
                model=config.TICKET_EXECUTION_CODEX_MODEL, reasoning_effort=config.TICKET_EXECUTION_CODEX_REASONING_EFFORT,
                codex_home=root / "home", codex_auth_link_source=auth, session_dir=root / "sessions", session_max_age_hours=1,
                ysupport_mcp_url=f"http://127.0.0.1:{port}/mcp", mcp_server_api_key=token,
                web_search_mode="disabled", env=child_env, timeout_seconds=180,
            )
            with patch.object(ysupport, "build_ticket_execution_json_endpoint", return_value=endpoint):
                bot = ysupport.TicketBot(intents=discord.Intents.none())
            delegate = bot.investigation_executor

            async def execute(request, hooks=None):
                nonlocal native_calls
                native_calls += 1
                endpoints = {entry["operation"]: list(entry["query"]) for entry in current.get("lookups", []) if entry["operation"] == "report"}
                instruction = (
                    "Controlled evaluation. Use ysupport MCP for docs/data and the private internal_rpc command "
                    "for transactions, wallet positions and withdrawal details. Provider fixtures require exact "
                    "identifiers from the user's message or ticket history before returning observations. "
                    "Use read-only HTTP GET only for a report artifact at the data API below. "
                    "Do not query live sites/providers, invent missing facts or write files.\n"
                    f"Evaluation data API: http://127.0.0.1:{port}/evidence/{current['nonce']}\n"
                    f"Available paths and required query keys: {json.dumps(endpoints)}"
                )
                updated = replace(request, input_list=[*request.input_list[:-1], {"role": "system", "content": instruction}, request.input_list[-1]])
                result = await delegate.execute_turn(updated, hooks=hooks)
                native_results.append(result.flow_outcome)
                return result

            bot.investigation_executor = SimpleNamespace(execute_turn=execute)
            for index, case in enumerate(cases):
                if callable(command_override):
                    endpoint.codex_command = list(command_override(case))
                current.clear()
                current.update(case, nonce=secrets.token_hex(12))
                calls.clear()
                native_results.clear()
                native_calls = 0
                boundaries.clear()
                started = time.monotonic()
                channel = DiscordChannel(970000 + index)
                bot.get_channel = lambda _id: channel
                record = {"id": case["id"], "aliases": case.get("aliases", []), "turns": [], "failures": []}
                state.ticket_owner_user_id_by_channel[channel.id] = 777
                try:
                    for turn_index, turn in enumerate(case["turns"]):
                        if "history" in turn:
                            state.conversation_threads[channel.id] = list(turn["history"])
                        job = state.get_or_create_ticket_investigation_job(channel.id)
                        if "remembered_withdrawal_target" in turn:
                            job.remember_withdrawal_target(*turn["remembered_withdrawal_target"])
                        if turn.get("initial_button_intent"):
                            state.channel_intent_after_button[channel.id] = turn["initial_button_intent"]
                        was_waiting = state.is_ticket_waiting_for_team(channel.id)
                        previous_native, previous_boundary, previous_messages = len(native_results), len(boundaries), len(channel.messages)
                        previous_calls = native_calls
                        message = SimpleNamespace(id=turn_index + 1, channel=channel, reference=None, content=turn["message"], attachments=[], author=SimpleNamespace(id=777, bot=False, name="ticket-owner"))
                        await bot.on_message(message)
                        task = state.pending_tasks.get(channel.id)
                        if task is not None:
                            await task
                        ran = native_calls - previous_calls
                        outcome = native_results[-1] if len(native_results) > previous_native else None
                        waiting = state.is_ticket_waiting_for_team(channel.id)
                        visible = [item.content for item in channel.messages[previous_messages:] if not item.deleted]
                        latest_boundary = boundaries[-1] if len(boundaries) > previous_boundary else {}
                        delivery = "answered" if outcome is not None else "error" if ran else "queued_for_team" if was_waiting else "boundary" if latest_boundary else "stopped"
                        answer = "\n".join(visible)
                        persisted = state._TICKET_STATE_DIR / f"{channel.id}.json"
                        saved = json.loads(persisted.read_text()) if persisted.exists() else {}
                        saved_history = saved.get("history", [])
                        record["turns"].append({
                            "answer": answer, "discord_messages": visible, "delivery": delivery, "native_calls": ran,
                            "model_answer": outcome.raw_final_reply if outcome is not None else None,
                            "handoff": outcome.requires_human_handoff if outcome is not None else waiting,
                            "boundary": latest_boundary.get("classification"), "waiting_for_team": waiting,
                            "handoff_persisted": bool(saved.get("team_handoff_notice")) and saved.get("investigation_job", {}).get("mode") == "escalated_to_human",
                            "followup_persisted": any(item.get("role") == "user" and item.get("content") == turn["message"] for item in saved_history),
                            "runtime_error": state.stop_reasons_by_channel.get(channel.id) == "runtime_error",
                        })
                    record["failures"] = failures(case, calls, record["turns"])
                except Exception as exc:
                    record["failures"] = [f"Execution error: {type(exc).__name__}"]
                    logging.error("Evaluation failed for %s: %s", case["id"], type(exc).__name__)
                finally:
                    state.clear_ticket_channel_state(channel.id, delete_persisted=True)
                record["calls"] = list(calls)
                record["seconds"] = round(time.monotonic() - started, 3)
                report["cases"].append(record)
                write_report(output, report)
                print(json.dumps({"id": case["id"], "failures": record["failures"], "seconds": record["seconds"]}), flush=True)
        finally:
            if bot is not None:
                await bot.close()
            http.should_exit = True
            await http_task
            sock.close()
            await docs_repo_tools.close_shared_clients()
        report["complete"] = len(report["cases"]) == len(cases)
        report["passed"] = sum(not case["failures"] for case in report["cases"])
        write_report(output, report)
        return int(report["passed"] != len(cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", help="Run a stable scenario ID or alias")
    logging.getLogger().setLevel(logging.WARNING)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
