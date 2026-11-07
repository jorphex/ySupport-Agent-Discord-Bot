"""Private per-turn RPC access; the command-line client imports only stdlib.

The bot owns providers and executes existing read helpers. Native subprocesses
receive only this client's command and a temporary Unix socket path, never RPC
credentials. This is independent of the MCP server.
"""

import argparse
import asyncio
import inspect
import json
import os
from pathlib import Path
import re
import socket
import sys


class InternalRpcServer:
    MAX_OPERATIONS = 8
    MAX_RPC_CALLS = 4096
    MAX_PACKET_BYTES = 32768
    MAX_RESULT_CHARS = 60000
    MAX_CLIENTS = 4
    OPERATIONS = {
        "inspect_onchain": {
            "required": ["chain", "mode"],
            "modes": ["call", "receipt", "logs", "tx_summary", "tx_investigate"],
            "optional": ["tx_hash", "to_address", "function_signature", "args_json", "output_types_json", "function_abi_json", "address", "topics_json", "from_block", "to_block", "event_abis_json", "block_identifier", "max_results"],
        },
        "check_all_deposits": {
            "required": ["user_address_or_ens"], "optional": ["token_symbol", "chain"],
            "description": "Scan wallet and gauge share balances, including older Yearn vaults. Use known chain and token/receipt-symbol filters; broad scans consume more RPC calls. For a known vault, prefer targeted reads or get_withdrawal_instructions. Balances do not establish recovery/distribution rights; use authoritative process documentation for those.",
        },
        "get_withdrawal_instructions": {
            "required": ["vault_address", "chain"], "optional": ["user_address_or_ens"],
        },
        "read_chain_state": {
            "required": ["chain", "method", "params_json"], "optional": [],
            "description": "Basic native balance, nonce, fee, block, code and storage reads. params_json is a JSON array of RPC parameters. Use specialised helpers for decoded transactions, contract calls, logs and wallet position scans.",
        },
    }

    def __init__(self, run_dir, *, timeout_seconds=300):
        self.run_dir = Path(run_dir)
        self.socket_path = self.run_dir / "internal_rpc.sock"
        self.timeout_seconds = timeout_seconds
        self.operation_count = 0
        self.reads = []
        self.tasks = set()
        self.lock = asyncio.Lock()
        self.server = None
        self.budget = None

    async def __aenter__(self):
        from ysupport.tools.chain_access import RpcReadBudget

        self.budget = RpcReadBudget(max_calls=self.MAX_RPC_CALLS, timeout_seconds=self.timeout_seconds)
        self.server = await asyncio.start_unix_server(self._handle, path=self.socket_path, limit=self.MAX_PACKET_BYTES)
        self.socket_path.chmod(0o600)
        return self

    async def __aexit__(self, *_args):
        await self.close()

    @property
    def description(self):
        return {
            "command": [sys.executable, "-B", str(Path(__file__).resolve()), str(self.socket_path)],
            "input": "Send one JSON object on stdin: {operation: name, arguments: object}.",
            "operations": {**self.OPERATIONS, "read_chain_state": {
                **self.OPERATIONS["read_chain_state"], "methods": sorted(self.budget.DIRECT_READ_METHODS),
            }},
            "limits": {"operations": self.MAX_OPERATIONS, "rpc_calls": self.MAX_RPC_CALLS, "log_blocks": 5000},
        }

    async def close(self):
        if self.budget is not None:
            self.budget.close()
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        active = list(self.tasks)
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        self.socket_path.unlink(missing_ok=True)
        if self.budget is not None:
            (self.run_dir / "internal_rpc_usage.json").write_text(json.dumps({
                "operations": self.operation_count, "rpc_calls": self.budget.used,
                "reads": self.reads,
                "budget_exhausted": self.budget.exhausted,
            }))

    async def _handle(self, reader, writer):
        if len(self.tasks) >= self.MAX_CLIENTS:
            writer.transport.abort()
            return
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            reply = await self._reply(reader)
            writer.write((json.dumps(reply) + "\n").encode())
            await asyncio.wait_for(writer.drain(), timeout=5)
        except (ConnectionError, asyncio.TimeoutError):
            pass
        finally:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), timeout=1)
            except (ConnectionError, asyncio.TimeoutError):
                writer.transport.abort()
            except asyncio.CancelledError:
                writer.transport.abort()
                raise
            finally:
                self.tasks.discard(task)

    async def _reply(self, reader):
        try:
            packet = await asyncio.wait_for(reader.readline(), timeout=5)
            query = json.loads(packet)
            async with self.lock:
                if reader.at_eof():
                    raise ValueError("Internal read caller disconnected.")
                if self.operation_count >= self.MAX_OPERATIONS:
                    raise ValueError("Internal operation limit reached.")
                self.operation_count += 1
                execution = asyncio.create_task(self._execute(query))
                disconnected = asyncio.create_task(reader.read(1))
                try:
                    done, _ = await asyncio.wait({execution, disconnected}, return_when=asyncio.FIRST_COMPLETED)
                    if disconnected in done and not execution.done():
                        self.budget.close()
                        raise ValueError("Internal read caller disconnected; observation is incomplete.")
                    result = await execution
                finally:
                    for pending in (execution, disconnected):
                        if not pending.done():
                            pending.cancel()
                    await asyncio.gather(execution, disconnected, return_exceptions=True)
            reply = {"ok": True, "result": result, "rpc_calls": self.budget.used, "budget_exhausted": self.budget.exhausted}
        except (ValueError, TypeError) as exc:
            reply = {"ok": False, "error": self._redact(str(exc))[:500]}
        except asyncio.TimeoutError:
            self.budget.close()
            reply = {"ok": False, "error": "Internal RPC timed out; observation is incomplete."}
        except Exception:
            reply = {"ok": False, "error": "Internal RPC read failed; observation is incomplete."}
        return reply

    async def _execute(self, query):
        from ysupport import config
        from ysupport.tools import chain_access

        if not isinstance(query, dict) or set(query) != {"operation", "arguments"}:
            raise ValueError("Expected operation and arguments only.")
        operation, args = query["operation"], query["arguments"]
        if not isinstance(operation, str) or operation not in self.OPERATIONS or not isinstance(args, dict):
            raise ValueError("Unknown internal read operation or invalid arguments.")
        args = dict(args)
        for name in ("chain", "mode"):
            if name in args:
                if not isinstance(args[name], str):
                    raise ValueError("Internal chain and mode must be names.")
                args[name] = args[name].strip().lower()
        if operation == "get_withdrawal_instructions":
            args.setdefault("user_address_or_ens", None)
        schema = self.OPERATIONS[operation]
        if set(args) - set(schema["required"] + schema["optional"]):
            raise ValueError("Unsupported internal read arguments.")
        if not set(schema["required"]) <= set(args):
            raise ValueError("Missing required internal read arguments.")
        if any(value is not None and not isinstance(value, (str, int)) for value in args.values()):
            raise ValueError("Internal read arguments must be strings, integers or null.")
        if "chain" in args and args["chain"] not in config.RPC_URLS:
            raise ValueError("Unsupported internal RPC chain.")
        if operation == "inspect_onchain" and args["mode"] not in schema["modes"]:
            raise ValueError("Unsupported internal read mode.")
        if operation == "inspect_onchain":
            if args["mode"] in {"receipt", "tx_summary", "tx_investigate"} and not args.get("tx_hash"):
                raise ValueError("Transaction reads require a transaction hash.")
            if args["mode"] == "call" and (not args.get("to_address") or not (args.get("function_signature") or args.get("function_abi_json"))):
                raise ValueError("Contract reads require a contract and function signature or ABI.")
            if args["mode"] == "logs" and not args.get("address"):
                raise ValueError("Log reads require a contract address.")
        if operation == "check_all_deposits" and not isinstance(args["user_address_or_ens"], str):
            raise ValueError("Position scans require a wallet address or ENS name.")
        for name in ["tx_hash", "address", "to_address", "vault_address"]:
            if args.get(name) is not None:
                length = 64 if name == "tx_hash" else 40
                if not isinstance(args[name], str) or not re.fullmatch(r"0x[0-9a-fA-F]{" + str(length) + "}", args[name]):
                    raise ValueError("Invalid internal read identifier.")
        if "max_results" in args and (type(args["max_results"]) is not int or not 1 <= args["max_results"] <= 25):
            raise ValueError("Internal read result limit must be between 1 and 25.")
        if operation == "inspect_onchain":
            from ysupport.tools.onchain_tools import core_inspect_onchain as helper
        elif operation == "check_all_deposits":
            from ysupport.tools.vault_position_tools import core_check_all_deposits as helper
        elif operation == "read_chain_state":
            from ysupport.tools.chain_access import core_read_chain_state as helper
        else:
            from ysupport.tools.vault_position_tools import core_get_withdrawal_instructions as helper
        inspect.signature(helper).bind(**args)
        self.reads.append(operation)
        with chain_access.rpc_read_scope(self.budget):
            result = await asyncio.wait_for(helper(**args), timeout=min(90, self.timeout_seconds))
        text = self._redact(str(result))
        if len(text) > self.MAX_RESULT_CHARS:
            text = text[:self.MAX_RESULT_CHARS] + "\nObservation truncated; additional positions or evidence may be omitted."
        return text

    @staticmethod
    def _redact(text):
        for key, value in os.environ.items():
            if value and (key.endswith("_RPC_URL") or key == "ALCHEMY_KEY"):
                text = text.replace(value, "[internal provider]")
        return text


def main():
    parser = argparse.ArgumentParser(description="Send one read request to this turn's private RPC connection.")
    parser.add_argument("socket_path")
    args = parser.parse_args()
    try:
        packet = sys.stdin.buffer.read(InternalRpcServer.MAX_PACKET_BYTES + 1)
        if len(packet) > InternalRpcServer.MAX_PACKET_BYTES:
            raise ValueError("Internal read request is too large.")
        query = json.loads(packet)
        with socket.socket(socket.AF_UNIX) as client:
            client.settimeout(100)
            client.connect(args.socket_path)
            client.sendall(json.dumps(query).encode() + b"\n")
            with client.makefile("rb") as stream:
                response = json.loads(stream.readline(InternalRpcServer.MAX_RESULT_CHARS * 8))
        print(json.dumps(response))
        return int(not response.get("ok"))
    except (OSError, ValueError):
        print(json.dumps({"ok": False, "error": "Internal read connection unavailable or request invalid."}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
