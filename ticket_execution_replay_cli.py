from ticket_execution.replay_cli import *  # noqa: F401,F403
from ticket_execution.replay_cli import _main


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(_main()))
