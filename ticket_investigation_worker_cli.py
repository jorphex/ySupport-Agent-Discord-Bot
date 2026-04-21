from ticket_investigation.worker_cli import *  # noqa: F401,F403


if __name__ == "__main__":
    import asyncio

    from ticket_investigation.worker_cli import _main

    raise SystemExit(asyncio.run(_main()))
