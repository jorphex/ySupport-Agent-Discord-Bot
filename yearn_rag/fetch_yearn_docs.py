"""Refresh the generated docs checkout without retaining full history or assets."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


SOURCE_URL = "https://github.com/yearn/yearn-devdocs.git"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, timeout=180
    ).stdout.strip()


def refresh_yearn_docs(target: Path) -> bool:
    target = target.absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise RuntimeError("Refusing to replace a symlinked docs checkout.")
    original_head = None
    if target.exists():
        if not (target / ".git").is_dir():
            raise RuntimeError(
                "Existing docs directory is not a generated Git checkout."
            )
        if _git("-C", str(target), "status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError(
                "Docs checkout has local changes; refusing to discard them."
            )
        origin = _git("-C", str(target), "config", "--get", "remote.origin.url")
        if origin.rstrip("/").removesuffix(".git") not in {
            SOURCE_URL.removesuffix(".git"),
            "git@github.com:yearn/yearn-devdocs",
        }:
            raise RuntimeError("Existing docs checkout has an unexpected origin.")
        original_head = _git("-C", str(target), "rev-parse", "HEAD")
        if (
            original_head
            != _git("-C", str(target), "rev-parse", "refs/remotes/origin/master")
            or _git(
                "-C", str(target), "for-each-ref", "--format=%(refname)", "refs/heads"
            )
            != "refs/heads/master"
        ):
            raise RuntimeError(
                "Docs checkout has local commits or branches; retaining it."
            )

    tip = _git("ls-remote", "--exit-code", SOURCE_URL, "refs/heads/master").split()[0]
    if target.exists():
        sparse = subprocess.run(
            ["git", "-C", str(target), "sparse-checkout", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if (
            _git("-C", str(target), "rev-parse", "--is-shallow-repository") == "true"
            and sparse.returncode == 0
            and sparse.stdout.strip() == "docs"
            and _git("-C", str(target), "rev-parse", "HEAD") == tip
        ):
            print(f"Docs checkout unchanged at {tip}.")
            return False

    with tempfile.TemporaryDirectory(prefix=".yearn-docs-", dir=target.parent) as temp:
        staging = Path(temp) / "checkout"
        _git(
            "clone",
            "--depth=1",
            "--filter=blob:none",
            "--sparse",
            "--single-branch",
            "--branch",
            "master",
            SOURCE_URL,
            str(staging),
        )
        _git("-C", str(staging), "sparse-checkout", "set", "docs")
        docs = staging / "docs"
        if not any(p.suffix in {".md", ".mdx"} for p in docs.rglob("*")):
            raise RuntimeError(
                "Fetched docs source is empty; retaining the previous checkout."
            )
        for path in docs.rglob("*"):
            if path.is_symlink():
                raise RuntimeError("Fetched docs source contains a symlink.")
            if not path.is_file():
                continue
            previous = target / path.relative_to(staging)
            if previous.is_file() and not previous.is_symlink():
                stat = previous.stat()
                if previous.read_bytes() == path.read_bytes():
                    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

        revision = _git("-C", str(staging), "rev-parse", "HEAD")
        backup = Path(temp).with_suffix(".previous")
        if target.exists():
            if (
                _git(
                    "-C", str(target), "status", "--porcelain", "--untracked-files=all"
                )
                or _git("-C", str(target), "rev-parse", "HEAD") != original_head
            ):
                raise RuntimeError(
                    "Docs checkout changed during refresh; retaining it."
                )
            target.rename(backup)
        elif original_head is not None:
            raise RuntimeError("Docs checkout disappeared during refresh.")
        try:
            staging.rename(target)
        except OSError:
            if backup.exists():
                backup.rename(target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        print(f"Refreshed shallow docs checkout at {revision}.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("yearn-devdocs"),
    )
    refresh_yearn_docs(parser.parse_args().target)
