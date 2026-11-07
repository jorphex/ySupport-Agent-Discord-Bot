from __future__ import annotations

from pathlib import Path
import shutil
import stat
import tempfile
import uuid


class TicketExecutionWorkspace:
    def __init__(
        self,
        *,
        artifact_dir: str | None = None,
        prefix: str,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.prefix = prefix
        self._tempdir = tempfile.TemporaryDirectory(prefix=prefix)
        self.run_dir: Path | None = None
        self._exported_run_dir: Path | None = None

    def __enter__(self) -> Path:
        self.run_dir = Path(self._tempdir.__enter__())
        return self.run_dir

    def __exit__(self, exc_type, exc, tb) -> None:
        return self._tempdir.__exit__(exc_type, exc, tb)

    def export_copy(self) -> Path | None:
        if self._exported_run_dir is not None:
            return self._exported_run_dir
        if self.run_dir is None:
            raise RuntimeError("Ticket execution workspace has not been entered.")
        if not self.artifact_dir:
            return None

        export_root = Path(self.artifact_dir)
        export_root.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        export_dir = export_root / f"run-{run_id}"
        staging_dir = export_root / f".run-{run_id}.tmp"
        try:
            shutil.copytree(self.run_dir, staging_dir)
            self._make_read_only(staging_dir)
            staging_dir.replace(export_dir)
        except Exception:
            self._remove_export_tree(staging_dir)
            self._remove_export_tree(export_dir)
            raise
        self._exported_run_dir = export_dir
        return export_dir

    def _make_read_only(self, export_dir: Path) -> None:
        for path in sorted(export_dir.rglob("*"), reverse=True):
            if path.is_dir():
                path.chmod(0o555)
            else:
                current_mode = path.stat().st_mode
                executable_bits = current_mode & (
                    stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
                )
                path.chmod(0o444 | executable_bits)
        export_dir.chmod(0o555)

    @staticmethod
    def _remove_export_tree(path: Path) -> None:
        if not path.exists():
            return
        path.chmod(0o755)
        for child in path.rglob("*"):
            if child.is_dir():
                child.chmod(0o755)
            else:
                child.chmod(0o644)
        shutil.rmtree(path)
