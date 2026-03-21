from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import uuid


class TicketExecutionWorkspace:
    def __init__(
        self,
        *,
        artifact_dir: str | None = None,
        run_dir_root: str | None = None,
        prefix: str,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.run_dir_root = run_dir_root
        self.prefix = prefix
        self._tempdir = tempfile.TemporaryDirectory(prefix=prefix)
        self.run_dir: Path | None = None
        self._exported_run_dir: Path | None = None

    def __enter__(self) -> Path:
        self.run_dir = Path(self._tempdir.__enter__())
        return self.run_dir

    def __exit__(self, exc_type, exc, tb) -> None:
        return self._tempdir.__exit__(exc_type, exc, tb)

    @property
    def captures_artifacts(self) -> bool:
        return bool(self.artifact_dir)

    @property
    def export_root(self) -> str | None:
        return self.artifact_dir or self.run_dir_root

    def export_copy(self) -> Path | None:
        if self._exported_run_dir is not None:
            return self._exported_run_dir
        if self.run_dir is None:
            raise RuntimeError("Ticket execution workspace has not been entered.")
        if not self.export_root:
            return None

        export_dir = Path(self.export_root) / f"run-{uuid.uuid4().hex}"
        shutil.copytree(self.run_dir, export_dir)
        self._exported_run_dir = export_dir
        return export_dir
