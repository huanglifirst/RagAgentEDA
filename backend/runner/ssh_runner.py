from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
import uuid


@dataclass
class SSHConfig:
    host: str
    user: str
    port: int = 22
    key_path: str = ""
    remote_work_dir: str = "/tmp/ragagent_tasks"


@dataclass
class RunOutput:
    status: str
    metrics: dict
    stdout: str
    stderr: str
    error: str = ""


class SSHRunner:
    def __init__(self, cfg: SSHConfig) -> None:
        self.cfg = cfg

    def run_script(self, local_script: Path, task_id: str) -> RunOutput:
        remote_dir = f"{self.cfg.remote_work_dir}/{task_id}"
        remote_target = f"{self._remote_prefix()}:{remote_dir}"

        self._run(["ssh", *self._ssh_opts(), self._remote_host(), f"mkdir -p {remote_dir}"])
        self._run(["scp", *self._scp_opts(), str(local_script), f"{remote_target}/main.py"])

        cmd = (
            "bash -lc '"
            "source ~/.bashrc && "
            f"cd {remote_dir} && "
            "python main.py > stdout.log 2> stderr.log || true"
            "'"
        )
        self._run(["ssh", *self._ssh_opts(), self._remote_host(), cmd])

        local_tmp = local_script.parent
        self._run(["scp", *self._scp_opts(), f"{remote_target}/result.json", str(local_tmp / "result.json")], allow_fail=True)
        self._run(["scp", *self._scp_opts(), f"{remote_target}/stdout.log", str(local_tmp / "stdout.log")], allow_fail=True)
        self._run(["scp", *self._scp_opts(), f"{remote_target}/stderr.log", str(local_tmp / "stderr.log")], allow_fail=True)

        stdout = (local_tmp / "stdout.log").read_text(encoding="utf-8", errors="ignore") if (local_tmp / "stdout.log").exists() else ""
        stderr = (local_tmp / "stderr.log").read_text(encoding="utf-8", errors="ignore") if (local_tmp / "stderr.log").exists() else ""

        if (local_tmp / "result.json").exists():
            data = json.loads((local_tmp / "result.json").read_text(encoding="utf-8"))
            return RunOutput(status=data.get("status", "success"), metrics=data.get("metrics", {}), stdout=stdout, stderr=stderr)

        return RunOutput(status="failed", metrics={}, stdout=stdout, stderr=stderr, error="result.json not found")

    def _remote_prefix(self) -> str:
        return self._remote_host()

    def _remote_host(self) -> str:
        return f"{self.cfg.user}@{self.cfg.host}" if self.cfg.user else self.cfg.host

    def _ssh_opts(self) -> list[str]:
        opts = ["-p", str(self.cfg.port)]
        if self.cfg.key_path:
            opts.extend(["-i", self.cfg.key_path])
        return opts

    def _scp_opts(self) -> list[str]:
        opts = ["-P", str(self.cfg.port)]
        if self.cfg.key_path:
            opts.extend(["-i", self.cfg.key_path])
        return opts

    @staticmethod
    def _run(cmd: list[str], allow_fail: bool = False) -> subprocess.CompletedProcess:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 and not allow_fail:
            raise RuntimeError(f"command failed: {' '.join(cmd)}\nstdout={proc.stdout}\nstderr={proc.stderr}")
        return proc


def new_task_id() -> str:
    return f"task_{uuid.uuid4().hex[:10]}"
