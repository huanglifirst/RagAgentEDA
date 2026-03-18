from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import re
import shlex
import subprocess
import uuid


_SAFE_CMD_TOKEN = re.compile(r"^[A-Za-z0-9._+-]+$")
_SAFE_MODULE_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


@dataclass
class SSHConfig:
    host: str
    user: str
    port: int = 22
    key_path: str = ""
    remote_work_dir: str = "/tmp/ragagent_tasks"
    remote_bashrc: str = "~/.bashrc"
    required_commands: tuple[str, ...] = ("python",)
    required_python_modules: tuple[str, ...] = ()


@dataclass
class RunOutput:
    status: str
    metrics: dict
    stdout: str
    stderr: str
    error: str = ""


@dataclass
class PreflightOutput:
    ok: bool
    stdout: str
    stderr: str
    error: str = ""


def validate_result_payload(data: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "result.json root must be an object"

    status = data.get("status")
    if not isinstance(status, str) or not status.strip():
        return False, "result.json missing non-empty status"

    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        return False, "result.json missing metrics object"
    if "bandwidth_hz" not in metrics:
        return False, "result.json metrics missing bandwidth_hz"
    try:
        bandwidth_hz = float(metrics["bandwidth_hz"])
    except (TypeError, ValueError):
        return False, "result.json metrics.bandwidth_hz must be numeric"

    notes = data.get("notes")
    if not isinstance(notes, list):
        return False, "result.json missing notes list"

    if status.strip().upper() == "PASS" and bandwidth_hz <= 0:
        return False, "PASS result must have bandwidth_hz > 0"
    return True, ""


class SSHRunner:
    def __init__(self, cfg: SSHConfig) -> None:
        self.cfg = cfg
        self.cfg.required_commands = tuple(c.strip() for c in cfg.required_commands if c and c.strip())
        self.cfg.required_python_modules = tuple(m.strip() for m in cfg.required_python_modules if m and m.strip())

    def preflight(self) -> PreflightOutput:
        try:
            checks = self._build_preflight_checks()
        except RuntimeError as exc:
            return PreflightOutput(ok=False, stdout="", stderr="", error=str(exc))

        shell_script = f"source {shlex.quote(self.cfg.remote_bashrc)} && " + " && ".join(checks)
        cmd = ["ssh", *self._ssh_opts(), self._remote_host(), "bash -lc " + shlex.quote(shell_script)]
        proc = self._run(cmd, allow_fail=True)
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip() or "remote preflight command failed"
            return PreflightOutput(ok=False, stdout=proc.stdout, stderr=proc.stderr, error=err)
        return PreflightOutput(ok=True, stdout=proc.stdout, stderr=proc.stderr)

    def run_script(self, local_script: Path, task_id: str) -> RunOutput:
        remote_dir = f"{self.cfg.remote_work_dir}/{task_id}"
        remote_target = f"{self._remote_prefix()}:{remote_dir}"

        self._run(["ssh", *self._ssh_opts(), self._remote_host(), f"mkdir -p {remote_dir}"])
        self._run(["scp", *self._scp_opts(), str(local_script), f"{remote_target}/main.py"])

        shell_script = (
            f"source {shlex.quote(self.cfg.remote_bashrc)} && "
            f"cd {shlex.quote(remote_dir)} && "
            "python main.py > stdout.log 2> stderr.log || true"
        )
        self._run(["ssh", *self._ssh_opts(), self._remote_host(), "bash -lc " + shlex.quote(shell_script)])

        local_tmp = local_script.parent
        self._run(["scp", *self._scp_opts(), f"{remote_target}/result.json", str(local_tmp / "result.json")], allow_fail=True)
        self._run(["scp", *self._scp_opts(), f"{remote_target}/stdout.log", str(local_tmp / "stdout.log")], allow_fail=True)
        self._run(["scp", *self._scp_opts(), f"{remote_target}/stderr.log", str(local_tmp / "stderr.log")], allow_fail=True)

        stdout = (local_tmp / "stdout.log").read_text(encoding="utf-8", errors="ignore") if (local_tmp / "stdout.log").exists() else ""
        stderr = (local_tmp / "stderr.log").read_text(encoding="utf-8", errors="ignore") if (local_tmp / "stderr.log").exists() else ""

        result_path = local_tmp / "result.json"
        if not result_path.exists():
            return RunOutput(status="failed", metrics={}, stdout=stdout, stderr=stderr, error="result.json not found")

        data = json.loads(result_path.read_text(encoding="utf-8"))
        ok, err = validate_result_payload(data)
        if not ok:
            return RunOutput(status="failed", metrics=data.get("metrics", {}), stdout=stdout, stderr=stderr, error=err)
        return RunOutput(status=data.get("status", "PASS"), metrics=data.get("metrics", {}), stdout=stdout, stderr=stderr)

    def _build_preflight_checks(self) -> list[str]:
        checks: list[str] = []
        for cmd in self.cfg.required_commands:
            if not _SAFE_CMD_TOKEN.match(cmd):
                raise RuntimeError(f"invalid command token in required_commands: {cmd}")
            checks.append(
                f"command -v {shlex.quote(cmd)} >/dev/null 2>&1 || {{ echo 'missing command: {cmd}' >&2; exit 12; }}"
            )

        for module in self.cfg.required_python_modules:
            if not _SAFE_MODULE_TOKEN.match(module):
                raise RuntimeError(f"invalid module token in required_python_modules: {module}")
            checks.append(
                f"python -c \"import {module}\" >/dev/null 2>&1 || {{ echo 'missing python module: {module}' >&2; exit 13; }}"
            )

        checks.append("echo preflight_ok")
        return checks

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
