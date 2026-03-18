from __future__ import annotations

from pathlib import Path
import re
import shlex
import json
import subprocess

from backend.runner.ssh_runner import PreflightOutput, RunOutput, validate_result_payload


_SAFE_CMD_TOKEN = re.compile(r"^[A-Za-z0-9._+-]+$")
_SAFE_MODULE_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


class LocalTedRunner:
    def __init__(
        self,
        bashrc_path: str = '~/.bashrc',
        required_commands: tuple[str, ...] = ("python",),
        required_python_modules: tuple[str, ...] = (),
    ) -> None:
        self.bashrc_path = bashrc_path
        self.required_commands = tuple(c.strip() for c in required_commands if c and c.strip())
        self.required_python_modules = tuple(m.strip() for m in required_python_modules if m and m.strip())

    def preflight(self) -> PreflightOutput:
        checks: list[str] = []
        for cmd in self.required_commands:
            if not _SAFE_CMD_TOKEN.match(cmd):
                return PreflightOutput(ok=False, stdout="", stderr="", error=f"invalid required command token: {cmd}")
            checks.append(f"command -v {shlex.quote(cmd)} >/dev/null 2>&1 || {{ echo 'missing command: {cmd}' >&2; exit 12; }}")

        for module in self.required_python_modules:
            if not _SAFE_MODULE_TOKEN.match(module):
                return PreflightOutput(ok=False, stdout="", stderr="", error=f"invalid required module token: {module}")
            checks.append(
                f"python -c \"import {module}\" >/dev/null 2>&1 || {{ echo 'missing python module: {module}' >&2; exit 13; }}"
            )

        checks.append("echo preflight_ok")
        shell_script = f"source {shlex.quote(self.bashrc_path)} && " + " && ".join(checks)
        cmd = f"bash -lc {shlex.quote(shell_script)}"
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip() or "local preflight command failed"
            return PreflightOutput(ok=False, stdout=proc.stdout, stderr=proc.stderr, error=err)
        return PreflightOutput(ok=True, stdout=proc.stdout, stderr=proc.stderr)

    def run_script(self, local_script: Path, task_id: str) -> RunOutput:  # noqa: ARG002
        task_dir = local_script.parent
        stdout_path = task_dir / 'stdout.log'
        stderr_path = task_dir / 'stderr.log'

        cmd = (
            "bash -lc '"
            f"source {self.bashrc_path} && "
            f"cd {task_dir} && "
            "python main.py > stdout.log 2> stderr.log || true"
            "'"
        )
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if proc.returncode != 0:
            return RunOutput(status='failed', metrics={}, stdout=proc.stdout, stderr=proc.stderr, error='local runner shell failed')

        stdout = stdout_path.read_text(encoding='utf-8', errors='ignore') if stdout_path.exists() else ''
        stderr = stderr_path.read_text(encoding='utf-8', errors='ignore') if stderr_path.exists() else ''

        result_path = task_dir / 'result.json'
        if not result_path.exists():
            return RunOutput(status='failed', metrics={}, stdout=stdout, stderr=stderr, error='result.json not found')

        data = json.loads(result_path.read_text(encoding='utf-8'))
        ok, err = validate_result_payload(data)
        if not ok:
            return RunOutput(status='failed', metrics=data.get('metrics', {}), stdout=stdout, stderr=stderr, error=err)
        return RunOutput(status=data.get('status', 'PASS'), metrics=data.get('metrics', {}), stdout=stdout, stderr=stderr)
