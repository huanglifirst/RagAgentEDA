from __future__ import annotations

from pathlib import Path
import json
import subprocess

from backend.runner.ssh_runner import RunOutput


class LocalTedRunner:
    def __init__(self, bashrc_path: str = '~/.bashrc') -> None:
        self.bashrc_path = bashrc_path

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
        return RunOutput(status=data.get('status', 'success'), metrics=data.get('metrics', {}), stdout=stdout, stderr=stderr)
