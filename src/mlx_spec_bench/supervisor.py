"""Time-bound worker supervision and independent memory sampling."""

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
from time import monotonic

from .memory import SAMPLING_MS, Sampler

TEARDOWN_SECONDS = 1.0


@dataclass
class Outcome:
    observations: dict
    error: dict | None


def _stop_process_group(process: subprocess.Popen) -> str:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=TEARDOWN_SECONDS / 2)
        except subprocess.TimeoutExpired:
            continue
        # Ensure descendants in the group cannot survive their lead process.
        if sig == signal.SIGTERM:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        break
    return " Worker did not exit during teardown." if process.poll() is None else ""


def supervise(command: list[str], *, timeout: float, allowance: int, log: Path) -> Outcome:
    deadline = monotonic() + timeout
    sampler = Sampler(None)
    sampler.sample()
    error = None
    with log.open("wb") as output:
        process = subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )
        sampler.pid = process.pid
        sampler.parent_pid = os.getpid()
        try:
            while True:
                observed = sampler.sample()
                remaining = deadline - monotonic()
                if remaining > 0 and observed is not None and observed > allowance:
                    error = {
                        "kind": "memory_limit",
                        "message": f"observed process-tree memory {observed} bytes exceeds {allowance}-byte allowance; "
                                   "reduce model size, batch_size, k, or cache budget",
                    }
                    break
                if remaining > 0:
                    try:
                        status = process.wait(timeout=min(remaining, SAMPLING_MS / 1000))
                    except subprocess.TimeoutExpired:
                        continue
                if remaining <= 0 or monotonic() >= deadline:
                    error = {"kind": "timeout", "message": f"worker exceeded {timeout:g}s deadline; see {log.name}"}
                elif status:
                    error = {"kind": "worker_exit", "message": f"worker exit status {status}; see {log.name}"}
                break
        except KeyboardInterrupt:
            error = {"kind": "interrupted", "message": "sweep interrupted while worker was active"}
        finally:
            teardown = _stop_process_group(process)
    if teardown and error:
        error["message"] += teardown
    return Outcome(sampler.observations, error)
