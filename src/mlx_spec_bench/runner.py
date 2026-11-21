"""Public Python API and durable sweep artifacts."""

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from typing import Any, Literal, TypedDict
from uuid import uuid4

from .config import Config, Sweep
from .io import atomic_write, json_text, read_json, write_json
from .memory import Sampler, environment
from .schema import validate_record
from .supervisor import supervise
from .trial import RUNTIME_RESERVE_BYTES, memory_allowance


class RunRecord(TypedDict):
    schema_version: int
    sweep_id: str
    case_id: str
    run_id: str
    repetition: int
    run_order: int
    status: Literal["success", "failure"]
    started_at: str
    wall_time_ns: int
    load_ns: int | None
    warmup_ns: int | None
    quality_ns: int | None
    warmups: int
    config: dict[str, Any]
    identities: dict[str, Any]
    environment: dict[str, Any]
    process_memory: dict[str, Any]
    metrics: dict[str, Any] | None
    details: str | None
    error: dict[str, str] | None


@dataclass(frozen=True)
class CaseResult:
    directory: Path
    manifest: dict
    records: tuple[RunRecord, ...]

    @property
    def status(self) -> str:
        return self.manifest["status"]

    def detail(self, record: RunRecord) -> dict | None:
        return read_json(self.directory / record["details"]) if record["details"] else None


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id(sweep_id: str, case_id: str, repetition: int) -> str:
    return f"{sweep_id}-{case_id}-r{repetition:03d}"


def _one(config: Config, warmups: int, directory: Path, ids: dict) -> RunRecord:
    started_at = _utc()
    start = perf_counter_ns()
    result = {}
    observations = Sampler(None).observations
    error = None
    log = directory / "worker-logs" / (ids["run_id"] + ".log")
    with TemporaryDirectory(prefix=".trial-", dir=directory) as temporary:
        request_path = Path(temporary) / "request.json"
        output_path = Path(temporary) / "result.json"
        write_json(request_path, {"config": config.to_dict(), "warmups": warmups})
        try:
            allowance = memory_allowance(config)
            if allowance < RUNTIME_RESERVE_BYTES:
                error = {
                    "kind": "preflight",
                    "message": f"memory_limit_bytes={config.memory_limit_bytes} gives a {allowance}-byte allowance "
                               f"below the {RUNTIME_RESERVE_BYTES}-byte runtime reserve; increase memory_limit_bytes",
                }
            else:
                outcome = supervise(
                    [sys.executable, "-m", "mlx_spec_bench.worker", str(request_path), str(output_path)],
                    timeout=config.timeout_seconds, allowance=allowance, log=log,
                )
                observations = outcome.observations
                error = outcome.error
                if error is None:
                    result = read_json(output_path)
                    error = result["error"]
        except (OSError, ValueError, KeyError) as exc:
            error = {"kind": "worker_exit", "message": f"worker evidence unavailable: {exc}; see {log.name}"}
        except KeyboardInterrupt:
            error = {"kind": "interrupted", "message": "sweep interrupted while preparing worker evidence"}

    details_path = None
    if error is None:
        details_path = "details/" + ids["run_id"] + ".json"
        write_json(directory / details_path, {**result["data"], **ids})
    return {
        "schema_version": 1, **ids, "status": "failure" if error else "success",
        "started_at": started_at, "wall_time_ns": perf_counter_ns() - start, "config": config.to_dict(),
        "load_ns": result.get("load_ns"), "warmup_ns": result.get("warmup_ns"), "quality_ns": result.get("quality_ns"),
        "warmups": warmups, "environment": result.get("environment") or environment(),
        "process_memory": observations, "metrics": None if error else result["metrics"],
        "identities": result.get("identities", dict.fromkeys(("target", "draft", "suite", "evaluation"))),
        "details": details_path, "error": error,
    }


def run(config: Config, *, output_dir: Path | str = "runs", warmups: int = 0,
        repetitions: int = 1) -> CaseResult:
    """Run one case in fresh worker processes and return each recorded repetition."""
    return run_sweep(Sweep((config,), warmups, repetitions), output_dir=output_dir)


def run_sweep(sweep: Sweep, *, output_dir: Path | str = "runs", host_notes: str | None = None,
              source_commit: str | None = None, on_record=None) -> CaseResult:
    """Execute the declared matrix in order, persisting each terminal row."""
    sweep.validate()
    cases = [Config.from_dict(config.to_dict()) for config in sweep.cases]
    sweep_id = uuid4().hex
    directory = Path(output_dir).resolve() / ("sweep-" + sweep_id)
    (directory / "details").mkdir(parents=True, exist_ok=False)
    (directory / "worker-logs").mkdir()
    manifest_path = directory / "manifest.json"
    journal = directory / "results.jsonl"
    expected = []
    case_map = {case.case_id: case for case in cases}
    for case in cases:
        for repetition in range(1, sweep.repetitions + 1):
            expected.append({
                "case_id": case.case_id, "run_id": _run_id(sweep_id, case.case_id, repetition),
                "repetition": repetition, "run_order": len(expected) + 1,
            })
    observer = Sampler(None)
    observer.sample()
    manifest = {
        "schema_version": 1, "sweep_id": sweep_id, "started_at": _utc(), "finished_at": None,
        "status": "running", "warmups": sweep.warmups, "repetitions": sweep.repetitions,
        "cases": {key: config.to_dict() for key, config in case_map.items()},
        "expected": expected, "host_notes": host_notes, "environment": environment(),
        "source_commit": source_commit or os.environ.get("GITHUB_SHA"),
        "before": observer.observations.copy(), "after": None,
    }
    write_json(manifest_path, manifest)
    atomic_write(journal, "")
    interrupted = False
    records = []
    lines = []
    try:
        for ids in expected:
            record = _one(case_map[ids["case_id"]], sweep.warmups, directory, {"sweep_id": sweep_id, **ids})
            validate_record(record)
            lines.append(json_text(record) + "\n")
            atomic_write(journal, "".join(lines))
            records.append(record)
            if on_record is not None:
                on_record(record, directory)
            if record["error"] and record["error"]["kind"] == "interrupted":
                interrupted = True
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        observer = Sampler(None)
        observer.sample()
        manifest["finished_at"] = _utc()
        manifest["after"] = observer.observations
        if interrupted:
            manifest["status"] = "interrupted"
        elif len(records) != len(expected) or any(record["status"] != "success" for record in records):
            manifest["status"] = "failure"
        else:
            manifest["status"] = "success"
        write_json(manifest_path, manifest)
    return CaseResult(directory, manifest, tuple(records))
