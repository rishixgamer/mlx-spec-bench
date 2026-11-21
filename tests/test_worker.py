from dataclasses import replace
import sys
from time import monotonic


from mlx_spec_bench import Config, ModelSpec, Sweep, run_sweep
from mlx_spec_bench.schema import read_manifest, read_records
from mlx_spec_bench.supervisor import supervise

WORKER = """
from pathlib import Path
import sys
from mlx_spec_bench.config import Config
from mlx_spec_bench.io import read_json, write_json
from mlx_spec_bench import trial
def allocation_failure(**kwargs):
    raise MemoryError("simulated allocation failed")
trial.initialize = allocation_failure
parameters = read_json(Path(sys.argv[1]))
result = trial.execute(Config.from_dict(parameters["config"]), 0)
write_json(Path(sys.argv[2]), result)
"""


def test_worker_failures_do_not_corrupt_records_or_stop_other_cases(tmp_path, monkeypatch):
    programs = iter((WORKER, "import time; time.sleep(20)", "import os; os._exit(42)", WORKER))

    def fixture_command(command, **kwargs):
        program = next(programs)
        return supervise([sys.executable, "-c", program, *command[-2:]], **kwargs)

    monkeypatch.setattr("mlx_spec_bench.runner.supervise", fixture_command)
    base = Config(ModelSpec("fixture"), timeout_seconds=5, quality=False)
    case = replace(base, max_tokens=3, timeout_seconds=0.2)
    sweep = Sweep((base, case, replace(base, max_tokens=7), replace(base, max_tokens=4)))
    started = monotonic()
    result = run_sweep(sweep, output_dir=tmp_path)
    assert monotonic() - started < 8
    assert result.status == "failure"
    rows = read_records(result.directory / "results.jsonl")
    assert read_manifest(result.directory / "manifest.json", rows)["status"] == "failure"
    assert [row["error"]["kind"] for row in rows] == [
        "allocation_failure", "timeout", "worker_exit", "allocation_failure",
    ]
    assert "42" in rows[2]["error"]["message"]
    assert "deadline" in rows[1]["error"]["message"]
    assert [row["run_order"] for row in rows] == [1, 2, 3, 4]
    assert rows[1]["wall_time_ns"] < 1_500_000_000
    assert len(list((result.directory / "worker-logs").glob("*.log"))) == 4


def test_interruption_keeps_completed_rows_and_stops_pending_cases(tmp_path, monkeypatch):
    programs = iter((WORKER, "import os,signal,time; os.kill(os.getppid(),signal.SIGINT); time.sleep(20)"))

    def fixture_command(command, **kwargs):
        return supervise([sys.executable, "-c", next(programs), *command[-2:]], **kwargs)

    monkeypatch.setattr("mlx_spec_bench.runner.supervise", fixture_command)
    base = Config(ModelSpec("fixture"), quality=False)
    result = run_sweep(Sweep((base,), repetitions=3), output_dir=tmp_path)
    rows = read_records(result.directory / "results.jsonl")
    manifest = read_manifest(result.directory / "manifest.json", rows)
    assert manifest["status"] == "interrupted"
    assert len(manifest["expected"]) == 3
    assert [row["error"]["kind"] for row in rows] == ["allocation_failure", "interrupted"]


def test_no_previous_evidence_is_overwritten(tmp_path, monkeypatch):
    def fixture_command(command, **kwargs):
        return supervise([sys.executable, "-c", WORKER, *command[-2:]], **kwargs)

    monkeypatch.setattr("mlx_spec_bench.runner.supervise", fixture_command)
    sweep = Sweep((Config(ModelSpec("fixture"), quality=False),))
    first = run_sweep(sweep, output_dir=tmp_path)
    original = (first.directory / "results.jsonl").read_bytes()
    second = run_sweep(sweep, output_dir=tmp_path)
    assert first.directory != second.directory
    assert (first.directory / "results.jsonl").read_bytes() == original
