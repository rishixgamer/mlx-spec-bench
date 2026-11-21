"""Timed offline smoke gate, invoked by the Apple Silicon CI job."""

from time import perf_counter

STARTED = perf_counter()

import argparse
from importlib.resources import files
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from tempfile import TemporaryDirectory

from mlx_spec_bench.io import read_json, write_json
from mlx_spec_bench.schema import read_manifest, read_records

LIMIT = 45


def command(*args, env, cwd):
    remaining = LIMIT - (perf_counter() - STARTED)
    return subprocess.run(
        ["mlx-bench", *map(str, args)], check=True, text=True, capture_output=True,
        env=env, cwd=cwd, timeout=max(0.001, remaining),
    ).stdout


def check_runs(directory: Path) -> list[dict]:
    rows = read_records(directory / "results.jsonl")
    manifest = read_manifest(directory / "manifest.json", rows)
    assert len(rows) == len(manifest["expected"]) == 2
    assert manifest["status"] == "success"
    assert {x["config"]["k"] for x in rows} == {0, 7}
    baseline = json.loads(files("mlx_spec_bench").joinpath("data/fixture-baseline-v1.json").read_text())
    outputs = []
    for row in rows:
        assert row["status"] == "success" and row["environment"]["backend"].startswith("Device(gpu")
        assert row["environment"]["platform"] == "Darwin" and row["environment"]["architecture"] == "arm64"
        assert row["environment"]["packages"]["mlx"] == "0.31.2"
        assert row["environment"]["packages"]["mlx-lm"] == "0.31.3"
        assert row["identities"]["target"]["artifact_sha256"]
        metrics = row["metrics"]
        assert metrics["tokens"]["input"] == baseline["input_tokens"]
        assert metrics["tokens"]["output"] == baseline["output_tokens"]
        assert metrics["mlx_allocator_peak_bytes"] >= metrics["mlx_resident_start_bytes"] > 0
        assert row["process_memory"]["peak_process_footprint_bytes"] > 0
        assert metrics["observations"]["ttft"] == 5
        assert metrics["kv"]["evictions"] > 0
        data = read_json(directory / row["details"])
        assert data["run_id"] == row["run_id"] and data["suite"] == row["identities"]["suite"]
        actual = [{
            "prompt_id": item["prompt_id"], "input_tokens": len(item["input_ids"]),
            "output_ids": item["output_ids"], "stop_reason": item["stop_reason"],
        } for item in data["requests"]]
        assert actual == baseline["requests"]
        outputs.append(actual)
        first_timings_ns = [trace["emissions"][0]["at_ns"] - trace["submit_ns"] for trace in data["requests"]]
        assert metrics["ttft_ms"] == sum(first_timings_ns) / 1e6 / len(first_timings_ns)
        if row["config"]["k"]:
            assert metrics["tokens"]["accepted"] > 0 and metrics["acceptance"] == 1
    assert outputs[0] == outputs[1]
    return rows


def check_rejects_corruption(directory: Path, rows: list[dict]) -> None:
    corrupt = directory / rows[-1]["details"]
    original = corrupt.read_text()
    value = json.loads(original)
    value["requests"][0]["output_ids"][0] += 1
    write_json(corrupt, value)
    try:
        try:
            check_runs(directory)
        except AssertionError:
            pass
        else:
            raise AssertionError("example with changed token passed the output assertions")
    finally:
        corrupt.write_text(original)

    invalid = directory / "invalid.jsonl"
    value = json.loads((directory / "results.jsonl").read_text().splitlines()[0])
    del value["environment"]["packages"]["mlx"]
    write_json(invalid, value)
    try:
        try:
            read_records(invalid)
        except ValueError as error:
            assert "'mlx' is a required property" in str(error)
        else:
            raise AssertionError("example with missing MLX metadata passed the schema")
    finally:
        invalid.unlink()


def smoke(output: Path) -> int:
    assert (platform.system(), platform.machine()) == ("Darwin", "arm64"), "Apple Silicon Metal required"
    assert shutil.which("mlx-bench"), "install the mlx-bench command first"
    with TemporaryDirectory() as temporary:
        work = Path(temporary)
        environment = {
            **os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": str(work / "hf-cache"), "MLX_SPEC_BENCH_CACHE": str(work / "fixture-cache"),
        }
        config = work / "smoke.yaml"
        config.write_text("""schema_version: 1
base:
  target: {id: fixture}
  draft: {id: fixture}
  batch_size: 2
  cache: {policy: sliding, budget_bytes: 45056}
  timeout_seconds: 30
matrix:
  k: [0, 7]
""")
        directory = Path(command("run", config, "--output-dir", output, env=environment, cwd=work).strip())
        rows = check_runs(directory)
        command("report", directory, "--output", output / "report.md", env=environment, cwd=work)
        report = (output / "report.md").read_text()
        assert "TTFT ms" in report and "ITL ms/token" in report and "fixture-v1" in report
        assert all(row["run_id"] in report for row in rows)
        check_rejects_corruption(directory, rows)
    return 4  # Public run/greedy, public report, token corruption, schema corruption.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="directory for smoke evidence")
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        count = smoke(output)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"{error.cmd} failed with {error.returncode}: {error.stderr}") from error
    finally:
        elapsed = perf_counter() - STARTED
        summary = {
            "elapsed_seconds": elapsed, "limit_seconds": LIMIT, "test_count": count,
            "runner_name": os.environ.get("RUNNER_NAME", platform.node()),
            "platform": platform.platform(), "architecture": platform.machine(),
            "commit": os.environ.get("GITHUB_SHA"),
        }
        write_json(output / "smoke-gate.json", summary)
        print(json.dumps(summary), flush=True)
    assert elapsed < LIMIT, f"smoke took {elapsed:.3f}s, limit {LIMIT}s"


if __name__ == "__main__":
    main()
