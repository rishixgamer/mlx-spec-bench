"""Reference-matrix evidence checks and a checksummed archive."""

from importlib.resources import files
import json
import os
from pathlib import Path
import re
import tarfile
from tempfile import TemporaryDirectory

from .config import Sweep
from .io import file_fingerprint, read_json, write_json
from .report import render_report
from .schema import read_manifest, read_records

REFERENCE_MEMORY = 48 * 1024**3
FOOTPRINT_GATE = 40_000_000_000
SUSTAINED_PRESSURE_SECONDS = 1


def reference_sweep() -> Sweep:
    resource = files("mlx_spec_bench").joinpath("data", "reference-sweep.json")
    return Sweep.from_dict(json.loads(resource.read_text(encoding="utf-8")))


def _trace_key(detail: dict) -> list[tuple]:
    return [
        (request["prompt_id"], request["input_ids"], request["output_ids"], request["stop_reason"])
        for request in detail["requests"]
    ]


def _control_key(row: dict) -> tuple:
    cfg = row["config"]
    return (
        cfg["target"]["weight_format"], cfg["batch_size"], cfg["cache"]["budget_bytes"],
    )


def _case_problems(root: Path, rows: list[dict]) -> list[str]:
    from .models import SOURCES
    from .quality import evaluation_identity
    from .suites import load_evaluation, load_suite

    issues = []
    controls = {}
    checked = []
    workload = load_suite("reference-v1")
    suite = workload.identity
    prompt_ids = [prompt["id"] for prompt in workload.data["prompts"]]
    evaluation = evaluation_identity(load_evaluation(), 32)
    known_artifacts = {}
    tokenizers = set()
    environments = set()
    for row in rows:
        run_id = row["run_id"]
        if row["status"] != "success":
            issues.append(f"{run_id}: {row['error']['kind']} ({row['error']['message']})")
            continue
        env = row["environment"]
        environments.add(json.dumps(env, sort_keys=True))
        if (
            (env["platform"], env["architecture"], env["memory_bytes"]) != ("Darwin", "arm64", REFERENCE_MEMORY)
            or not env["backend"].startswith("Device(gpu")
        ):
            issues.append(f"{run_id}: expected Apple Silicon Metal on a 48 GB Mac")
        memory = row["process_memory"]
        if (
            not memory["footprint_samples"] or memory["peak_process_footprint_bytes"] is None
            or memory["peak_process_footprint_bytes"] > FOOTPRINT_GATE
        ):
            issues.append(f"{run_id}: missing Apple footprint or exceeds {FOOTPRINT_GATE} bytes")
        pressure = memory["pressure_samples"]
        if (
            pressure["normal"] == 0
            or memory["pressure_max_elevated_span_ns"] >= SUSTAINED_PRESSURE_SECONDS * 1_000_000_000
            or memory["system_swap_start_bytes"] is None or memory["system_swap_end_bytes"] is None
        ):
            issues.append(f"{run_id}: pressure or swap evidence missing, or sustained pressure observed")
        identities = row["identities"]
        if identities["suite"] != suite or identities["evaluation"] != evaluation:
            issues.append(f"{run_id}: pinned workload/evaluation fingerprint mismatch")
        for role in ("target", "draft"):
            if model := identities[role]:
                source = SOURCES[model["id"]]
                key = model["id"], model["weight_format"]
                if model["source"] != source.repo or model["revision"] != source.revision:
                    issues.append(f"{run_id}: {role} source revision mismatch")
                previous = known_artifacts.setdefault(key, model["artifact_sha256"])
                if previous != model["artifact_sha256"]:
                    issues.append(f"{run_id}: multiple verified artifacts for {key}")
                tokenizers.add(model["tokenizer_sha256"])
        cfg = row["config"]
        metrics = row["metrics"]
        if cfg["cache"]["policy"] == "sliding" and metrics["kv"]["evictions"] == 0:
            issues.append(f"{run_id}: bounded trial did not evict any cache positions")
        if cfg["k"] and not metrics["tokens"]["proposed"]:
            issues.append(f"{run_id}: speculative trial never proposed a token")
        try:
            detail = read_json(root / row["details"])
            tokens = _trace_key(detail)
            if (
                detail["run_id"] != run_id or detail["case_id"] != row["case_id"]
                or detail["suite"] != suite or detail["evaluation"] != evaluation
                or [item[0] for item in tokens] != prompt_ids
                or any(not item[1] or not item[2] or item[3] not in ("eos", "max_tokens") for item in tokens)
                or metrics["tokens"]["input"] != sum(len(item[1]) for item in tokens)
                or metrics["tokens"]["output"] != sum(len(item[2]) for item in tokens)
            ):
                raise ValueError("trace identity, counts, or stop reasons disagree with the record")
        except (ValueError, TypeError, OSError, KeyError) as error:
            issues.append(f"{run_id}: invalid detailed evidence ({error})")
            continue
        checked.append((row, tokens))
        if cfg["k"] == 0:
            controls.setdefault(_control_key(row), (row, tokens))
    if len(tokenizers) > 1:
        issues.append("Trials did not use one verified vocabulary and tokenizer.")
    if len(environments) > 1:
        issues.append("Trials span different hardware or software environments.")
    for row, tokens in checked:
        baseline = controls.get(_control_key(row))
        if baseline is None:
            issues.append(f"{row['run_id']}: missing matched target-only control")
        elif (
            tokens != baseline[1]
            or row["metrics"]["kv"]["window_tokens"] != baseline[0]["metrics"]["kv"]["window_tokens"]
        ):
            issues.append(f"{row['run_id']}: greedy tokens, input IDs, stopping, or target window differ from "
                          f"{baseline[0]['run_id']}")
    return issues


def qualify(artifact: str | Path, output: str | Path) -> list[str]:
    root = Path(artifact).resolve()
    output = Path(output).resolve()
    if not output.name.endswith(".tar.gz") or output.exists() or root in output.parents:
        raise ValueError("qualification output must be a new .tar.gz path outside the sweep directory")
    rows = read_records(root / "results.jsonl")
    manifest = read_manifest(root / "manifest.json", rows)
    reference = reference_sweep()
    expected_cases = {case.case_id: case.to_dict() for case in reference.cases}
    issues = []
    if (
        manifest["cases"] != expected_cases or manifest["warmups"] != reference.warmups
        or manifest["repetitions"] != reference.repetitions
    ):
        issues.append("Manifest differs from the declared v1 48 GB reference matrix.")
    if manifest["status"] != "success":
        issues.append(f"Sweep status is {manifest['status']}.")
    missing = {entry["run_id"] for entry in manifest["expected"]} - {row["run_id"] for row in rows}
    if missing:
        issues.append(f"Missing {len(missing)} declared repetition(s): " + ", ".join(sorted(missing)))
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["source_commit"] or ""):
        issues.append("Manifest must record the 40-character release candidate commit.")
    if not (manifest["host_notes"] or "").strip():
        issues.append("Manifest must describe background load on the reference host.")
    issues.extend(_case_problems(root, rows))
    if issues:
        return issues
    with TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary)
        report_path = staging / "report.md"
        report_path.write_text(render_report(rows, manifest), encoding="utf-8")
        members = [
            ("manifest.json", root / "manifest.json"), ("results.jsonl", root / "results.jsonl"),
            ("report.md", report_path),
            *[(row["details"], root / row["details"]) for row in rows],
            *[(path.relative_to(root).as_posix(), path) for path in sorted((root / "worker-logs").glob("*.log"))],
        ]
        checksums = staging / "SHA256SUMS"
        checksums.write_text("".join(f"{file_fingerprint(path)}  {name}\n" for name, path in members), encoding="utf-8")
        gate = staging / "qualification.json"
        write_json(gate, {
            "schema_version": 1, "sweep_id": manifest["sweep_id"], "source_commit": manifest["source_commit"],
            "cases": len(expected_cases), "repetitions": len(rows), "peak_gate_bytes": FOOTPRINT_GATE,
            "observed_peak_footprint_bytes": max(row["process_memory"]["peak_process_footprint_bytes"] for row in rows),
            "sustained_pressure_seconds": SUSTAINED_PRESSURE_SECONDS,
        })
        with checksums.open("a", encoding="utf-8") as stream:
            stream.write(f"{file_fingerprint(gate)}  qualification.json\n")
        archive_path = staging / "reference.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            for name, path in [*members, ("SHA256SUMS", checksums), ("qualification.json", gate)]:
                archive.add(path, arcname=name, recursive=False)
        # Exclusive publication keeps an existing qualification intact.
        os.link(archive_path, output)
    return []
