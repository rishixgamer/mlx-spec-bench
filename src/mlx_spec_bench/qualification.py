"""Reference-matrix evidence checks and a checksummed archive."""

from importlib.resources import files
import json
import math
import os
from pathlib import Path
import re
import tarfile
from tempfile import TemporaryDirectory

from .config import Sweep
from .decode import CacheStats, Generation, Trace
from .io import file_fingerprint, read_json, write_json
from .metrics import generation_metrics
from .quality import task_matches
from .report import render_report
from .schema import read_manifest, read_records

REFERENCE_MEMORY = 48 * 1024**3
FOOTPRINT_GATE = 40_000_000_000
SUSTAINED_PRESSURE_SECONDS = 1
# Both pinned base Qwen2.5 revisions use this single EOS ID.
REFERENCE_EOS_ID = 151643


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


def _nat(value) -> bool:
    return type(value) is int and value >= 0


def _ids(values) -> bool:
    return isinstance(values, list) and all(_nat(token) for token in values)


def _valid_stop(output, stop, cap):
    return (
        _ids(output) and 0 < len(output) <= cap and REFERENCE_EOS_ID not in output[:-1]
        and (stop == "eos" if output[-1] == REFERENCE_EOS_ID else stop == "max_tokens" and len(output) == cap)
    )


def _check_quality(detail, evaluation, metrics, cap):
    text, tasks = detail["text"], detail["tasks"]
    if (
        [p["id"] for p in text] != [p["id"] for p in evaluation["text"]]
        or [t["id"] for t in tasks] != [t["id"] for t in evaluation["tasks"]]
    ):
        raise ValueError("evaluation evidence does not cover the pinned split")
    for passage in text:
        ids, mask = passage["input_ids"], passage["scored_mask"]
        count, loss = passage["scored_tokens"], passage["total_nll"]
        if (
            not _ids(ids) or len(mask) != len(ids) or any(type(flag) is not bool for flag in mask)
            or (mask and mask[0]) or not _nat(count) or count != sum(mask)
            or type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0 or (not count and loss != 0)
        ):
            raise ValueError("invalid perplexity evidence")
    for task, expected in zip(tasks, evaluation["tasks"], strict=True):
        if (
            not _ids(task["input_ids"]) or not task["input_ids"]
            or not _valid_stop(task["output_ids"], task["stop_reason"], cap)
            or task["expected"] != expected["answer"]
            or task["correct"] is not task_matches(task["actual"], expected["answer"])
        ):
            raise ValueError("invalid task-score evidence")
    if (
        sum(p["scored_tokens"] for p in text) != metrics["scored_tokens"]
        or math.fsum(p["total_nll"] for p in text) != metrics["total_nll"]
        or sum(t["correct"] for t in tasks) != metrics["task_correct"]
        or len(tasks) != metrics["task_examples"]
    ):
        raise ValueError("evaluation totals disagree with the record")


def _check_requests(detail, row, prompt_ids):
    cfg, metrics = row["config"], row["metrics"]
    target_window = metrics["kv"]["window_tokens"]
    plan = detail["cache_plan"]
    if (
        plan["target_window"] != target_window or plan["budget_bytes"] != cfg["cache"]["budget_bytes"]
        or not _nat(plan["prefill_chunk"]) or plan["prefill_chunk"] == 0
        or (not cfg["k"] and plan["draft_window"] is not None)
        or (cfg["cache"]["policy"] == "sliding" and (
            not _nat(target_window) or target_window == 0
            or cfg["k"] and (not _nat(plan["draft_window"]) or plan["draft_window"] == 0)
        ))
        or (cfg["cache"]["policy"] == "unbounded" and (target_window is not None or plan["draft_window"] is not None))
    ):
        raise ValueError("cache plan disagrees with the record")
    traces = [Trace(**trace) for trace in detail["requests"]]
    if [trace.prompt_id for trace in traces] != prompt_ids:
        raise ValueError("trace does not cover the pinned prompt suite")
    evictions = 0
    for trace in traces:
        if (
            not _ids(trace.input_ids) or not trace.input_ids
            or not _valid_stop(trace.output_ids, trace.stop_reason, cfg["max_tokens"])
            or not _nat(trace.submit_ns) or not _nat(trace.proposed)
        ):
            raise ValueError(f"{trace.prompt_id}: invalid token IDs, stop reason, or submission time")
        previous_time = trace.submit_ns
        for event in trace.emissions:
            at, count, kind = event["at_ns"], event["count"], event["kind"]
            if (
                not _nat(at) or not previous_time <= at <= metrics["generation_ns"]
                or not _nat(count) or not 0 < count <= len(trace.output_ids)
                or kind not in ("autoregressive", "accepted", "corrections", "bonus")
            ):
                raise ValueError(f"{trace.prompt_id}: invalid emission event")
            previous_time = at
        if sum(event["count"] for event in trace.emissions) != len(trace.output_ids):
            raise ValueError(f"{trace.prompt_id}: emission count disagrees with output IDs")
        offsets = {"target": 0, **({"draft": 0} if cfg["k"] else {})}
        for event in trace.cache_events:
            role, start, end = event["role"], event["start"], event["end"]
            size, staged = event["bytes"], event["staged_bytes"]
            window = plan[f"{role}_window"]
            if (
                role not in offsets or not all(_nat(v) for v in (start, end, size, staged, event["evictions"]))
                or not offsets[role] <= end <= len(trace.input_ids) + len(trace.output_ids)
                or end <= start or start != (max(0, end - window) if window is not None else 0)
                or event["evictions"] != start or not 0 < size <= staged
                or staged > metrics["kv"]["working_peak_bytes"]
            ):
                raise ValueError(f"{trace.prompt_id}: invalid KV cache event")
            evictions += start - (max(0, offsets[role] - window) if window is not None else 0)
            offsets[role] = end
        if any(offset < len(trace.input_ids) for offset in offsets.values()):
            raise ValueError(f"{trace.prompt_id}: missing prefill cache evidence")
    if evictions != metrics["kv"]["evictions"]:
        raise ValueError("cache eviction total disagrees with the record")
    measured = generation_metrics(
        Generation(traces, metrics["prefill_ns"], metrics["generation_ns"], CacheStats()),
        resident=metrics["mlx_resident_start_bytes"], peak=metrics["mlx_allocator_peak_bytes"],
        quality=metrics["quality"], window=target_window,
    )
    if any(measured[key] != metrics[key] for key in ("tokens", "observations")):
        raise ValueError("emission timing or token totals disagree with the record")


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
    evaluation_suite = load_evaluation()
    evaluation = evaluation_identity(evaluation_suite, 32)
    known_artifacts = {}
    tokenizers = set()
    environments = set()
    for row in rows:
        run_id = row["run_id"]
        if row["status"] != "success":
            issues.append(f"{run_id}: {row['error']['kind']} ({row['error']['message']})")
            continue
        if not (root / "worker-logs" / f"{run_id}.log").is_file():
            issues.append(f"{run_id}: missing worker log")
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
            if (
                type(detail["schema_version"]) is not int or detail["schema_version"] != 1
                or any(detail[key] != row[key] for key in ("run_id", "case_id", "sweep_id", "repetition", "run_order"))
                or detail["suite"] != suite or detail["evaluation"] != evaluation
            ):
                raise ValueError("trace identity disagrees with the record")
            _check_requests(detail, row, prompt_ids)
            _check_quality(detail["quality"], evaluation_suite.data, metrics["quality"], cfg["max_tokens"])
            tokens = _trace_key(detail)
        except (ValueError, TypeError, OSError, KeyError, OverflowError) as error:
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
            *[(f"worker-logs/{row['run_id']}.log", root / "worker-logs" / f"{row['run_id']}.log") for row in rows],
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
