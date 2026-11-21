"""Offline comparison tables over the raw v1 evidence."""

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path

from .io import fingerprint
from .schema import read_manifest, read_records

GB = 1_000_000_000


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _format(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _escape(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def _workload(row):
    suite = row["identities"]["suite"]
    return (suite["id"], suite["sha256"]) if suite else (row["config"]["suite"], None)


def _hardware(row):
    return fingerprint(row["environment"])


def _latency_key(row):
    cfg = row["config"]
    return (
        _hardware(row), _workload(row), cfg["batch_size"], cfg["max_tokens"], cfg["temperature"], cfg["seed"],
        fingerprint(row["identities"]["target"]), fingerprint(cfg["cache"]),
        row["metrics"]["kv"]["window_tokens"], row["warmups"],
    )


def _quality_key(row):
    cfg = row["config"]
    target = row["identities"]["target"]
    return (
        _hardware(row), _workload(row), cfg["batch_size"], cfg["max_tokens"], cfg["temperature"], cfg["seed"],
        target["id"], target["source"], target["revision"], target["tokenizer_sha256"],
        fingerprint(row["identities"]["evaluation"]),
    )


@dataclass
class Group:
    row: dict
    records: list[dict]
    attempted: int
    mixed: bool = False

    @property
    def metrics(self) -> list[dict]:
        return [record["metrics"] for record in self.records if record["status"] == "success"]

    @property
    def successful(self) -> bool:
        return self.row["status"] == "success" and not self.mixed

    def aggregate(self) -> dict:
        metrics = self.metrics
        observations = [item["observations"] for item in metrics]
        quality = [item["quality"] for item in metrics if item["quality"] is not None]
        scored = sum(item["scored_tokens"] for item in quality)
        memory = [row["process_memory"] for row in self.records]
        footprint = [
            item["peak_process_footprint_bytes"] for item in memory
            if item["peak_process_footprint_bytes"] is not None
        ]
        rss = [item["peak_process_rss_bytes"] for item in memory if item["peak_process_rss_bytes"] is not None]
        source, peak = ("footprint", max(footprint)) if len(footprint) == len(memory) else (
            ("RSS", max(rss)) if rss else (None, None)
        )
        return {
            "ttft": _ratio(sum(x["ttft_total_ns"] for x in observations) / 1e6, sum(x["ttft"] for x in observations)),
            "itl": _ratio(sum(x["itl_total_ns"] for x in observations) / 1e6, sum(x["itl"] for x in observations)),
            "throughput": _ratio(
                sum(x["decode_tokens"] for x in observations) * 1e9,
                sum(x["decode_duration_ns"] for x in observations),
            ),
            "acceptance": _ratio(
                sum(x["tokens"]["accepted"] for x in metrics), sum(x["tokens"]["proposed"] for x in metrics),
            ),
            "peak": peak,
            "mlx_peak": max((item["mlx_allocator_peak_bytes"] for item in metrics), default=None),
            "memory_source": source,
            "ppl": math.exp(math.fsum(x["total_nll"] for x in quality) / scored) if scored else None,
            "task": _ratio(sum(x["task_correct"] for x in quality), sum(x["task_examples"] for x in quality)),
        }


def _groups(records):
    cases = defaultdict(list)
    for row in records:
        cases[(row["sweep_id"], row["case_id"])].append(row)
    for rows in cases.values():
        compatible = defaultdict(list)
        failed = []
        for row in rows:
            if row["status"] == "success":
                key = fingerprint({
                    "environment": row["environment"], "identities": row["identities"], "warmups": row["warmups"],
                })
                compatible[key].append(row)
            else:
                failed.append(row)
        if len(compatible) == 1:
            successful = next(iter(compatible.values()))
            yield Group(successful[0], successful + failed, len(rows))
        elif compatible:
            for successful in compatible.values():
                yield Group(successful[0], successful, len(rows), mixed=True)
            if failed:
                yield Group(failed[0], failed, len(rows), mixed=True)
        else:
            yield Group(failed[0], failed, len(rows))


def render_report(records: list[dict], manifest: dict | None = None) -> str:
    lines = ["# MLX benchmark comparison", ""]
    if manifest:
        missing = {entry["run_id"] for entry in manifest["expected"]} - {row["run_id"] for row in records}
        lines += [
            f"Sweep `{manifest['sweep_id']}` · {manifest['status']} · "
            f"{len(records)}/{len(manifest['expected'])} declared repetitions recorded.", "",
        ]
        if missing:
            lines += ["**Missing run IDs** " + ", ".join(f"`{key}`" for key in sorted(missing)), ""]

    groups = list(_groups(records))
    by_workload = defaultdict(list)
    latency_controls = defaultdict(list)
    quality_controls = defaultdict(list)
    aggregate = {}
    for group in groups:
        row = group.row
        by_workload[_workload(row)].append(group)
        aggregate[id(group)] = group.aggregate()
        cfg = row["config"]
        if group.successful and cfg["k"] == 0:
            latency_controls[_latency_key(row)].append(group)
            if cfg["target"]["weight_format"] == "fp16" and cfg["cache"]["policy"] == "unbounded":
                quality_controls[_quality_key(row)].append(group)

    if not groups:
        lines += ["No completed repetitions.", ""]
    for (name, sha), workload in by_workload.items():
        lines += [
            f"## {_escape(name)}" + (f" · `{sha[:12]}`" if sha else " · fingerprint unavailable"), "",
            "| Case | Target | Draft | Batch | K | KV bytes | TTFT ms | ITL ms/token | Decode tok/s | "
            "Acceptance | Peak GB | MLX GB | PPL | Task | OK/attempted | Speedup | Δ PPL | Δ task pp |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        evidence = []
        for group in workload:
            row = group.row
            cfg = row["config"]
            result = aggregate[id(group)]
            case_id = row["case_id"]
            run_ids = ", ".join(f"`{x['run_id']}`" for x in sorted(group.records, key=lambda x: x["run_order"]))
            controls = []
            speedup = delta_ppl = delta_task = None
            if group.successful:
                latency = latency_controls[_latency_key(row)]
                if len(latency) == 1:
                    baseline = latency[0]
                    base_value = aggregate[id(baseline)]["throughput"]
                    if result["throughput"] is not None:
                        speedup = _ratio(result["throughput"], base_value)
                    controls.append(f"speed control `{baseline.row['case_id']}`")
                    if speedup is None:
                        controls.append("decode speedup undefined")
                else:
                    controls.append("missing/incomparable target-only control" if not latency else "ambiguous target-only controls")
                if cfg["quality"]:
                    quality = quality_controls[_quality_key(row)]
                    if len(quality) == 1:
                        baseline = quality[0]
                        base_result = aggregate[id(baseline)]
                        if result["ppl"] is not None and base_result["ppl"] is not None:
                            delta_ppl = result["ppl"] - base_result["ppl"]
                        if result["task"] is not None and base_result["task"] is not None:
                            delta_task = (result["task"] - base_result["task"]) * 100
                        controls.append(f"quality control `{baseline.row['case_id']}` (FP16/unbounded)")
                    else:
                        controls.append("missing/incomparable FP16 quality control" if not quality else "ambiguous FP16 quality controls")
            elif group.mixed:
                controls.append("mixed artifact, hardware, or warm-up settings across repetitions")
            if result["acceptance"] is None and group.metrics:
                controls.append("acceptance undefined (zero proposals)")
            failures = [f"{x['error']['kind']} ({_escape(x['error']['message'])})" for x in group.records if x["error"]]
            peak_value = "—" if result["peak"] is None else f"{result['peak'] / GB:.3f} {result['memory_source']}"
            draft = cfg["draft"]
            table = [
                f"`{case_id}`", cfg["target"]["id"] + "/" + cfg["target"]["weight_format"],
                draft["id"] + "/" + draft["weight_format"] if draft else "—",
                cfg["batch_size"], cfg["k"], cfg["cache"]["budget_bytes"] or "unbounded",
                _format(result["ttft"]), _format(result["itl"]), _format(result["throughput"]),
                _format(result["acceptance"]), peak_value,
                _format(result["mlx_peak"] / GB if result["mlx_peak"] is not None else None),
                _format(result["ppl"]), _format(result["task"]), f"{len(group.metrics)}/{group.attempted}",
                _format(speedup), _format(delta_ppl), _format(delta_task),
            ]
            lines.append("| " + " | ".join(_escape(value) for value in table) + " |")
            notes = controls + failures
            evidence.append(f"- `{case_id}` · {run_ids}" + (" · " + " · ".join(notes) if notes else ""))
        lines += ["", "### Run evidence", "", *evidence, ""]
    lines += [
        "TTFT and ITL pool their observation counts. Decode tok/s divides pooled post-first tokens "
        "by the sum of measured decode spans. Acceptance and task score pool their counts. "
        "PPL exponentiates pooled NLL per scored token. Peaks are maxima. GB is 1,000,000,000 bytes. "
        "Speedup uses decode throughput, and quality deltas are measured minus control.", "",
    ]
    return "\n".join(lines)


def report(path: str | Path) -> str:
    path = Path(path)
    if path.is_dir():
        rows = read_records(path / "results.jsonl")
        return render_report(rows, read_manifest(path / "manifest.json", rows))
    return render_report(read_records(path))
