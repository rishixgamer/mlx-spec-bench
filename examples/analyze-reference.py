"""Summarize matched Qwen2.5 reference trials without exporting host identifiers."""

import argparse
import json
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from statistics import median

from mlx_spec_bench.schema import read_manifest, read_records


def read_requests(root, row):
    detail = json.loads((root / row["details"]).read_text(encoding="utf-8"))
    if detail["run_id"] != row["run_id"]:
        raise ValueError(f"trace belongs to another trial: {row['details']}")
    return {request["prompt_id"]: request for request in detail["requests"]}


def control_key(row):
    config = {**row["config"], "k": 0, "draft": None}
    return (json.dumps(config, sort_keys=True), row["repetition"])


def per_request_rate(request):
    emissions = request["emissions"]
    if not emissions:
        return None
    span = emissions[-1]["at_ns"] - emissions[0]["at_ns"]
    return (len(request["output_ids"]) - 1) * 1e9 / span if span > 0 else None


def compare(root, baseline, trial):
    for identity in ("target", "suite", "evaluation"):
        if baseline["identities"][identity] != trial["identities"][identity]:
            raise ValueError(f"matched trials disagree on {identity}")
    if (
        baseline["environment"] != trial["environment"]
        or baseline["warmups"] != trial["warmups"]
    ):
        raise ValueError("matched trials disagree on environment or warm-ups")
    control = read_requests(root, baseline)
    speculative = read_requests(root, trial)
    if control.keys() != speculative.keys():
        raise ValueError("matched trials disagree on prompt IDs")
    prompts = []
    for prompt, request in speculative.items():
        reference = control[prompt]
        if request["input_ids"] != reference["input_ids"]:
            raise ValueError(f"matched trials disagree on input tokens for {prompt}")
        first_difference = next(
            (
                {"index": index, "target_only_id": expected, "speculative_id": observed}
                for index, (expected, observed) in enumerate(
                    zip_longest(reference["output_ids"], request["output_ids"])
                )
                if expected != observed
            ),
            None,
        )
        control_rate = per_request_rate(reference)
        trial_rate = per_request_rate(request)
        prompts.append(
            {
                "prompt": prompt,
                "input_tokens": len(request["input_ids"]),
                "output_tokens": len(request["output_ids"]),
                "target_only_output_tokens": len(reference["output_ids"]),
                "first_difference": first_difference,
                "stop_reason_matches": request["stop_reason"]
                == reference["stop_reason"],
                "decode_ratio": trial_rate / control_rate
                if control_rate and trial_rate
                else None,
            }
        )
    control_rate = baseline["metrics"]["decode_tokens_per_second"]
    trial_rate = trial["metrics"]["decode_tokens_per_second"]
    return {
        "repetition": trial["repetition"],
        "draft_acceptance": trial["metrics"]["acceptance"],
        "decode_ratio": trial_rate / control_rate
        if control_rate and trial_rate
        else None,
        "target_only_decode_tps": control_rate,
        "speculative_decode_tps": trial_rate,
        "target_only_peak_footprint_bytes": baseline["process_memory"][
            "peak_process_footprint_bytes"
        ],
        "speculative_peak_footprint_bytes": trial["process_memory"][
            "peak_process_footprint_bytes"
        ],
        "target_window_matches": (
            baseline["metrics"]["kv"]["window_tokens"]
            == trial["metrics"]["kv"]["window_tokens"]
        ),
        "prompts": prompts,
    }


def summarize(root):
    rows = read_records(root / "results.jsonl")
    source = read_manifest(root / "manifest.json", rows)
    controls = {
        control_key(row): row
        for row in rows
        if row["status"] == "success" and row["config"]["k"] == 0
    }
    groups = defaultdict(list)
    unmatched_trials = failed_trials = 0
    for row in rows:
        if row["status"] != "success":
            failed_trials += 1
            continue
        config = row["config"]
        if not config["k"]:
            continue
        control = controls.get(control_key(row))
        if control is None:
            unmatched_trials += 1
            continue
        group = (
            config["target"]["weight_format"],
            config["cache"]["policy"],
            config["batch_size"],
            config["k"],
        )
        groups[group].append(compare(root, control, row))
    return {
        "sweep_status": source["status"],
        "recorded_trials": len(rows),
        "expected_trials": len(source["expected"]),
        "failed_trials": failed_trials,
        "unmatched_speculative_trials": unmatched_trials,
        "groups": [
            {
                "format": format_,
                "cache": cache,
                "batch": batch,
                "k": k,
                "median_decode_ratio": median(
                    row["decode_ratio"]
                    for row in trials
                    if row["decode_ratio"] is not None
                )
                if any(row["decode_ratio"] is not None for row in trials)
                else None,
                "trials": trials,
            }
            for (format_, cache, batch, k), trials in sorted(groups.items())
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.sweep), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
