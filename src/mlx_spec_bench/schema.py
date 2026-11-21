"""Validate the portable v1 configuration and evidence contracts."""

from functools import lru_cache
from importlib.resources import files
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator, validators
from referencing import Registry
from referencing.jsonschema import DRAFT202012

from .io import reject_constant

StrictValidator = validators.extend(
    Draft202012Validator,
    type_checker=Draft202012Validator.TYPE_CHECKER.redefine(
        "integer", lambda checker, value: type(value) is int
    ),
)

SCHEMA_ROOT = "https://mlx-spec-bench.invalid/schemas/v1/"


@lru_cache
def _registry() -> Registry:
    return Registry().with_resources(
        (SCHEMA_ROOT + path.name, DRAFT202012.create_resource(json.loads(path.read_text())))
        for path in files("mlx_spec_bench").joinpath("schemas").iterdir()
        if path.name.endswith(".json")
    )


@lru_cache
def _validator(kind: str) -> Draft202012Validator:
    registry = _registry()
    schema = registry.get_or_retrieve(SCHEMA_ROOT + kind + ".json").value.contents
    return StrictValidator(schema, registry=registry)


def _check_numbers(value, location: str = "") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location or 'document'}: expected a finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _check_numbers(item, f"{location}.{key}".lstrip("."))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_numbers(item, f"{location}[{index}]")


def validate(kind: str, value) -> None:
    _check_numbers(value)
    error = next(_validator(kind).iter_errors(value), None)
    if error is not None:
        while error.context:
            narrowed = next((child for child in error.context if child.validator != "type"), None)
            if narrowed is None:
                break
            error = narrowed
        path = ".".join(map(str, error.absolute_path))
        raise ValueError(f"{path or kind}: {error.message}")


def validate_record(record: dict) -> None:
    from .config import Config

    validate("record", record)
    config = Config.from_dict(record["config"])
    if record["case_id"] != config.case_id or record["config"] != config.to_dict():
        raise ValueError("config: expected a fully resolved config matching case_id")
    expected_id = f"{record['sweep_id']}-{record['case_id']}-r{record['repetition']:03d}"
    if record["run_id"] != expected_id:
        raise ValueError("run_id: must match sweep_id, case_id and repetition")
    if record["status"] == "success":
        metrics = record["metrics"]
        reasons = metrics["undefined"]
        for key in ("ttft_ms", "itl_ms_per_token", "decode_tokens_per_second", "acceptance", "perplexity", "task_score"):
            if (metrics[key] is None) != (key in reasons):
                raise ValueError(f"metrics.{key}: null metrics require exactly one undefined reason")
        counts = metrics["tokens"]
        if counts["accepted"] > counts["proposed"] or counts["output"] != sum(
            counts[key] for key in ("accepted", "corrections", "bonus", "autoregressive")
        ):
            raise ValueError("metrics.tokens: inconsistent accepted, proposed or emitted counts")
        expected = counts["accepted"] / counts["proposed"] if counts["proposed"] else None
        if metrics["acceptance"] != expected:
            raise ValueError("metrics.acceptance: must equal accepted / proposed")
        if record["details"] != f"details/{expected_id}.json":
            raise ValueError("details: must name the current run's evidence")
        observations = metrics["observations"]
        for key, total, count in (
            ("ttft_ms", observations["ttft_total_ns"] / 1e6, observations["ttft"]),
            ("itl_ms_per_token", observations["itl_total_ns"] / 1e6, observations["itl"]),
            ("decode_tokens_per_second", observations["decode_tokens"] * 1e9, observations["decode_duration_ns"]),
        ):
            if metrics[key] != (total / count if count else None):
                raise ValueError(f"metrics.{key}: inconsistent observation totals")
        if (
            observations["ttft"] != observations["requests"]
            or observations["itl"] != observations["decode_tokens"]
            or counts["output"] != observations["decode_tokens"] + observations["requests"]
        ):
            raise ValueError("metrics.observations: inconsistent request and token counts")
        if metrics["mlx_allocator_peak_bytes"] < metrics["mlx_resident_start_bytes"]:
            raise ValueError("metrics.mlx_allocator_peak_bytes: must include the resident start")
        kv = metrics["kv"]
        if config.cache.budget_bytes is not None and (
            kv["window_tokens"] is None or kv["working_peak_bytes"] > config.cache.budget_bytes
        ):
            raise ValueError("metrics.kv: sliding policy is missing its window or exceeded budget")
        if config.cache.budget_bytes is None and (kv["window_tokens"] is not None or kv["evictions"]):
            raise ValueError("metrics.kv: unbounded policy cannot evict")
        quality = metrics["quality"]
        if bool(quality) != config.quality:
            raise ValueError("metrics.quality: must correspond to an enabled evaluation")
        if quality:
            examples, correct = quality["task_examples"], quality["task_correct"]
            scored = quality["scored_tokens"]
            if correct > examples or metrics["task_score"] != (correct / examples if examples else None):
                raise ValueError("metrics.task_score: inconsistent task counts")
            try:
                perplexity_ok = (
                    metrics["perplexity"] is not None
                    and math.isclose(metrics["perplexity"], math.exp(quality["total_nll"] / scored), rel_tol=1e-12)
                ) if scored else metrics["perplexity"] is None
            except OverflowError:
                perplexity_ok = False
            if not perplexity_ok:
                raise ValueError("metrics.perplexity: inconsistent corpus loss")
        elif metrics["perplexity"] is not None or metrics["task_score"] is not None:
            raise ValueError("metrics.quality: disabled evaluation cannot have scores")
        identities = record["identities"]
        if identities["target"] is None or identities["suite"] is None:
            raise ValueError("identities: successful runs require target and suite identities")
        for role in ("target", "draft"):
            spec = getattr(config, role)
            identity = identities[role]
            if spec is not None and identity is not None and (
                identity["id"] != spec.id or identity["weight_format"] != spec.weight_format
            ):
                raise ValueError(f"identities.{role}: model or weight format disagrees with config")
        if (identities["draft"] is not None) != bool(config.k):
            raise ValueError("identities.draft: must correspond to an active draft")
        if (identities["evaluation"] is not None) != config.quality:
            raise ValueError("identities.evaluation: must correspond to an enabled evaluation")


def read_records(path: str | Path) -> list[dict]:
    path = Path(path)
    records = []
    ids = set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                record = json.loads(line, parse_constant=reject_constant)
                validate_record(record)
                if record["run_id"] in ids:
                    raise ValueError("duplicate run_id")
                ids.add(record["run_id"])
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            records.append(record)
    return records


def read_manifest(path: str | Path, rows: list[dict] | None = None) -> dict:
    from .config import Config
    from .io import read_json

    manifest = read_json(Path(path))
    validate("manifest", manifest)
    cases = manifest["cases"]
    for case_id, resolved in cases.items():
        config = Config.from_dict(resolved)
        if config.to_dict() != resolved or config.case_id != case_id:
            raise ValueError(f"manifest.cases.{case_id}: config and case ID disagree")
    expected = manifest["expected"]
    repetitions = manifest["repetitions"]
    sweep_id = manifest["sweep_id"]
    seen = set()
    for order, entry in enumerate(expected, 1):
        case_id, repetition = entry["case_id"], entry["repetition"]
        key = (case_id, repetition)
        if (
            case_id not in cases or repetition > repetitions or key in seen
            or entry["run_order"] != order or entry["run_id"] != f"{sweep_id}-{case_id}-r{repetition:03d}"
        ):
            raise ValueError(f"manifest.expected[{order - 1}]: invalid case, repetition or run ID")
        seen.add(key)
    if len(seen) != len(cases) * repetitions:
        raise ValueError("manifest.expected: missing declared repetitions")
    if rows is not None:
        index = {entry["run_id"]: entry for entry in expected}
        for row in rows:
            declared = index.get(row["run_id"])
            if (
                row["sweep_id"] != sweep_id or declared is None
                or any(row[key] != value for key, value in declared.items())
                or row["config"] != cases[row["case_id"]] or row["warmups"] != manifest["warmups"]
            ):
                raise ValueError(f"manifest: record {row['run_id']} disagrees with the declaration")
    return manifest
