"""Versioned YAML configuration and deterministic sweep expansion."""

from dataclasses import asdict, dataclass
from itertools import product
from os.path import abspath
from pathlib import Path

import yaml

from .io import fingerprint
from .schema import validate

FIXED_SUITE = "fixture-v1"
REFERENCE_SUITE = "reference-v1"
BUILTIN_SUITES = {FIXED_SUITE, REFERENCE_SUITE}


class ConfigLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError(f"YAML key at line {key_node.start_mark.line + 1} must be text")
            if key in mapping:
                raise ValueError(f"duplicate YAML field {key!r}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _read_yaml(path: Path):
    return yaml.load(path.read_text(encoding="utf-8"), Loader=ConfigLoader)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    weight_format: str = "fp16"


@dataclass(frozen=True)
class CachePolicy:
    policy: str = "unbounded"
    budget_bytes: int | None = None


@dataclass(frozen=True)
class Config:
    target: ModelSpec
    draft: ModelSpec | None = None
    suite: str = FIXED_SUITE
    batch_size: int = 1
    max_tokens: int = 8
    k: int = 0
    temperature: float = 0.0
    cache: CachePolicy = CachePolicy()
    memory_limit_bytes: int | None = None
    timeout_seconds: float = 120.0
    seed: int = 0
    quality: bool = True
    schema_version: int = 1

    @classmethod
    def from_dict(cls, data: dict, *, base_dir: Path | None = None) -> "Config":
        validate("config", data)
        values = dict(data)
        suite = values.get("suite", FIXED_SUITE)
        if suite not in BUILTIN_SUITES:
            looks_like_path = Path(suite).suffix.lower() in {".json", ".yaml", ".yml"}
            if not looks_like_path:
                raise ValueError(f"suite: unknown suite {suite!r}; use a built-in ID or suite path")
            values["suite"] = abspath((base_dir or Path.cwd()) / suite)
        values["target"] = ModelSpec(**values["target"])
        if values.get("draft") is not None:
            values["draft"] = ModelSpec(**values["draft"])
        values["cache"] = CachePolicy(**values.get("cache", {}))
        values["temperature"] = float(values.get("temperature", 0)) or 0.0
        values["timeout_seconds"] = float(values.get("timeout_seconds", 120))
        config = cls(**values)
        if config.k > 0 and config.draft is None:
            raise ValueError("draft: required when k is greater than zero")
        if config.k == 0 and config.draft is not None:
            raise ValueError("draft: set k greater than zero to use a draft")
        if config.k and config.temperature:
            raise ValueError("temperature: speculative decoding requires temperature 0")
        if config.cache.policy == "sliding" and config.cache.budget_bytes is None:
            raise ValueError("cache.budget_bytes: required for the sliding policy")
        if config.cache.policy == "unbounded" and config.cache.budget_bytes is not None:
            raise ValueError("cache.budget_bytes: use null for the unbounded policy")
        return config

    def to_dict(self) -> dict:
        values = asdict(self)
        for name in ("temperature", "timeout_seconds"):
            value = values[name]
            if type(value) in (float, int):
                values[name] = float(value) or 0.0
        return values

    @property
    def case_id(self) -> str:
        return "case-" + fingerprint(self.to_dict())[:24]


@dataclass(frozen=True)
class Sweep:
    cases: tuple[Config, ...]
    warmups: int = 0
    repetitions: int = 1

    @classmethod
    def from_dict(cls, data: dict, *, base_dir: Path | None = None) -> "Sweep":
        validate("sweep", data)
        base = data["base"]
        matrix = data.get("matrix", {})
        names = ("weight_format", "k", "cache", "batch_size")
        defaults = (
            [base["target"].get("weight_format", "fp16")],
            [base.get("k", 0)],
            [base.get("cache", {})],
            [base.get("batch_size", 1)],
        )
        cases = []
        for weight_format, k, cache, batch_size in product(
            *(matrix.get(name, default) for name, default in zip(names, defaults))
        ):
            values = {
                **base,
                "schema_version": 1,
                "target": {**base["target"], "weight_format": weight_format},
                "k": k,
                "cache": cache,
                "batch_size": batch_size,
            }
            if k == 0 and "k" in matrix:
                values.pop("draft", None)
            cases.append(Config.from_dict(values, base_dir=base_dir))
        result = cls(tuple(cases), data.get("warmups", 0), data.get("repetitions", 1))
        result.validate()
        return result

    def validate(self) -> None:
        if type(self.warmups) is not int or not 0 <= self.warmups <= 20:
            raise ValueError("warmups: expected an integer in 0..20")
        if type(self.repetitions) is not int or not 1 <= self.repetitions <= 100:
            raise ValueError("repetitions: expected an integer in 1..100")
        if not self.cases:
            raise ValueError("cases: expected at least one case")
        ids = [Config.from_dict(case.to_dict()).case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("matrix: contains duplicate resolved cases")


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    return Config.from_dict(_read_yaml(path), base_dir=path.parent)


def load_sweep(path: str | Path) -> Sweep:
    path = Path(path).resolve()
    data = _read_yaml(path)
    if isinstance(data, dict) and "base" in data:
        return Sweep.from_dict(data, base_dir=path.parent)
    return Sweep((Config.from_dict(data, base_dir=path.parent),))
