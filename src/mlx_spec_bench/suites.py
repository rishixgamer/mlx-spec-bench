"""Redistributable versioned workloads and their rendered prompt text."""

from dataclasses import dataclass
from importlib.resources import files
import json
from pathlib import Path
from string import Formatter

from .config import BUILTIN_SUITES, _read_yaml
from .io import fingerprint, reject_constant
from .schema import validate


@dataclass(frozen=True)
class Suite:
    data: dict
    sha256: str

    @property
    def identity(self) -> dict:
        return {"id": self.data["id"], "sha256": self.sha256}

    def render(self, text: str) -> str:
        return self.data["template"].format(text=text)


def _load_builtin(name: str) -> dict:
    resource = files("mlx_spec_bench").joinpath("data", name + ".json")
    return json.loads(resource.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _checked(kind: str, data: dict, groups: tuple[str, ...]) -> Suite:
    validate(kind, data)
    template = data["template"]
    fields = [(key, spec, conversion) for _, key, spec, conversion in Formatter().parse(template) if key is not None]
    if fields != [("text", "", None)]:
        raise ValueError(f"{kind}.template: use exactly one {{text}} field without format modifiers")
    for group in groups:
        ids = [entry["id"] for entry in data[group]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{kind}.{group}: duplicate IDs")
    return Suite(data, fingerprint(data))


def load_suite(name: str) -> Suite:
    if name in BUILTIN_SUITES:
        data = _load_builtin(name)
    else:
        path = Path(name)
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
        else:
            data = _read_yaml(path)
    return _checked("suite", data, ("prompts",))


def load_evaluation() -> Suite:
    return _checked("evaluation", _load_builtin("held-out-v1"), ("text", "tasks"))
