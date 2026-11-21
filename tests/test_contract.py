from dataclasses import replace
from pathlib import Path

import pytest

from mlx_spec_bench import Config, ModelSpec, Sweep, load_config, load_sweep
from mlx_spec_bench.qualification import reference_sweep

ROOT = Path(__file__).resolve().parents[1]


def test_minimal_and_speculative_config():
    config = Config.from_dict({"target": {"id": "fixture"}})
    assert config == Config(ModelSpec("fixture"))
    values = config.to_dict()
    values.update(k=2, draft={"id": "fixture", "weight_format": "q4"})
    assert Config.from_dict(values).draft.weight_format == "q4"
    assert Config.from_dict(values).target.weight_format == "fp16"


@pytest.mark.parametrize(("fields", "diagnostic"), [
    ({"batch_size": 0}, "batch_size"),
    ({"batch_size": 2.0}, "batch_size"),
    ({"max_tokens": True}, "max_tokens"),
    ({"target": {"id": "fixture", "weight_format": "q5"}}, r"target.weight_format"),
    ({"k": 1}, "draft"),
    ({"temperature": float("nan")}, "temperature"),
    ({"cache": {"budget_bytes": 0}}, "cache.budget_bytes"),
    ({"cache": {"policy": "sliding"}}, "cache.budget_bytes"),
    ({"cache": {"policy": "unbounded", "budget_bytes": 32}}, "cache.budget_bytes"),
    ({"memory_limit_bytes": -1}, "memory_limit_bytes"),
    ({"schema_version": 2}, "schema_version"),
    ({"nonsense": 5}, "nonsense"),
    ({"suite": "unknown-suite"}, "suite"),
])
def test_invalid_config(fields, diagnostic):
    with pytest.raises(ValueError, match=diagnostic):
        Config.from_dict({"target": {"id": "fixture"}, **fields})


def test_paths_and_duplicate_yaml(tmp_path):
    test_file = tmp_path / "config.yaml"
    test_file.write_text("target: {id: fixture}\nsuite: suite.json\n")
    assert load_config(test_file).suite == str(tmp_path / "suite.json")
    test_file.write_text("target: {id: fixture}\nsuite: fixture-v1\nsuite: fixture-v1\n")
    with pytest.raises(ValueError, match="duplicate YAML field 'suite'"):
        load_config(test_file)


def test_matrices_are_declared_and_stable():
    fixture = load_sweep(ROOT / "examples/fixture-sweep.yaml")
    assert len(fixture.cases) == 24
    assert all((case.draft is not None) == bool(case.k) for case in fixture.cases)
    editable = load_sweep(ROOT / "examples/reference-sweep.yaml")
    packaged = reference_sweep()
    assert editable == packaged
    assert len(editable.cases) * editable.repetitions == 96
    assert len({case.case_id for case in editable.cases}) == 48
    with pytest.raises(ValueError, match="draft"):
        Sweep.from_dict({"base": {"target": {"id": "fixture"}}, "matrix": {"k": [0, 3]}})
    same = replace(fixture.cases[0])
    with pytest.raises(ValueError, match="duplicate"):
        Sweep((same, same)).validate()
