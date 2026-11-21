from dataclasses import replace
from importlib.resources import files
import json
from types import SimpleNamespace

import pytest

from mlx_spec_bench import CachePolicy, Config, ModelSpec
from mlx_spec_bench.cache import CachePlan
from mlx_spec_bench.decode import Prompt, generate
from mlx_spec_bench.models import check_tokenizers, metadata, resolve_artifact
from mlx_spec_bench.trial import execute

DATA = files("mlx_spec_bench").joinpath("data")


def _trial(backend, values, monkeypatch, model_cache):
    _, metal = backend
    monkeypatch.setenv("MLX_SPEC_BENCH_CACHE", str(model_cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    config = Config.from_dict(values)
    result = execute(config, 0, require_metal=metal)
    assert result["status"] == "success", result["error"]
    return result


def _tokens(result):
    return [
        (trace["input_ids"], trace["output_ids"], trace["stop_reason"]) for trace in result["data"]["requests"]
    ]


def test_baseline_suite_resolution_mixed_lengths_and_partial_batch(backend, tmp_path, monkeypatch):
    values = {"target": {"id": "fixture"}, "batch_size": 2}
    by_id = _trial(backend, values, monkeypatch, tmp_path)
    by_path = _trial(backend, {**values, "suite": str(DATA / "fixture-v1.json")}, monkeypatch, tmp_path)
    repeat = _trial(backend, values, monkeypatch, tmp_path)
    assert by_id["identities"] == by_path["identities"] == repeat["identities"]
    assert _tokens(by_id) == _tokens(by_path) == _tokens(repeat)
    expected = json.loads((DATA / "fixture-baseline-v1.json").read_text())
    assert by_id["metrics"]["tokens"]["input"] == expected["input_tokens"]
    assert by_id["metrics"]["tokens"]["output"] == expected["output_tokens"]
    for actual, expected in zip(by_id["data"]["requests"], expected["requests"], strict=True):
        assert actual["prompt_id"] == expected["prompt_id"]
        assert len(actual["input_ids"]) == expected["input_tokens"]
        assert actual["output_ids"] == expected["output_ids"]
        assert actual["stop_reason"] == expected["stop_reason"]
    assert by_id["metrics"]["task_score"] == 0.25
    assert by_id["metrics"]["quality"]["task_examples"] == 4
    assert next(item for item in by_id["data"]["quality"]["text"] if item["id"] == "empty")["scored_tokens"] == 0


@pytest.mark.parametrize("fmt", ["fp16", "q8", "q6", "q4"])
def test_real_formats_and_matched_greedy(backend, tmp_path, monkeypatch, fmt):
    values = {"target": {"id": "fixture", "weight_format": fmt}, "batch_size": 2}
    for cache in ({}, {"policy": "sliding", "budget_bytes": 45056}):
        control = _trial(backend, {**values, "cache": cache}, monkeypatch, tmp_path)
        for k in (1, 7, 64):
            result = _trial(backend, {
                **values, "cache": cache, "draft": {"id": "fixture"}, "k": k,
            }, monkeypatch, tmp_path)
            assert _tokens(result) == _tokens(control)
            assert result["identities"]["target"] == control["identities"]["target"]
            assert result["metrics"]["quality"] == control["metrics"]["quality"]
            ident = result["identities"]["target"]
            assert ident["weight_format"] == fmt
            assert bool(ident["quantized_modules"]) == (fmt != "fp16")
            assert result["metrics"]["tokens"]["accepted"] > 0
            if cache:
                kv = result["metrics"]["kv"]
                assert kv["window_tokens"] == control["metrics"]["kv"]["window_tokens"] == 3
                assert 0 < kv["working_peak_bytes"] <= cache["budget_bytes"]
                assert kv["evictions"] > 30
                for trace in result["data"]["requests"]:
                    for event in trace["cache_events"]:
                        length = event["end"] - event["start"]
                        assert 0 < length <= kv["window_tokens"]
                        assert event["bytes"] == length * 512
                        assert event["evictions"] == event["start"]
            else:
                assert result["metrics"]["kv"]["evictions"] == 0


def test_fixture_conversion_reuses_verified_artifact(backend, tmp_path, monkeypatch):
    mx, _ = backend
    model = metadata(ModelSpec("fixture", "q6"), tmp_path)
    first = resolve_artifact(mx, model, tmp_path)
    monkeypatch.setattr("mlx_spec_bench.models._convert", lambda *_: pytest.fail("reconverted a verified artifact"))
    assert resolve_artifact(mx, model, tmp_path) == first
    (first[0] / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="cached file hashes disagree"):
        resolve_artifact(mx, model, tmp_path)


def test_equal_vocab_size_swapped_ids():
    target = SimpleNamespace(spec=ModelSpec("fixture"), signature={"vocabulary": {"A": 0, "B": 1}})
    draft = SimpleNamespace(spec=ModelSpec("qwen2.5-0.5b"), signature={"vocabulary": {"A": 1, "B": 0}})
    with pytest.raises(ValueError, match=r"fixture .* versus qwen2.5-0.5b .*vocabulary-to-ID mapping"):
        check_tokenizers(target, draft)


def test_preflight_and_context_are_checked_before_model_allocation(backend, tmp_path, monkeypatch):
    _, metal = backend
    monkeypatch.setenv("MLX_SPEC_BENCH_CACHE", str(tmp_path))
    monkeypatch.setattr("mlx_spec_bench.trial.resolve_artifact", lambda *_: pytest.fail("allocated model weights"))
    impossible = Config(ModelSpec("fixture"), cache=CachePolicy("sliding", 100))
    result = execute(impossible, 0, require_metal=metal)
    assert result["error"]["kind"] == "preflight"
    assert "requested 100 bytes, minimum" in result["error"]["message"]
    result = execute(replace(impossible, cache=CachePolicy(), memory_limit_bytes=1_000_000), 0, require_metal=metal)
    assert result["error"]["kind"] == "preflight"
    assert "1000000-byte allowance" in result["error"]["message"]
    over_limit = tmp_path / "over-limit.json"
    over_limit.write_text(json.dumps({
        "schema_version": 1, "id": "custom", "license": "MIT", "template": "{text}",
        "prompts": [{"id": "too-long", "text": "red " * 129, "reference": "green"}],
    }))
    result = execute(Config(ModelSpec("fixture"), suite=str(over_limit)), 0, require_metal=metal)
    assert result["error"]["kind"] == "config"
    assert "prompt 'too-long': observed 129 input tokens, context limit 128" in result["error"]["message"]


class ScriptedQwen:
    """Make acceptance deterministic while exercising the real transactional cache."""

    layers = [object()]

    def __init__(self, mx, transitions, window):
        self.mx, self.transitions, self.window = mx, transitions, window

    def __call__(self, ids, cache):
        mx = self.mx
        layer = cache[0]
        inputs = ids.tolist()[0]
        offsets = list(range(layer.offset, layer.offset + len(inputs)))
        shape = (1, 1, len(inputs), 1)
        mask = layer.make_mask(len(inputs), return_array=True).tolist()
        keys, _ = layer.update_and_fetch(mx.array(offsets).reshape(shape), mx.array(inputs).reshape(shape))
        positions = keys.reshape(-1).tolist()
        for row, position in zip(mask, offsets, strict=True):
            visible = [p for p, allowed in zip(positions, row, strict=True) if allowed]
            expected_begin = 0 if self.window is None else max(0, position + 1 - self.window)
            assert visible == list(range(expected_begin, position + 1))
        next_ids = mx.array([[self.transitions.get(token, 2) for token in inputs]])
        return (next_ids[..., None] == mx.arange(12)).astype(mx.float32)


@pytest.mark.parametrize("window", [None, 2])
@pytest.mark.parametrize(("alter", "expected_accepted", "expected_corrections"), [
    ({}, 3, 0), ({4: 9, 5: 9, 6: 9}, 0, 3), ({5: 9, 6: 2}, 2, 1), ({4: 2}, 2, 1),
])
def test_rejection_rollback_across_eviction(backend, window, alter, expected_accepted, expected_corrections):
    mx, _ = backend
    transitions = {1: 4, 4: 5, 5: 6, 6: 2}
    token_info = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_ids={2}))
    target = SimpleNamespace(model=ScriptedQwen(mx, transitions, window), metadata=token_info)
    draft = SimpleNamespace(model=ScriptedQwen(mx, {**transitions, **alter}, window), metadata=token_info)
    config = Config(ModelSpec("fixture"), ModelSpec("fixture"), batch_size=2, max_tokens=4, k=4)
    prompts = [Prompt("span", [10, 10, 10, 1]), Prompt("early-eos", [7])]
    cache = CachePlan(window, window, 2, 4096 if window else None)
    results = generate(mx, target, draft, config, cache, prompts)
    control = generate(mx, target, None, replace(config, draft=None, k=0), cache, prompts)
    assert [item.output_ids for item in results.traces] == [item.output_ids for item in control.traces]
    assert [item.stop_reason for item in results.traces] == ["eos", "eos"]
    assert [item.stop_reason for item in results.traces] == [item.stop_reason for item in control.traces]
    trace = results.traces[0]
    assert sum(e["count"] for e in trace.emissions if e["kind"] == "accepted") == expected_accepted
    assert sum(e["count"] for e in trace.emissions if e["kind"] == "corrections") == expected_corrections
    roles = [[(e["start"], e["end"]) for e in trace.cache_events if e["role"] == role] for role in ("target", "draft")]
    assert roles[0] == roles[1]
    assert bool(results.cache.evictions) == bool(window)
