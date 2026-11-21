from dataclasses import asdict

import pytest

from mlx_spec_bench.decode import CacheStats, Generation, Trace
from mlx_spec_bench.metrics import generation_metrics
from mlx_spec_bench.qualification import REFERENCE_EOS_ID, _check_quality, _check_requests


def _evidence():
    trace = Trace("prompt", [3], 0, [7, 8], stop_reason="max_tokens")
    trace.emissions = [
        {"at_ns": 1, "count": 1, "kind": "autoregressive"},
        {"at_ns": 2, "count": 1, "kind": "autoregressive"},
    ]
    trace.cache_events = [
        {"role": "target", "start": 0, "end": 1, "bytes": 4, "staged_bytes": 4, "evictions": 0},
        {"role": "target", "start": 1, "end": 2, "bytes": 4, "staged_bytes": 8, "evictions": 1},
    ]
    generation = Generation([trace], 1, 2, CacheStats(committed_peak=4, working_peak=16, evictions=1))
    metrics = generation_metrics(generation, resident=1, peak=1, quality=None, window=1)
    row = {"config": {"k": 0, "cache": {"policy": "sliding", "budget_bytes": 16}, "max_tokens": 2}, "metrics": metrics}
    detail = {
        "cache_plan": {"target_window": 1, "draft_window": None, "budget_bytes": 16, "prefill_chunk": 1},
        "requests": [asdict(trace)],
    }
    return row, detail


def test_qualification_reconciles_detailed_requests():
    row, detail = _evidence()
    _check_requests(detail, row, ["prompt"])


@pytest.mark.parametrize(("change", "message"), [
    ({"output_ids": [7]}, "invalid token IDs, stop reason"),
    ({"output_ids": [REFERENCE_EOS_ID, 8]}, "invalid token IDs, stop reason"),
    ({"emissions": []}, "emission count disagrees"),
    ({"cache_events": []}, "missing prefill cache evidence"),
])
def test_qualification_rejects_invalid_raw_requests(change, message):
    row, detail = _evidence()
    detail["requests"][0].update(change)
    with pytest.raises(ValueError, match=message):
        _check_requests(detail, row, ["prompt"])


def test_qualification_reconciles_quality_totals():
    evaluation = {"text": [{"id": "passage"}], "tasks": [{"id": "task", "answer": "test"}]}
    quality = {
        "text": [{"id": "passage", "input_ids": [1, 2], "scored_mask": [False, True], "scored_tokens": 1, "total_nll": 2.0}],
        "tasks": [{"id": "task", "input_ids": [3], "output_ids": [REFERENCE_EOS_ID], "stop_reason": "eos",
                   "actual": "", "expected": "test", "correct": False}],
    }
    totals = {"scored_tokens": 1, "total_nll": 2.0, "task_correct": 0, "task_examples": 1}
    _check_quality(quality, evaluation, totals, 2)
    with pytest.raises(ValueError, match="evaluation totals disagree"):
        _check_quality(quality, evaluation, {**totals, "scored_tokens": 2}, 2)


def test_qualification_requires_the_log_for_each_completed_run(tmp_path):
    from pathlib import Path
    from mlx_spec_bench.qualification import _case_problems
    from mlx_spec_bench.schema import read_records

    row = read_records(Path(__file__).parent / "fixtures" / "golden.jsonl")[0]
    log = tmp_path / "worker-logs" / (row["run_id"] + ".log")
    log.parent.mkdir()
    problem = f"{row['run_id']}: missing worker log"
    assert problem in _case_problems(tmp_path, [row])
    log.touch()  # A successful worker may produce no console output.
    assert problem not in _case_problems(tmp_path, [row])
