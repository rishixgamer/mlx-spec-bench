import math

import pytest

from mlx_spec_bench.decode import CacheStats, Generation, Trace
from mlx_spec_bench.metrics import generation_metrics
from mlx_spec_bench.quality import corpus_perplexity, masked_nll, task_matches


def test_timing_simultaneous_staggered_and_single_token():
    first = Trace("first", [1], 1_000_000)
    first.emit([4], "autoregressive", 3_000_000)
    first.emit([5, 6], "accepted", 7_000_000)
    first.proposed = 3
    second = Trace("second", [2], 1_000_000)
    second.emit([4], "autoregressive", 5_000_000)
    second.emit([3], "corrections", 9_000_000)
    single = Trace("single", [8], 20_000_000)
    single.emit([8], "autoregressive", 25_000_000)
    generation = Generation([first, second, single], 10_000_000, 26_000_000, CacheStats())
    result = generation_metrics(generation, resident=300, peak=800, quality=None, window=None)
    assert result["ttft_ms"] == pytest.approx((2 + 4 + 5) / 3)
    assert result["itl_ms_per_token"] == pytest.approx((4 + 0 + 4) / 3)
    assert result["decode_tokens_per_second"] == 500
    assert result["acceptance"] == pytest.approx(2 / 3)
    assert result["observations"]["decode_duration_ns"] == 6_000_000


def test_undefined_rates_and_zero_itl():
    only = Trace("only", [1], 1)
    only.emit([2], "autoregressive", 5)
    result = generation_metrics(Generation([only], 1, 6, CacheStats()), resident=0, peak=0, quality=None, window=None)
    assert result["ttft_ms"] is not None
    assert result["itl_ms_per_token"] is None
    assert result["undefined"]["decode_tokens_per_second"] == "no_post_first_token_emissions"
    only.emit([3], "autoregressive", 5)
    result = generation_metrics(Generation([only], 1, 6, CacheStats()), resident=0, peak=0, quality=None, window=None)
    assert result["itl_ms_per_token"] == 0
    assert result["undefined"]["decode_tokens_per_second"] == "zero_decode_span"


def test_perplexity_with_padding_and_empty_span(backend):
    mx, _ = backend
    logits = mx.array([[0, math.log(3)], [9, 0], [0, 0]])
    loss, count = masked_nll(mx, logits, [1, 0, 1], [True, False, True])
    assert count == 2
    assert loss == pytest.approx(-math.log(0.75) - math.log(0.5))
    assert corpus_perplexity(loss, count) == pytest.approx(1 / math.sqrt(0.75 * 0.5))
    assert masked_nll(mx, logits, [0, 0, 0], [False] * 3) == (0, 0)
    assert corpus_perplexity(0, 0) is None


def test_task_exact_match_rules():
    assert task_matches("  GREEN\nＧＯＬＤ  ", "green gold")
    assert not task_matches("green gold.", "green gold")
