"""Timing and token totals from observable emission timestamps."""

from .decode import Generation
from .quality import corpus_perplexity


def generation_metrics(generation: Generation, *, resident: int, peak: int, quality: dict | None,
                       window: int | None) -> dict:
    ttft_total_ns = itl_total_ns = 0
    ttft_count = itl_count = decode_count = 0
    first_with_decode = []
    last_with_decode = []
    tokens = {
        "input": 0, "output": 0, "accepted": 0, "proposed": 0,
        "corrections": 0, "bonus": 0, "autoregressive": 0,
    }
    for trace in generation.traces:
        times = []
        tokens["input"] += len(trace.input_ids)
        tokens["output"] += len(trace.output_ids)
        tokens["proposed"] += trace.proposed
        for event in trace.emissions:
            times.extend([event["at_ns"]] * event["count"])
            tokens[event["kind"]] += event["count"]
        if times:
            ttft_count += 1
            ttft_total_ns += times[0] - trace.submit_ns
        if len(times) > 1:
            post_first = len(times) - 1
            itl_count += post_first
            itl_total_ns += times[-1] - times[0]
            decode_count += post_first
            first_with_decode.append(times[0])
            last_with_decode.append(times[-1])

    # Wall-clock span across decoding cohorts. Includes time between cohorts.
    duration = max(last_with_decode) - min(first_with_decode) if decode_count else 0
    reasons = {}

    def rate(key, numerator, denominator, reason):
        if denominator:
            return numerator / denominator
        reasons[key] = reason
        return None

    ppl = corpus_perplexity(quality["total_nll"], quality["scored_tokens"]) if quality else None
    if ppl is None:
        reasons["perplexity"] = "no_scored_tokens" if quality else "quality_disabled"
    return {
        "ttft_ms": rate("ttft_ms", ttft_total_ns / 1e6, ttft_count, "no_output_tokens"),
        "itl_ms_per_token": rate("itl_ms_per_token", itl_total_ns / 1e6, itl_count,
                                 "fewer_than_two_tokens_per_request"),
        "decode_tokens_per_second": rate(
            "decode_tokens_per_second", decode_count * 1e9, duration,
            "zero_decode_span" if decode_count else "no_post_first_token_emissions",
        ),
        "acceptance": rate("acceptance", tokens["accepted"], tokens["proposed"], "no_draft_proposals"),
        "perplexity": ppl,
        "task_score": rate("task_score", quality["task_correct"] if quality else 0,
                           quality["task_examples"] if quality else 0,
                           "no_scored_examples" if quality else "quality_disabled"),
        "undefined": reasons,
        "tokens": tokens,
        "observations": {
            "requests": len(generation.traces), "ttft": ttft_count, "itl": itl_count,
            "decode_tokens": decode_count, "decode_duration_ns": duration,
            "ttft_total_ns": ttft_total_ns, "itl_total_ns": itl_total_ns,
        },
        "generation_ns": generation.generation_ns,
        "prefill_ns": generation.prefill_ns,
        "mlx_allocator_peak_bytes": peak,
        "mlx_resident_start_bytes": resident,
        "kv": {
            "window_tokens": window, "committed_peak_bytes": generation.cache.committed_peak,
            "working_peak_bytes": generation.cache.working_peak, "evictions": generation.cache.evictions,
        },
        "quality": quality,
    }
