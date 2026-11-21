# v1 configuration and measurement contract

The package ships JSON Schemas for [configs](../src/mlx_spec_bench/schemas/config.json), [sweeps](../src/mlx_spec_bench/schemas/sweep.json), [prompt suites](../src/mlx_spec_bench/schemas/suite.json), [run rows](../src/mlx_spec_bench/schemas/record.json), and [manifests](../src/mlx_spec_bench/schemas/manifest.json). The validators additionally check relationships among fields, reject non-finite numbers, and require integer fields to use integer values. All contracts use schema version 1.

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `schema_version` | 1 | Only 1 is supported. |
| `target.id` | Required | A registered fixture or Qwen2.5 model ID. |
| `target.weight_format`, `draft.weight_format` | `fp16` | Precision for each model. Use `fp16`, `q8`, `q6`, or `q4`. |
| `draft` | `null` | Model ID and format, required when `k` is positive. |
| `suite` | `fixture-v1` | Built-in suite ID or a JSON/YAML file path relative to the config. The other built-in ID is `reference-v1`. |
| `batch_size` | 1 | Maximum live requests in a logical cohort, 1–64. Workers schedule individual requests round-robin without padding. |
| `max_tokens` | 8 | Maximum emitted tokens per request, 1–4,096. An emitted EOS counts toward this limit. |
| `k` | 0 | Maximum draft block length, 0–64. Positive K selects speculative greedy decoding. |
| `temperature` | 0.0 | 0 gives greedy argmax with the lowest token ID on ties. Positive values up to 10 sample target-only with the supplied seed. |
| `cache.policy` | `unbounded` | Use `unbounded` or `sliding`. |
| `cache.budget_bytes` | `null` | Sliding requires a positive integer byte budget for all KV slots. Unbounded uses null. |
| `memory_limit_bytes` | `null` | Trial allowance in bytes. Default is 85% of physical RAM. The effective allowance is capped at physical RAM. |
| `timeout_seconds` | 120 | Total deadline per worker including setup, warm-up, inference, and quality, 0.1–86,400 seconds. |
| `seed` | 0 | Sampling seed, an unsigned 32-bit integer. |
| `quality` | true | Enables the pinned held-out evaluation. |

A sweep has a base configuration and optional Cartesian axes for target `weight_format`, `k`, `cache`, and `batch_size`. Axis values iterate in that order with the final axis changing fastest. A zero in the K axis drops the draft from the resolved control. Warm-ups default to zero (range 0–20) and repetitions to one (range 1–100). Case IDs derive from the canonical resolved config. Run IDs include the sweep UUID, case ID, and repetition index.

Suite templates contain one literal `{text}` field. The runner encodes rendered text with `add_special_tokens=False` and checks input plus generation allowance against both models' advertised context limits. Input IDs are preserved in full. References describe the prompt workload. Scored tasks come from the held-out evaluation.

## Greedy verification and cache accounting

The target emits its first greedy token and then verifies each draft block conditioned on the committed prefix. The run emits the accepted prefix and a target correction at the first disagreement, or a bonus token when room remains after full acceptance. The target and draft caches commit the same accepted prefix and discard the remaining staged candidates. A correction or bonus is forwarded to both models before the next round. Draft EOS caps the proposals in a block. EOS and output length are checked for every request.

The sliding policy preserves the most recent window of committed positions. Each layer uses the absolute offset for rotary position embeddings and masks positions outside the query's sliding window, including while verifying a block. The budget is divided into equal target and draft slots for each configured cohort member. Each slot reserves two copies for append and compaction overlap and `min(64, max_tokens)` token rows for staging. The remaining rows define that model's window. Target-only reserves the same target slot, so its target eviction semantics match a speculative case at the same budget and batch size. The draft slot is idle for target-only. Each row uses two FP16 values per key/value channel per layer. Prefill is chunked, with at most 32 tokens or the bounded staging reserve, whichever is smaller.

Detailed cache events give the retained half-open range, bytes in live per-layer KV arrays, staged bytes, and cumulative committed evictions for each model and request. The cohort working peak applies the two-copy charge under sliding. GPU kernel workspace and allocator caching are accounted for separately by the allocator and process observations. Numerical differences between single- and multi-token kernels can matter near ties. The qualification gate compares each full greedy output against its matched autoregressive control.

## Timing and memory

Each measured repetition runs in a fresh worker. Load, warm-up, timed generation, and quality are separate spans. Timestamps use the worker's monotonic clock with nanosecond resolution. MLX output and cache arrays are explicitly evaluated and synchronized before a token or block is observed.

| Metric | Unit and denominator |
|---|---|
| `ttft_ms` | Mean milliseconds from cohort submission to each request's first emitted token. Uses the request count. Includes queueing inside the cohort and requested draft prefill. |
| `itl_ms_per_token` | Mean milliseconds between successive emissions for each request. Pools all post-first intervals. Tokens in one verified block use the same timestamp and contribute zero-length intervals. |
| `decode_tokens_per_second` | Total post-first tokens for all requests, divided by wall time from the earliest first token of a request with a decode interval to the latest last token of those requests. Includes time between staggered requests and cohorts. |
| `acceptance` | Accepted draft tokens divided by all proposals, including proposed tokens discarded after a disagreement. Corrections, bonuses, and autoregressive tokens are counted separately. |
| `mlx_allocator_peak_bytes` | MLX allocator peak since the post-warm-up reset, floored by MLX active bytes before generation. Includes resident evaluated model weights. |
| `peak_process_footprint_bytes` | Peak sampled sum of macOS `ri_phys_footprint` for the parent worker supervisor, active worker, and worker descendants. Includes loading, warm-up, timed generation, and quality. |
| `peak_process_rss_bytes` | Independent sampled process-tree RSS peak. The report identifies it when physical-footprint samples are unavailable. |

Process and system observations target 50 ms intervals. Short-lived peaks can fall between samples. Footprint and RSS can account for shared pages across processes. System fields include available memory and swap in bytes, and macOS VM pressure samples. GB in reports means 1,000,000,000 bytes. The allocator and footprint describe overlapping unified memory and are reported separately.

Preflight estimates converted resident weights, cold conversion when necessary, KV storage, verification workspace, and a fixed runtime allowance. It reports oversized settings as `preflight`. A worker-reported allocation error is `allocation_failure`, a sampled overage is `memory_limit`, a deadline is `timeout`, an unexplained exit is `worker_exit`, and a cancellation is `interrupted`. Failure rows retain any available identities and process observations. Metrics without a measurement use JSON null and an explicit reason, such as `no_draft_proposals`, `fewer_than_two_tokens_per_request`, `zero_decode_span`, or `quality_disabled`. Terminal failures use null metrics and provide `error.kind` and `error.message`.

## Quality and comparisons

The [held-out split](../src/mlx_spec_bench/data/held-out-v1.json) includes text continuations and scored generation tasks. Evaluation uses the case's target weights and target cache policy, with target-only greedy generation for tasks. Scored text tokens must lie wholly inside the continuation and have a preceding token. Padding, empty spans, and boundary-crossing tokens contribute zero scored tokens. Corpus perplexity is `exp(sum(negative log likelihood) / scored tokens)`. Exact-match task score divides normalized matches by all examples. Normalization applies Unicode NFKC, case folding, and whitespace collapsing after dropping special tokens.

Reports pool observation totals across repetitions and take maxima for memory. Speedup compares decode throughput against a target-only control on the same host/software, workload, target artifact, cache policy and window, batch, token cap, temperature, and seed. Quality deltas compare against an explicitly identified FP16/unbounded target-only case with the same source revision, tokenizer, evaluation rules and identity, host, workload, and generation controls. Missing or ambiguous controls are flagged. Every row provides the case ID and contributing run IDs.
