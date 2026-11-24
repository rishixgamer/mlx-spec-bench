# Speculative decoding on Apple M5 Pro

All 96 runs completed. Speculation was slower in every matched configuration.
Across 32 configurations, the median decode rate was 0.59× target-only
(range 0.49×–0.85×). Six of 64 speculative repetitions produced a different
greedy token. Stop reasons and target-cache windows agreed in all pairs.

## Setup

The target was Qwen2.5-1.5B, tested at FP16, Q8, Q6 and Q4. All speculative trials used a Qwen2.5-0.5B FP16 draft at K=1 or K=4. K=0 was the target-only control. Each case ran five prompts, generated up to 32 tokens at temperature zero, and ran one warm-up and two measured repetitions. Cache policies were unbounded and 16 MiB sliding. The batch setting caps cohort size. The implementation runs a single model forward at a time.

Ratios pair the same weight format, cache policy, cohort size, and repetition. Decode rate counts post-first tokens over the observed wall-clock decode span. Draft acceptance is accepted tokens divided by proposals. The table reports each case's two-repetition median and counts mismatching requests.

| Target | Cache | Cohort cap | K | Acceptance | Decode rate ratio | Mismatched requests |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | sliding | 1 | 1 | 87.1% | 0.626× | 0 |
| FP16 | sliding | 1 | 4 | 58.5% | 0.574× | 2 |
| FP16 | sliding | 2 | 1 | 91.5% | 0.595× | 0 |
| FP16 | sliding | 2 | 4 | 76.6% | 0.675× | 0 |
| FP16 | unbounded | 1 | 1 | 90.4% | 0.704× | 0 |
| FP16 | unbounded | 1 | 4 | 68.5% | 0.585× | 2 |
| FP16 | unbounded | 2 | 1 | 90.4% | 0.683× | 0 |
| FP16 | unbounded | 2 | 4 | 68.5% | 0.525× | 2 |
| Q4 | sliding | 1 | 1 | 72.5% | 0.514× | 0 |
| Q4 | sliding | 1 | 4 | 50.2% | 0.488× | 0 |
| Q4 | sliding | 2 | 1 | 74.4% | 0.506× | 0 |
| Q4 | sliding | 2 | 4 | 52.7% | 0.527× | 0 |
| Q4 | unbounded | 1 | 1 | 78.7% | 0.512× | 0 |
| Q4 | unbounded | 1 | 4 | 56.8% | 0.502× | 0 |
| Q4 | unbounded | 2 | 1 | 78.7% | 0.503× | 0 |
| Q4 | unbounded | 2 | 4 | 56.8% | 0.494× | 0 |
| Q6 | sliding | 1 | 1 | 77.5% | 0.549× | 0 |
| Q6 | sliding | 1 | 4 | 54.1% | 0.536× | 0 |
| Q6 | sliding | 2 | 1 | 89.2% | 0.613× | 0 |
| Q6 | sliding | 2 | 4 | 71.4% | 0.637× | 0 |
| Q6 | unbounded | 1 | 1 | 82.8% | 0.564× | 0 |
| Q6 | unbounded | 1 | 4 | 65.1% | 0.606× | 0 |
| Q6 | unbounded | 2 | 1 | 82.8% | 0.541× | 0 |
| Q6 | unbounded | 2 | 4 | 65.1% | 0.579× | 0 |
| Q8 | sliding | 1 | 1 | 88.1% | 0.668× | 0 |
| Q8 | sliding | 1 | 4 | 62.8% | 0.714× | 0 |
| Q8 | sliding | 2 | 1 | 91.5% | 0.742× | 0 |
| Q8 | sliding | 2 | 4 | 79.3% | 0.851× | 0 |
| Q8 | unbounded | 1 | 1 | 90.4% | 0.688× | 0 |
| Q8 | unbounded | 1 | 4 | 71.8% | 0.813× | 0 |
| Q8 | unbounded | 2 | 1 | 90.4% | 0.617× | 0 |
| Q8 | unbounded | 2 | 4 | 71.8% | 0.729× | 0 |

## Output agreement and memory

- The 44-token story prompt first differed at generated token 27 in six
  repetitions across three FP16 K=4 settings. The target-only token was 1435
  and the speculative token was 279. A focused unbounded replay found these
  two logits nearly tied. Separate calls to upstream mlx-lm on the same
  weights and prompt also disagreed at token 27, with the two choices
  reversed. The probe does not establish which code path causes the divergence.

Peak process footprint across all repetitions was 6.79 GB. The median paired footprint increase with speculation was 1.21 GB.

The reference release checks require identical greedy tokens, stops and target-cache windows. The sweep failed that gate. It also lacks the required source commit field in the manifest.

## Reproduce

```sh
mlx-bench run examples/reference-sweep.yaml --output-dir runs \
  --source-commit "$(git rev-parse HEAD)" --host-notes "Describe the actual machine load"
python examples/analyze-reference.py runs/sweep-ID > summary.json
```

This session used an Apple M5 Pro with 48 GiB, macOS 27.2.1, MLX 0.31.2, and mlx-lm 0.31.3. The desktop was active, with brief concurrent build and benchmark work. With two repetitions per case, the rates describe this run and give little precision about small differences. The target and draft parameter counts were fixed. The five fixed prompts range from 6 to 691 input tokens and also differ in
content. All 30 installed source and data files were checked against commit
`7109b92`, which was used to build the benchmark package.
