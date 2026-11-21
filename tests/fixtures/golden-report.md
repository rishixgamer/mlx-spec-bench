# MLX benchmark comparison

## golden-v1 · `aaaaaaaaaaaa`

| Case | Target | Draft | Batch | K | KV bytes | TTFT ms | ITL ms/token | Decode tok/s | Acceptance | Peak GB | MLX GB | PPL | Task | OK/attempted | Speedup | Δ PPL | Δ task pp |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `case-2b728ac4c5361dc4384a4364` | fixture/fp16 | — | 1 | 0 | unbounded | 3.000 | 10.000 | 100.000 | — | 0.380 footprint | 0.100 | 3.175 | 0.750 | 2/2 | 1.000 | 0.000 | 0.000 |
| `case-13860d7ce630e5ef01521d21` | fixture/fp16 | fixture/fp16 | 1 | 2 | unbounded | 3.000 | 5.000 | 200.000 | 0.500 | 0.480 footprint | 0.100 | 2.000 | 0.500 | 1/1 | 2.000 | -1.175 | -25.000 |
| `case-3f10622b8fb1b123a93f326c` | fixture/q4 | — | 1 | 0 | 16384 | 4.000 | 9.000 | 111.111 | — | 0.580 footprint | 0.100 | 4.000 | 0.000 | 1/1 | 1.000 | 0.825 | -75.000 |
| `case-4083836b192d89ab8fcf4c72` | fixture/q8 | fixture/fp16 | 1 | 4 | unbounded | 4.000 | 8.000 | 125.000 | 0.500 | 0.680 footprint | 0.100 | 2.000 | 0.500 | 1/1 | — | -1.175 | -25.000 |
| `case-739447885e1b9c43148ed777` | fixture/q8 | fixture/fp16 | 1 | 5 | unbounded | — | — | — | — | 0.780 footprint | — | — | — | 0/1 | — | — | — |

### Run evidence

- `case-2b728ac4c5361dc4384a4364` · `00000000000000000000000000000000-case-2b728ac4c5361dc4384a4364-r001`, `00000000000000000000000000000000-case-2b728ac4c5361dc4384a4364-r002` · speed control `case-2b728ac4c5361dc4384a4364` · quality control `case-2b728ac4c5361dc4384a4364` (FP16/unbounded) · acceptance undefined (zero proposals)
- `case-13860d7ce630e5ef01521d21` · `00000000000000000000000000000000-case-13860d7ce630e5ef01521d21-r001` · speed control `case-2b728ac4c5361dc4384a4364` · quality control `case-2b728ac4c5361dc4384a4364` (FP16/unbounded)
- `case-3f10622b8fb1b123a93f326c` · `00000000000000000000000000000000-case-3f10622b8fb1b123a93f326c-r001` · speed control `case-3f10622b8fb1b123a93f326c` · quality control `case-2b728ac4c5361dc4384a4364` (FP16/unbounded) · acceptance undefined (zero proposals)
- `case-4083836b192d89ab8fcf4c72` · `00000000000000000000000000000000-case-4083836b192d89ab8fcf4c72-r001` · missing/incomparable target-only control · quality control `case-2b728ac4c5361dc4384a4364` (FP16/unbounded)
- `case-739447885e1b9c43148ed777` · `00000000000000000000000000000000-case-739447885e1b9c43148ed777-r001` · timeout (fixture deadline expired)

TTFT and ITL pool their observation counts. Decode tok/s divides pooled post-first tokens by the sum of measured decode spans. Acceptance and task score pool their counts. PPL exponentiates pooled NLL per scored token. Peaks are maxima. GB is 1,000,000,000 bytes. Speedup uses decode throughput, and quality deltas are measured minus control.
