# mlx-spec-bench

Benchmark target-only and speculative decoding on Apple Silicon. Compare latency, throughput, unified-memory use, and quality from the same workload. Each result retains the exact input and generated token IDs, stop reasons, model identities, and measurement denominators.

## Install and run the offline fixture

Use a native arm64 Python 3.11–3.13 environment on an Apple Silicon Mac running macOS 15 or later. The Metal extra pins MLX 0.31.2, mlx-lm 0.31.3, and Transformers 5.23.0. The fixture is generated locally from a fixed seed and requires no model downloads. See the [fixture token baseline](src/mlx_spec_bench/data/fixture-baseline-v1.json).

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[metal]'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
run_dir="$(mlx-bench run examples/fixture.yaml --output-dir runs)"
mlx-bench validate records "$run_dir/results.jsonl"
mlx-bench report "$run_dir" --output "$run_dir/report.md"
```

The base package supports offline validation and reporting with Python 3.11–3.13 on Linux and macOS, including Intel Macs. Benchmark workers require Metal. A 48 GB Apple Silicon Mac is the reference host. The small fixture is designed to run on Apple Silicon CI and Macs with less memory.

### Python API

```python
from mlx_spec_bench import Config, ModelSpec, run

case = Config(target=ModelSpec("fixture"), batch_size=2)
result = run(case, output_dir="runs", warmups=1, repetitions=2)
assert result.status == "success", [row["error"] for row in result.records]
for row in result.records:
    request = result.detail(row)["requests"][0]
    print(row["run_id"], request["output_ids"])  # [5, 6, 2]
```

Use `load_config` for one case or `load_sweep` and `run_sweep` for a matrix. `CaseResult` provides the artifact directory, manifest, and typed `RunRecord` rows. `result.detail(row)` reads the raw request and quality evidence. The same worker path serves the API and CLI.

### Commands and failure status

| Command | Output |
|---|---|
| `mlx-bench run CONFIG --output-dir runs` | Prints a new sweep directory. Stores `manifest.json`, `results.jsonl`, per-run details, and worker logs. |
| `mlx-bench report DIRECTORY [--output FILE]` | Markdown from a copied sweep directory or a standalone JSONL file. |
| `mlx-bench validate {config,records} PATH` | Validates a v1 configuration or JSONL file offline. |
| `mlx-bench qualify DIRECTORY --output ARCHIVE.tar.gz` | Validates and archives the complete 48 GB reference sweep. |

Exit codes are 0 for success, 1 for failed trials or failed qualification, 2 for invalid input, and 130 for interruption. Each invocation has its own directory. Trials execute in separate processes. The runner replaces the JSONL journal atomically when a row completes and records missing attempts in the manifest after an interruption.

## Configuration and evidence

See the [v1 configuration and measurement contract](docs/contract.md), [example matrix](examples/fixture-sweep.yaml), and [synthetic JSONL example](tests/fixtures/golden.jsonl). Use the generated fixture or the pinned Qwen2.5 model pair below. Absolute positions and input IDs survive cache eviction. The scheduler uses ragged request cohorts with one model forward at a time. The reported batch size is the maximum size of a cohort.

| Model ID | Source and pinned revision | License |
|---|---|---|
| `fixture` | Local Qwen2 fixture generator v1, seed 20261120 | MIT |
| `qwen2.5-1.5b` | [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323) | Apache 2.0 |
| `qwen2.5-0.5b` | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/060db6499f32faf8b98477b0a26969ef7d8b9987) | Apache 2.0 |

The package, fixture generator, prompt suites, and held-out evaluation data use the [MIT license](LICENSE). Qwen weights download from Hugging Face and are converted locally. FP16 casts floating tensors to float16. Q8, Q6, and Q4 use mlx-lm affine quantization in groups of 64 after the cast. Tensors left in FP16 are recorded with the loaded recipe and artifact hashes. The second use of an artifact verifies its cached files before loading. Set `MLX_SPEC_BENCH_CACHE` to use a dedicated cache directory.

## Reproduce the 48 GB reference sweep

See the [reference qualification guide](docs/release.md) for the declared 48-case matrix, hardware and memory criteria, archival format, and release checks. See the [changelog](CHANGELOG.md) for candidate results and release status.
