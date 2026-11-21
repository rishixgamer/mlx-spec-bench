<p align="center">
  <img src="docs/assets/readme-banner.webp" alt="" width="1440">
</p>

<h1 align="center">mlx-spec-bench</h1>

<p align="center">
  <strong>Reproducible LLM decoding benchmarks for Apple Silicon</strong><br>
  Compare latency, throughput, unified memory, and quality across decoding strategies, weight formats, and KV cache policies.
</p>

<p align="center">
  <a href="https://github.com/rishixgamer/mlx-spec-bench/actions/workflows/ci.yml?query=branch%3Amain"><img src="https://github.com/rishixgamer/mlx-spec-bench/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI status on main"></a>
  &nbsp;macOS 15+ · arm64 Python 3.11–3.13 · <a href="LICENSE">MIT</a>
</p>

<p align="center">
  <a href="#install-and-run-the-offline-fixture"><strong>Quickstart</strong></a>
  &nbsp;·&nbsp;
  <a href="#benchmark-results">Benchmark results</a>
  &nbsp;·&nbsp;
  <a href="#python-api">Python API</a>
  &nbsp;·&nbsp;
  <a href="#reproduce-the-48-gb-reference-sweep">Reference sweep</a>
</p>

## Install and run the offline fixture

Use native arm64 Python 3.11–3.13 on an Apple Silicon Mac running macOS 15 or later. The fixture is generated locally from a fixed seed and requires no model downloads after installation. [Fixture token baseline](src/mlx_spec_bench/data/fixture-baseline-v1.json)

```sh
git clone https://github.com/rishixgamer/mlx-spec-bench.git
cd mlx-spec-bench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[metal]'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

run_dir="$(mlx-bench run examples/fixture.yaml --output-dir runs)"
mlx-bench validate records "$run_dir/results.jsonl"
mlx-bench report "$run_dir" --output "$run_dir/report.md"
```

The Metal extra pins MLX 0.31.2, mlx-lm 0.31.3, and Transformers 5.23.0. The base package supports offline validation and reporting with Python 3.11–3.13 on Linux and macOS, including Intel Macs.

## Benchmark results

These target-only Qwen2.5‑0.5B trials ran on a GitHub-hosted Apple virtual GPU with 7 GiB RAM. Each format used the same five prompts, a batch size of 1, a four-token cap, and an unbounded KV cache. Values combine two repetitions per format. Peak GB is sampled process footprint. [CI run and raw artifacts](https://github.com/rishixgamer/mlx-spec-bench/actions/runs/44931301868#artifacts)

| Weights | First token (ms) | Decode (tok/s) | Peak footprint (GB) | Perplexity |
| :--- | ---: | ---: | ---: | ---: |
| FP16 | 571.3 | 6.37 | 1.598 | **38.33** |
| Q8 | **363.9** | **8.76** | 1.313 | 40.23 |
| Q6 | 412.9 | 8.12 | 1.282 | 54.23 |
| Q4 | 413.5 | 8.15 | **1.159** | 161.39 |

Q8 had the highest decode throughput in this CI run. Q4 used the least memory, while FP16 had the lowest perplexity. The separate offline Metal fixture passed in **7.30 seconds** against a 45-second CI limit. [Metal smoke and weight-format evidence](https://github.com/rishixgamer/mlx-spec-bench/actions/runs/44931301868)

## Configuration and evidence

- **Define a sweep.** Set the target and draft models, weight formats, batch size, and KV cache budget in YAML. [Example matrix](examples/fixture-sweep.yaml)
- **Verify greedy decoding.** Compare the full generated token IDs and stop reasons against a target-only control at the same precision and cache policy. Absolute positions and input IDs survive cache eviction. [Decoding and cache contract](docs/contract.md#greedy-verification-and-cache-accounting)
- **Inspect the run.** Each result retains exact input and generated token IDs, stop reasons, model identities, and measurement denominators. Regenerate a Markdown comparison offline. [Sample report](tests/fixtures/golden-report.md) · [Example JSONL](tests/fixtures/golden.jsonl)

The scheduler uses ragged request cohorts with one model forward at a time. The reported batch size is the maximum size of a cohort. [Measurement contract](docs/contract.md)

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

<details>
<summary><strong>Pinned models and weight conversion</strong></summary>

| Model ID | Source and pinned revision | License |
|---|---|---|
| `fixture` | Local Qwen2 fixture generator v1, seed 20261120 | MIT |
| `qwen2.5-1.5b` | [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323) | Apache 2.0 |
| `qwen2.5-0.5b` | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/060db6499f32faf8b98477b0a26969ef7d8b9987) | Apache 2.0 |

Qwen weights download from Hugging Face and are converted locally. FP16 casts floating tensors to float16. Q8, Q6, and Q4 use mlx-lm affine quantization in groups of 64 after the cast. Tensors left in FP16 are recorded with the loaded recipe and artifact hashes. The second use of an artifact verifies its cached files before loading. Set `MLX_SPEC_BENCH_CACHE` to use a dedicated cache directory.

</details>

## Reproduce the 48 GB reference sweep

The release matrix declares **48 cases and 96 measured repetitions** across precision, draft length, batch size, and cache policy. The small fixture runs on Apple Silicon Macs with less memory. The v1.0 release candidate awaits qualification on a 48 GB Apple Silicon Mac. [Reference qualification guide](docs/release.md) · [Changelog](CHANGELOG.md)

The package, fixture generator, prompt suites, and held-out evaluation data are MIT-licensed. [License](LICENSE)
