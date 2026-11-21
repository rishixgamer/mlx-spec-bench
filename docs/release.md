# v1.0 release qualification

The [reference manifest](../examples/reference-sweep.yaml) declares 48 cases and 96 measured repetitions. The target is Qwen2.5-1.5B at FP16, Q8, Q6, and Q4. The FP16 Qwen2.5-0.5B draft runs at K=1 and K=4 alongside K=0 controls. Each precision and K uses batch sizes 1 and 2, unbounded or 16,777,216-byte sliding KV, five prompts from `reference-v1`, 32 output tokens, seed zero, one warm-up, and two repetitions. The long prompts cross the sliding windows at both batch sizes. Source revisions and prompt/evaluation fingerprints are in the artifacts.

## Apple Silicon smoke gate

The `Apple Silicon Metal smoke` job runs on the arm64 `macos-15` GitHub-hosted runner. It installs the candidate wheel before starting the timed gate. The gate requires actual Metal execution with the local fixture, exact greedy output, a schema-valid report input, a rendered report, and deliberate corruption checks. `smoke-gate.json` records the elapsed time, runner, commit, and test count. The job must pass with a wall time under 45 seconds. Configure the `Apple Silicon Metal smoke` check as required in the default-branch rules to prevent merges when it fails.

The separate `Pinned Qwen2.5 weight formats` job downloads the 0.5B pinned source and verifies two genuine Metal repetitions in each of FP16, Q8, Q6, and Q4. It uses the [weight-format check manifest](../examples/pinned-weights.yaml) and retains the raw rows and report. It runs on candidate-branch pushes and manual CI dispatches.

## 48 GB run and archive

On a 48 GB Apple Silicon Mac, check out the candidate and use a fresh arm64 virtual environment. Install the candidate wheel with the `metal` extra and allow Hugging Face access for the pinned model downloads. Record the actual idle and background load in `--host-notes`.

```sh
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
run_dir="$(mlx-bench run examples/reference-sweep.yaml \
  --source-commit "$(git rev-parse HEAD)" \
  --host-notes 'Record the actual background load here' --output-dir runs)"
mlx-bench report "$run_dir" --output "$run_dir/report.md"
mlx-bench qualify "$run_dir" --output reference-v1.tar.gz
```

Qualification reconciles every expected run ID, verifies exact greedy input/output and stopping against each matching precision/cache/batch control, and requires eviction in every bounded run. The footprint limit is 40,000,000,000 bytes. Worker failures, missing footprint or swap observations, or a sampled elevated pressure span of at least one second keep the gate open. Peak MLX allocator memory is reported separately. On success, the archive contains the resolved manifest, raw rows and details, worker logs, Markdown report, hardware gate summary, and SHA-256 checksums.

## Build and publish

Build the wheel and source distribution with `python -m build`. Check the archive and candidate point to the same commit that passed Apple Silicon CI. In a fresh supported Mac environment install the wheel with the `metal` extra and run both [quickstarts](../README.md) with downloads disabled. Generate `SHA256SUMS` for the wheel, source archive, and reference archive. Tag that commit `v1.0.0` and publish the distributions, checksums, report, and raw reference evidence as release assets. Verify the asset hashes after download, install the published wheel in a new environment, and reproduce the schema-valid smoke result from it.
