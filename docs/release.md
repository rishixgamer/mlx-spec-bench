# v1.0 release qualification

The [reference manifest](../examples/reference-sweep.yaml) declares 48 cases and 96 measured repetitions. The target is Qwen2.5-1.5B at FP16, Q8, Q6, and Q4. The FP16 Qwen2.5-0.5B draft runs at K=1 and K=4 alongside K=0 controls. Each precision and K uses batch sizes 1 and 2, unbounded or 16,777,216-byte sliding KV, five prompts from `reference-v1`, 32 output tokens, seed zero, one warm-up, and two repetitions. The long prompts cross the sliding windows at both batch sizes. Source revisions and prompt/evaluation fingerprints are in the artifacts.

[Apple M5 Pro study](m5-speculative-decoding.md) reports matched decode rates
and token agreement for the reference matrix.

## Apple Silicon smoke gate

The `Apple Silicon Metal smoke` job runs on the arm64 `macos-15` GitHub-hosted runner. It installs the candidate wheel before starting the timed gate. The gate requires actual Metal execution with the local fixture, exact greedy output, a schema-valid report input, a rendered report, and deliberate corruption checks. `smoke-gate.json` records the elapsed time, runner, commit, and test count. The job must pass with a wall time under 45 seconds. Configure the `Apple Silicon Metal smoke` check as required in the default-branch rules to prevent merges when it fails.

The separate `Pinned Qwen2.5 weight formats` job downloads the 0.5B pinned source and verifies two genuine Metal repetitions in each of FP16, Q8, Q6, and Q4. It uses the [weight-format check manifest](../examples/pinned-weights.yaml) and retains the raw rows and report. It runs on pushes to `main` and manual CI dispatches.

## 48 GB run and archive

Run the following from the repository checkout on the 48 GB Mac. This local run only needs outbound access to PyPI and Hugging Face. Use native arm64 Python 3.11–3.13 on macOS 15+. Set `PYTHON=python3.12` if the default interpreter is outside that range.

```sh
git pull --ff-only
bash scripts/prepare-release.sh "$HOME/mlx-release-$(date +%Y%m%d-%H%M%S)" 'Describe the actual background load here'
```

Replace the host note with the conditions during measurement. Keep the Mac awake through the run (prefix the command with `caffeinate -i` if needed). Use a new output directory for each attempt. The script builds from the current Git commit in a temporary directory. It builds twice with a commit-derived timestamp and compares both copies of
the wheel and source archive byte for byte. It then installs the wheel and runs
the offline Metal smoke, test suite and Python quickstart before starting the
reference sweep. Completed trial records and worker diagnostics survive a failure under `runs/`. A successful run writes the wheel, source distribution, report, smoke result, reference archive and verified `SHA256SUMS` to `dist/`. To share results for release review, transfer that directory and identify the commit printed at startup.

Qualification reconciles every expected run ID, verifies exact greedy input/output and stopping against each matching precision/cache/batch control, and requires eviction in every bounded run. The footprint limit is 40,000,000,000 bytes. Worker failures, missing footprint or swap observations, or a sampled elevated pressure span of at least one second keep the gate open. Peak MLX allocator memory is reported separately. On success, the archive contains the resolved manifest, raw rows and details, worker logs, Markdown report, hardware gate summary, and SHA-256 checksums.

## Build and publish

Compare the commit in the smoke result and qualification archive with the commit that passed CI. On the matching commit, tag `v1.0.0` and publish the contents of the qualified `dist/` directory as GitHub Release assets. Use the tested wheel and source distribution from that directory. After downloading the published assets, run `shasum -a 256 -c SHA256SUMS`, install the downloaded wheel with the `metal` extra in a fresh environment, and reproduce the offline quickstart.
