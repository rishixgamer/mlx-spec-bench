"""Check both repetitions of the pinned Qwen2.5 conversion exercise."""

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from mlx_spec_bench.schema import read_manifest, read_records


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    root = parser.parse_args().directory
    rows = read_records(root / "results.jsonl")
    manifest = read_manifest(root / "manifest.json", rows)
    assert manifest["status"] == "success", [(r["run_id"], r["error"]) for r in rows if r["error"]]
    formats = defaultdict(list)
    tokenizers = set()
    revisions = set()
    for row in rows:
        assert row["environment"]["backend"].startswith("Device(gpu")
        assert row["metrics"]["tokens"]["output"] > 0
        model = row["identities"]["target"]
        assert model["id"] == "qwen2.5-0.5b"
        tokenizers.add(model["tokenizer_sha256"])
        revisions.add(model["revision"])
        assert bool(model["quantized_modules"]) == (model["weight_format"] != "fp16")
        formats[model["weight_format"]].append(model["artifact_sha256"])
    assert set(formats) == {"fp16", "q8", "q6", "q4"}
    assert all(len(hashes) == 2 and len(set(hashes)) == 1 for hashes in formats.values())
    assert len(tokenizers) == len(revisions) == 1
    print(f"Verified {len(formats)} pinned formats and {len(rows)} Metal repetitions.")


if __name__ == "__main__":
    main()
