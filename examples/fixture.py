"""Run the packaged offline fixture through the installed Python API."""

from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_spec_bench import Config, ModelSpec, run


def main() -> None:
    config = Config(target=ModelSpec("fixture"), batch_size=2)
    with TemporaryDirectory() as temporary:
        result = run(config, output_dir=Path(temporary), warmups=1, repetitions=2)
        assert result.status == "success", [row["error"] for row in result.records]
        for row in result.records:
            first = result.detail(row)["requests"][0]
            assert first["output_ids"] == [5, 6, 2]
            print(row["run_id"], row["environment"]["device"],
                  row["identities"]["target"]["artifact_sha256"], first["output_ids"])


if __name__ == "__main__":
    main()
