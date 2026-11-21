import copy
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from mlx_spec_bench.io import json_text
from mlx_spec_bench.report import report
from mlx_spec_bench.schema import read_records, validate_record

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_golden_comparison_checks_pooled_arithmetic_and_absent_controls():
    expected = (FIXTURES / "golden-report.md").read_text()
    text = report(FIXTURES / "golden.jsonl")
    assert text == expected
    assert "| 3.000 | 10.000 | 100.000 | — |" in text
    assert "| 1/1 | 2.000 |" in text
    assert "| 0.500 |" in text
    assert "missing/incomparable target-only control" in text
    assert "acceptance undefined (zero proposals)" in text
    assert "timeout (fixture deadline expired)" in text
    assert "(FP16/unbounded)" in text


def test_installed_report_reads_copied_evidence(tmp_path):
    copied = shutil.copyfile(FIXTURES / "golden.jsonl", tmp_path / "copied.jsonl")
    command = [sys.executable, "-m", "mlx_spec_bench", "report", str(copied)]
    rendered = subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    assert rendered.stdout == (FIXTURES / "golden-report.md").read_text()


def test_success_and_failure_round_trip(tmp_path):
    records = read_records(FIXTURES / "golden.jsonl")
    copy_path = tmp_path / "roundtrip.jsonl"
    copy_path.write_text("".join(json_text(row) + "\n" for row in records))
    assert read_records(copy_path) == records


@pytest.mark.parametrize("mutation,diagnostic", [
    (lambda r: r["metrics"].pop("ttft_ms"), "ttft_ms"),
    (lambda r: r["metrics"].update(ttft_ms=float("inf")), "finite"),
    (lambda r: r["metrics"]["undefined"].pop("acceptance"), "undefined reason"),
    (lambda r: r.update(schema_version=2), "schema_version"),
])
def test_bad_evidence_is_rejected(mutation, diagnostic):
    record = copy.deepcopy(read_records(FIXTURES / "golden.jsonl")[0])
    mutation(record)
    with pytest.raises(ValueError, match=diagnostic):
        validate_record(record)
