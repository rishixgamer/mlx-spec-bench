"""Installed run, validate, report, and reference qualification commands."""

import argparse
from pathlib import Path
import signal
import sys

import yaml

from .config import load_sweep
from .io import atomic_write
from .report import report
from .runner import run_sweep
from .schema import read_records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-bench",
        description="Measure MLX speculative decoding on Apple Silicon and compare saved runs offline.",
        epilog="Exit 0 on success, 1 for failed trials or a failed qualification, 2 for input errors, "
               "and 130 for an interrupted sweep.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="execute a v1 YAML case or sweep")
    run.add_argument("config", type=Path, help="path to the YAML case or sweep")
    run.add_argument("--output-dir", type=Path, default=Path("runs"),
                     help="parent for a new sweep-ID directory (default: runs/)")
    run.add_argument("--host-notes", help="background load and host notes stored in the manifest")
    run.add_argument("--source-commit", help="source commit to associate with the recorded manifest")

    report_parser = sub.add_parser("report", help="render a Markdown report from saved evidence")
    report_parser.add_argument("artifact", type=Path, help="sweep directory or a results.jsonl file")
    report_parser.add_argument("--output", type=Path, help="write the report to this path (default: stdout)")

    validate = sub.add_parser("validate", help="validate a v1 YAML config/sweep or JSONL file without Metal")
    validate.add_argument("kind", choices=("config", "records"))
    validate.add_argument("path", type=Path)

    qualify = sub.add_parser("qualify", help="verify and archive the 48 GB reference sweep")
    qualify.add_argument("artifact", type=Path, help="finished sweep directory")
    qualify.add_argument("--output", type=Path, required=True, help="new .tar.gz archive path")
    return parser


def _progress(row, _directory):
    if row["error"]:
        print(f"{row['run_id']} {row['error']['kind']}: {row['error']['message']}", file=sys.stderr, flush=True)


def _interrupted(_signum, _frame):
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            sweep = load_sweep(args.config)
            previous = signal.signal(signal.SIGTERM, _interrupted)
            try:
                result = run_sweep(
                    sweep, output_dir=args.output_dir, host_notes=args.host_notes,
                    source_commit=args.source_commit, on_record=_progress,
                )
            finally:
                signal.signal(signal.SIGTERM, previous)
            print(result.directory)
            return {"success": 0, "failure": 1, "interrupted": 130}[result.status]
        if args.command == "report":
            content = report(args.artifact)
            if args.output:
                atomic_write(args.output, content)
            else:
                print(content, end="")
            return 0
        if args.command == "validate":
            count = len(load_sweep(args.path).cases) if args.kind == "config" else len(read_records(args.path))
            print(f"Validated {count} {'case(s)' if args.kind == 'config' else 'record(s)'}.")
            return 0
        from .qualification import qualify

        checks = qualify(args.artifact, args.output)
        for problem in checks:
            print(problem, file=sys.stderr)
        if not checks:
            print(args.output.resolve())
        return 1 if checks else 0
    except (ValueError, OSError, KeyError, yaml.YAMLError) as error:
        print(f"mlx-bench: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
