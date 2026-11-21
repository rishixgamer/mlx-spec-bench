"""Durable file I/O and canonical evidence fingerprints."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def json_text(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def fingerprint(value) -> str:
    return hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()


def file_fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_write(path: Path, text: str) -> None:
    """Publish all bytes together and make the rename durable."""
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    atomic_write(path, json_text(value) + "\n")


def reject_constant(value: str):
    raise ValueError(f"non-finite JSON number {value}")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
