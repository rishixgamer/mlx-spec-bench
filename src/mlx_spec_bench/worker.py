"""Private subprocess entry point for a single benchmark trial."""

import sys
from pathlib import Path

from .config import Config
from .io import read_json, write_json
from .trial import execute


def main() -> None:
    request = read_json(Path(sys.argv[1]))
    result = execute(Config.from_dict(request["config"]), request["warmups"])
    write_json(Path(sys.argv[2]), result)


if __name__ == "__main__":
    main()
