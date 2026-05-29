"""Allow `python -m jarvis_cli <subcommand>` as an alternative to console scripts."""
from __future__ import annotations

import sys

from .install import main as install_main


def main() -> int:
    return install_main()


if __name__ == "__main__":
    sys.exit(main())
