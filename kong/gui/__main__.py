"""Entry point for `python -m kong.gui`.

Handy when Kong is not installed on PATH: run it from the repository root
without going through the click CLI.
"""

from __future__ import annotations

import sys

from kong.gui.app import launch

if __name__ == "__main__":
    launch(initial_binary=sys.argv[1] if len(sys.argv) > 1 else "")
