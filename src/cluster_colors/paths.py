"""Paths to project binary files.

:author: Shay Hill
:created: 2022-10-25
"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent

BINARIES_DIR = _PROJECT_ROOT / "binaries"
TEST_DIR = _PROJECT_ROOT / "tests"

TEST_DIR.mkdir(parents=True, exist_ok=True)

BINARIES_DIR.mkdir(parents=True, exist_ok=True)
