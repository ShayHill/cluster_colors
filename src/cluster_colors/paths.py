#!/usr/bin/env python3
# last modified: 230117 15:09:40
"""Paths to project binary files.

:author: Shay Hill
:created: 2022-10-25
"""

from pathlib import Path

_PROJECT_DIR = Path(__file__).parent.parent.parent

BINARIES_DIR = _PROJECT_DIR / "binaries"
TEST_DIR = _PROJECT_DIR / "tests"
PICKLE_DIR = BINARIES_DIR / "pickle"

TEST_DIR.mkdir(parents=True, exist_ok=True)
PICKLE_DIR.mkdir(parents=True, exist_ok=True)
