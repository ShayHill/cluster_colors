#!/usr/bin/env python3
# last modified: 221107 18:05:03
"""Paths to project binary files.

:author: Shay Hill
:created: 2022-10-25
"""

from pathlib import Path

_PROJECT_DIR = Path(__file__).parent.parent

BINARIES_DIR = _PROJECT_DIR / "binaries"
TEST_DIR = _PROJECT_DIR / "test"
PICKLE_DIR = BINARIES_DIR / "pickle"

TEST_DIR.mkdir(parents=True, exist_ok=True)
PICKLE_DIR.mkdir(parents=True, exist_ok=True)
