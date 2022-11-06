#!/usr/bin/env python3
# last modified: 221101 14:43:41
"""Paths to project binary files.

:author: Shay Hill
:created: 2022-10-25
"""

from pathlib import Path

_PROJECT_DIR = Path(__file__).parent.parent
TEST_DIR = _PROJECT_DIR / "test"
PICKLE_DIR = _PROJECT_DIR / "pickle"
