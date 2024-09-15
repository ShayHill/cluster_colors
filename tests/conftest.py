"""Show full error diffs in pytest

:author: Shay Hill
:created: 2024-09-15
"""

from typing import Any


def pytest_assertrepr_compare(config: Any, op: str, left: str, right: str):
    """See full error diffs"""
    if op in ("==", "!="):
        return [f"{left} {op} {right}"]
