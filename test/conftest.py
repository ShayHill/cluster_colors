import os
import sys

# keep Python, linters, and myself happy with import conventions
_project = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.append(_project)


def pytest_assertrepr_compare(config, op, left, right):
    """See full error diffs"""
    if op in ("==", "!="):
        return ["{0} {1} {2}".format(left, op, right)]

