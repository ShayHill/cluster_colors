"""Try to find an address any potential non-deterministic cases irt split floats.

:author: Shay Hill
:created: 2024-09-12
"""

# pyright: reportPrivateUsage=false

import pytest

from cluster_colors.cluster_cluster import _split_floats


class TestSplitFloats:
    def test_empty(self):
        """Raise ValueError if no floats are passed."""
        with pytest.raises(ValueError):
            _ = _split_floats([])

    def test_single(self):
        """Raise ValueError if only one float is passed."""
        with pytest.raises(ValueError):
            _ = _split_floats([1.0])

    def test_identical(self):
        """Return 0 if all floats are identical."""
        assert _split_floats([1.0, 1.0, 1.0]) == 0

    def test_explicit(self):
        """Return the expected split."""
        assert _split_floats([1.0, 2.0, 3.0]) == 1

    def test_explicit_2(self):
        """Return the expected split."""
        assert _split_floats([1.0, 2.0, 2.0, 3, 3, 3, 3, 3]) == 3

    def test_explicit_3(self):
        """Return the expected split."""
        for i in range(76):
            assert _split_floats([-i, *range(100)]) == 50
        for i in range(76, 124):
            assert _split_floats([-i, *range(100)]) == 49
        for i in range(124, 169):
            assert _split_floats([-i, *range(100)]) == 48
        for i in range(169, 213):
            assert _split_floats([-i, *range(100)]) == 47
        for i in range(213, 254):
            assert _split_floats([-i, *range(100)]) == 46
