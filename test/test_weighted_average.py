#!/usr/bin/env python3
# last modified: 220929 09:52:14
"""Test weighted_average.py

:author: Shay Hill
:created: 2022-09-21
"""
# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false

from typing import cast

import numpy as np
import pytest

from cluster_colors import weighted_average


class TestApplyWeight:
    def test_convert_to_float(self):
        """Convert uint8 to float to avoid rolling over when multiplied"""
        color = cast(
            weighted_average._FPArray, np.array([255, 255, 255, 2], dtype=np.uint8)
        )
        np.testing.assert_array_equal( 
            weighted_average._apply_weight(color), (510, 510, 510)
        )


class TestGetWeightedAverage:
    def test_get_weighted_average(self):
        colors = np.array([[0, 0, 0, 50], [255, 255, 255, 1]])
        np.testing.assert_array_equal( 
            weighted_average.get_weighted_average(colors), (5, 5, 5, 51)
        )

    def test_value_error_if_no_alpha_channel(self):
        """Raise a ValueError if no alpha channel is present.
        Assert "Empty array" in error message.
        """
        colors = np.array([[0, 0, 0], [255, 255, 255]])
        with pytest.raises(ValueError) as excinfo:
            _ = weighted_average.get_weighted_average(colors)
        assert "No alpha channel" in str(excinfo.value)

    def test_value_error_if_input_array_is_empty(self):
        """Raise a ValueError if input array is empty.
        Assert "no alpha channel" in error message.
        """
        colors = np.array([])
        with np.testing.assert_raises(ValueError) as err:
            _ = weighted_average.get_weighted_average(colors)
        assert "Empty array" in str(err.exception)
