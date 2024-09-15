"""Test the vector_stacker module.

:author: Shay Hill
:created: 2022-10-18
"""

# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownParameterType=false

import numpy as np

from cluster_colors import vector_stacker as sv


class TestAddWeightAxis:
    """Test the add_weight_axis method."""

    def test_add_weight_axis(self):
        """Test the add_weight_axis method."""
        np.testing.assert_array_equal(
            sv.add_weight_axis(np.array([[1, 2, 3], [4, 5, 6]])),
            np.array([[1, 2, 3, 255], [4, 5, 6, 255]]),
        )

    def test_optional_weight_parameter(self):
        """Test the optional weight parameter."""
        np.testing.assert_array_equal(
            sv.add_weight_axis(np.array([[1, 2, 3], [4, 5, 6]]), 128),
            np.array([[1, 2, 3, 128], [4, 5, 6, 128]]),
        )

    def test_add_weight_axis_does_not_alter_input(self):
        """Test the add_weight_axis method does not alter input."""
        input_array = np.array([[1, 2, 3], [4, 5, 6]])
        _ = sv.add_weight_axis(input_array)
        np.testing.assert_array_equal(input_array, np.array([[1, 2, 3], [4, 5, 6]]))

    def test_uint8_input_returns_floats_in_output(self):
        """Test the add_weight_axis method returns floats for uint8 input."""
        output = sv.add_weight_axis(
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8), weight=np.uint8(128)  # type: ignore
        )
        assert output.dtype == np.float64


class TestStackVectors:
    def test_stack_unique_vectors(self):
        """Test the stack_unique_vectors method."""
        np.testing.assert_array_equal(
            sv.stack_vectors(np.array([[1, 2, 3, 55], [4, 5, 6, 5], [1, 2, 3, 45]])),
            np.array([[1, 2, 3, 100], [4, 5, 6, 5]]),
        )

    def test_uint8_input_returns_floats_in_output(self):
        """Test the stack_unique_vectors method returns floats for uint8 input."""
        output = sv.stack_vectors(
            np.array([[1, 2, 3, 55], [4, 5, 6, 5], [1, 2, 3, 45]], dtype=np.uint8)
        )
        assert output.dtype == np.float64

    def test_stack_vectors_does_not_alter_input(self):
        """Test the stack_unique_vectors method does not alter input."""
        input_array = np.array([[1, 2, 3, 55], [4, 5, 6, 5], [1, 2, 3, 45]])
        _ = sv.stack_vectors(input_array)
        np.testing.assert_array_equal(
            input_array, np.array([[1, 2, 3, 55], [4, 5, 6, 5], [1, 2, 3, 45]])
        )

    def test_optional_weight_parameter(self):
        """Test the optional weight parameter."""
        np.testing.assert_array_equal(
            sv.stack_vectors(np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]]), weight=128),
            np.array([[1, 2, 3, 256], [4, 5, 6, 128]]),
        )


class TestStackColors:
    def test_stack_vectors(self):
        """Test the stack_vectors method on 4-axis vectors."""
        np.testing.assert_array_equal(
            sv.stack_vectors(np.array([[1, 2, 3, 55], [4, 5, 6, 5], [1, 2, 3, 45]])),
            np.array([[1, 2, 3, 100], [4, 5, 6, 5]]),
        )

    def test_stack_single_vectors(self):
        """Test the stack_vectors method on 1-axis vectors."""
        np.testing.assert_array_equal(
            sv.stack_vectors(np.array([[1, 1], [4, 1], [1, 1]])),
            np.array([[1, 2], [4, 1]]),
        )
