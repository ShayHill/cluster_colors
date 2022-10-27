from cluster_colors.stack_colors import stack_colors as py_stack_colors
from stack_colors import stack_colors as cy_stack_colors

import numpy as np
import time

TEST_IMAGE = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

def race_stacking():
    """call python and cython version of stack_colors. print a time comparison"""
    start = time.time()
    _ = py_stack_colors(TEST_IMAGE)
    py_time = time.time() - start

    start = time.time()
    cy_stack_colors(TEST_IMAGE)
    cy_time = time.time() - start

    print(f"python: {py_time:.4f}s, cython: {cy_time:.4f}s")

if __name__ == "__main__":
    race_stacking()

