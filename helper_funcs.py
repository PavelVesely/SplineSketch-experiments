import numpy as np
import bisect
import math
import uuid
import os
from joblib import Parallel, delayed


def compute_true_ranks(data, queries):
    """
    Compute the number of items in the data that are less than or equal to each query.

    Args:
        data (list of float): The data to query.
        queries (list of float): The queries to compute fractions for.

    Returns:
        list of float: A list of fractions, where each fraction is the proportion of items in the data that are less than or equal to the corresponding query.
    """
    data_sorted = np.sort(np.asarray(data, dtype=np.float64))
    return np.searchsorted(data_sorted, queries, side="right").tolist()


def calculate_aspect_ratio(data, k = 0):
    n = len(data)
    data = list(set(data)) # distinct items
    data.sort()
    # Ensure the list has at least two distinct elements
    if len(data) < 2:
        raise ValueError("The list must contain at least two distinct elements.")

    # Calculate the max and min of the data
    max_value = max(data)
    min_value = min(data)
    
    d = 1
    if k > 0:
        d = min(max(1, int(0.01 * n / k)), len(data) - 1)
    # Calculate the differences between each pair of distinct items
    differences = [data[i+d] - data[i] for i in range(len(data)-d)]
    
    # Find the minimum difference between distinct items
    min_difference = min(differences)
    
    # Ensure there is a positive minimum difference
    if min_difference == 0:
        raise ValueError("The list must contain distinct elements with non-zero differences.")

    # Calculate the aspect ratio
    aspect_ratio = (max_value - min_value) / min_difference

    return aspect_ratio

def load_floats_from_file(file_path):
    """Load values, propagating missing files and invalid input to the caller."""
    with open(file_path) as stream:
        return [float(line.strip()) for line in stream]


def write_floats_to_file(floats_list, file_path):
    with open(file_path, "w") as stream:
        for number in floats_list:
            stream.write(f"{number}\n")
