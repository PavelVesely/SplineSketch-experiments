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
    data_sorted = sorted(data)
    ranks = []
    for query in queries:
        idx = bisect.bisect_right(data_sorted, query)
        rank = idx
        ranks.append(rank)
    return ranks

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
    floats_list = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Convert the line to a float and append to the list
                floats_list.append(float(line.strip()))
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except ValueError:
        print("There was an error converting a line to float.")
    return floats_list

def write_floats_to_file(floats_list, file_path):
    try:
        with open(file_path, 'w') as file:
            for number in floats_list:
                # Write each float to a new line in the file
                file.write(f"{number}\n")
    except IOError:
        print(f"An error occurred while writing to the file {file_path}.")