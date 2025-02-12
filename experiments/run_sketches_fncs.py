import numpy as np
import matplotlib.pyplot as plt
import subprocess
import bisect
import math
import uuid
import os
import time

# Define the sketch classes to compare

from spline_sketch_uniform import SplineSketchUniform

from helper_funcs import *

import csv

# FUNCTIONS FOR EXECUTING THE SKETCHES
######################################

def run_program_parse_stdout_and_outputFile(command, output_file, n):
    output = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
    # Split the output into lines and parse the integers
    output_lines = output.stdout.strip().split('\n')
    integers = [int(line.strip()) for line in output_lines]
    if len(integers) != 3: 
        raise Exception(f"EXPECTED 3 INTEGERS but output of running {command} is {output.stdout}")
    size, updateTime, queryTime = integers[0], integers[1], integers[2]

    # Read the output file and convert it to an array
    with open(output_file, 'r') as f_out:
        results = np.array([int(line.strip()) for line in f_out])
    os.remove(output_file)

    return results, size, updateTime, queryTime


def run_splinesketch_java(dataFile, queriesFile, n, sketch_size, print_info, tmpPath='./output_files/'): #
    output_file = tmpPath + "splinesketch_" + uuid.uuid4().hex
    
    command = f"java SplineSketchProgram {dataFile} {queriesFile} {sketch_size} {output_file}"
    return run_program_parse_stdout_and_outputFile(command, output_file, n)

def run_tdigest(dataFile, queriesFile, n, sketch_size, print_info, tmpPath='./output_files/'): #
    output_file = tmpPath + "t-digest_" + uuid.uuid4().hex

    # unfortunatley t-digest does not really have the desired size, so we need to adjust its compression parameter
    adjusted_size = 5 * sketch_size / 3
    adjusted_size = max(10, adjusted_size)
    
    command = f"java -cp .:t-digest-3.3.jar TDigestProgram {dataFile} {queriesFile} {adjusted_size} {output_file}"
    return run_program_parse_stdout_and_outputFile(command, output_file, n)

def run_MomentSketch(dataFile, queriesFile, n,k, print_info, tmpPath='./output_files/'): #
    output_file = tmpPath + "MomentSketch_" + uuid.uuid4().hex

    command = f"java -Xmx100g -cp .:msolver-1.0-SNAPSHOT.jar:quantile-bench-1.0-SNAPSHOT.jar:commons-math3-3.6.1.jar MomentSketchProgram {dataFile} {queriesFile} {k} {output_file}"
    return run_program_parse_stdout_and_outputFile(command, output_file, n)

def run_kll(dataFile, queriesFile, n,k, print_info, tmpPath='./output_files/'): #
    output_file = tmpPath + "KLL_" + uuid.uuid4().hex
    
    command = f"java -cp .:datasketches-java-6.0.0.jar:datasketches-memory-2.2.1.jar KLLProgram {dataFile} {queriesFile} {k} {output_file}"
    return run_program_parse_stdout_and_outputFile(command, output_file, n)

def run_splineSketchUniform(dataFile, queriesFile, n, sketch_size, print_info=""):
    data = load_floats_from_file(dataFile)
    queries = load_floats_from_file(queriesFile)
    start_time = time.perf_counter_ns()
    sketch = SplineSketchUniform(sketch_size, print_info)
    for x in data:
        sketch.update(x)
    sketch.consolidate()
    after_updates_time = time.perf_counter_ns()
    estimated_ranks = sketch.query(queries)
    after_queries_time = time.perf_counter_ns()
    sketch.print_avg_iters_stats()
    return estimated_ranks, 12*sketch_size + 8, after_updates_time - start_time, after_queries_time - after_updates_time # 12 bytes per bucket
