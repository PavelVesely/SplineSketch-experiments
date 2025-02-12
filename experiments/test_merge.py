import numpy as np
import matplotlib.pyplot as plt
import subprocess
import bisect
import math
import uuid
import os
from joblib import Parallel, delayed
import random

from IID_generators import *
from run_experiments_IID import compute_true_ranks, one_experiment
from spline_sketch_uniform import SplineSketchUniform

NUM_PARALLEL_JOBS = 30 # number of jobs run in parallel

import csv

########################################
# SO FAR, IMPLEMENTED FOR SPLINE SKETCH IN PYTHON ONLY

def split_list(data, num_parts):
    # Calculate the length of each part
    avg = len(data) / float(num_parts)
    out = []
    last = 0.0

    while last < len(data):
        # Append the next chunk to the result list
        out.append(data[int(last):int(last + avg)])
        last += avg

    return out

def run_mergeTest_splineSketchUniform(data, num_parts, queries, sketch_size, print_info=""):
    sketches = []
    inputs = split_list(data, num_parts)
    # create num_parts sketches
    for i in range(num_parts):
        sketch = SplineSketchUniform(sketch_size, print_info)
        for x in inputs[i]:
            sketch.update(x)
        sketches.append(sketch)
    #sketch.print_avg_iters_stats()

    # random merging strategy -- take two random sketches in each step and merge them
    while len(sketches) > 1:
        i, j = random.sample(range(len(sketches)), 2)
        #print(f"merging sketches {i} and {j}")
        sketches[i] = SplineSketchUniform.merge(sketches[i], sketches[j])
        sketches.pop(j) # TODO: speed up

    sketch = sketches[0]
    assert sketch.n == len(data)


    estimated_ranks = sketch.query(queries)
    return estimated_ranks, 2*sketch_size # two numbers per bucket


# HERE IS THE MAIN PROGRAM
##########################
def one_experiment(data, logN, num_parts, queries, true_values, sketch_size, run_function, sketch_name, data_name):
    info = f"{sketch_name}, logN {logN}, size {sketch_size}, dataset {data_name}"
    estimated_ranks, actual_sketch_size = run_function(data, num_parts, queries, sketch_size, info)

    # Calculate the errors
    if len(estimated_ranks) == len(true_values):
        errors = np.abs(np.array(estimated_ranks) - np.array(true_values))
    else: # in case of an error (e.g. from MomentSketch), inf is the estimate
        errors = np.array([float('inf') for _ in true_values])
        print(f"!!!!! {sketch_name}, logN {logN}, size {actual_sketch_size}, dataset {data_name} -- probably FAILED")

    average_error = np.mean(errors)
    max_error = max(errors)

    print(f"{sketch_name}, logN {logN}, size {actual_sketch_size}, dataset {data_name}. Average error {average_error}, max error {max_error}")


    return average_error, max_error, actual_sketch_size, true_values, errors

if __name__ == "__main__":
    data_sources = [(distinct_values_42, "distinct_values_42"),
                    (distinct_values_5, "distinct_values_5"),
	 	            (distinct_values_150, "distinct_values_150"),
                    (uniform, "uniform"), 
                    (lognormal, "lognormal"), (pareto, "pareto"), 
                    (loguniform, "loguniform"), (normal, "normal"),
                    (signed_lognormal, "signed_lognormal"),
                    (signed_loguniform, "signed_loguniform")]
    sketch_sizes = [25,50,75,100,125,150,175,200]
    MomentSketch_ks = [5,7,9,11,13,15,17,19]
    KLL_ks = [8,24,42,54,66,78,90,102] #FIXME: KLL size is also influenced by log(N) in the DataSketches implementation
    input_sizes_log10 = [6,7] 
    num_parts_array = [100, 1000, 10000]
    sketch_functions = [(run_mergeTest_splineSketchUniform, "spline sketch uniform")]

    print(f"data sources = {data_sources}")
    print(f"input_sizes_log10 = {input_sizes_log10}")
    print(f"sketch_sizes = {sketch_sizes}")
    print(f"sketch_functions = {sketch_functions}")

    jobsData = []
    jobs = [] # jobs for sketches


    def gen_one_dataset(data_func, data_name, logN):
        data = data_func(np.power(10, logN))
        #print(f"loaded data for {data_name} for logN={logN}")
        queries = []
        if len(data) < 10000: # changed from if len(queries) < 10000:
            queries = sorted(data)
        else:
            queries = sorted(data)[::int(len(data)/10000)]

        true_values = compute_true_ranks(data, queries)
        print(f"loaded data and computed true ranks for {data_name} for logN={logN}")
        jobs = [] # jobs for sketches
        for num_parts in num_parts_array:
            for sketch_size in sketch_sizes:    
                for run_function, sketch_name in sketch_functions:
                    size = sketch_size
                    if sketch_name == "MomentSketch":
                        size = MomentSketch_ks[sketch_sizes.index(size)] # translating sketch_size to MomentSketch k
                    elif sketch_name == "KLL sketch":
                        size = KLL_ks[sketch_sizes.index(size)] # translating sketch_size to KLL k

                    jobs.append(delayed(one_experiment)(data, logN, num_parts, queries, true_values, size, run_function, sketch_name, data_name))
        
        #return data, queries, true_values # no need to return
        return jobs

    for data_func, data_name in data_sources:
        for logN in input_sizes_log10:
            jobsData.append(delayed(gen_one_dataset)(data_func, data_name, logN))
            
    
    resJobs = Parallel(n_jobs=NUM_PARALLEL_JOBS)(jobsData)
    jobs = []
    [ jobs.extend(subjobs) for subjobs in resJobs] 
    #print(jobs)
    print(f"============== DATA LOADED ===================")
    results = Parallel(n_jobs=NUM_PARALLEL_JOBS)(jobs)

    idx = 0
    for data_func, data_name in data_sources:
        for logN in input_sizes_log10:
            for num_parts in num_parts_array:
                average_errors = {fn.__name__: [] for fn,_ in sketch_functions}
                max_errors = {fn.__name__: [] for fn,_ in sketch_functions}
                actual_sketch_sizes = {fn.__name__: [] for fn,_ in sketch_functions}

                for sketch_size in sketch_sizes:
                    fig_re, ax_re = plt.subplots()
                    ax_re.set_xlabel('true rank')
                    ax_re.set_ylabel('error')
                    #ax_re.set_yscale('log') # using default linear scale here as it doesn't look very good with log scale
                    fig_re.tight_layout()

                    for run_function, sketch_name in sketch_functions:
                        average_error, max_error, actual_sketch_size, true_values, errors = results[idx]
                        idx += 1
                        if average_error == float('inf'):
                            print(f"skipping plot for: {data_name}, logN={logN}, sketch={sketch_name}, size={sketch_size}")
                            continue
                        average_errors[run_function.__name__].append(average_error)
                        max_errors[run_function.__name__].append(max_error)
                        actual_sketch_sizes[run_function.__name__].append(actual_sketch_size)

                        ax_re.plot(true_values, errors, label=f"{sketch_name}, size {actual_sketch_size}")


                    #fig_re.savefig(f"plots/{data_name}_logN={logN}_size={sketch_size}_nolegend.pdf", format='pdf')
                    fig_re.legend()
                    #ax_re.set_title(f"{data_name}_logN={logN}_size={sketch_size}") #TODO: to be removed (now for info which plot is which)
                    ax_re.grid(True)
                    fig_re.savefig(f"plots/testMerge-{data_name}_logN{logN}_parts{num_parts}_size{sketch_size}_legend.pdf", format='pdf')

                fig_avg, ax_avg = plt.subplots()
                fig_max, ax_max = plt.subplots()
                for run_function, sketch_name in sketch_functions:
                    ax_avg.plot(actual_sketch_sizes[run_function.__name__], average_errors[run_function.__name__], label=f"{sketch_name}")
                    ax_max.plot(actual_sketch_sizes[run_function.__name__], max_errors[run_function.__name__], label=f"{sketch_name}")

                ax_avg.set_xlabel('sketch size')
                ax_avg.set_ylabel('average error')
                ax_avg.set_yscale('log')
                ax_avg.grid(True)
                fig_avg.tight_layout()
                #fig_avg.savefig(f"plots/{data_name}_logN={logN}_average_nolegend.pdf", format='pdf')
                ax_avg.legend()
                #ax_avg.set_title(f"{data_name}_logN={logN}_average")
                fig_avg.savefig(f"plots/testMerge-{data_name}_logN{logN}_parts{num_parts}_average_legend.pdf", format='pdf')

                ax_max.set_xlabel('sketch size')
                ax_max.set_ylabel('max error')
                ax_max.set_yscale('log')
                ax_max.grid(True)
                fig_max.tight_layout()
                #fig_max.savefig(f"plots/{data_name}_logN={logN}_max_nolegend.pdf", format='pdf')
                ax_max.legend()
                #ax_max.set_title(f"{data_name}_logN={logN}_max")
                fig_max.savefig(f"plots/testMerge-{data_name}_logN{logN}_parts{num_parts}_max_legend.pdf", format='pdf')
