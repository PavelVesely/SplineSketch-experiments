import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

import os
from joblib import Parallel, delayed

from IID_generators import *
from run_sketches_fncs import *
from helper_funcs import *

import csv

# in case of TKAgg, the following works: https://stackoverflow.com/questions/55811545/importerror-cannot-load-backend-tkagg-which-requires-the-tk-interactive-fra

########################## PARAMETERS OF EXPERIMENTS ######################3

NUM_PARALLEL_JOBS = 8 # number of jobs run in parallel

data_sources = [(normal, "normal")]
sketch_sizes = [100]
MomentSketch_ks = [15]
KLL_ks = [8] # smallest meaningful k for KLL

data_sizes = [10**8] 
MAX_LOG_SIZE = 16
NUM_QUERIES_LIST = [int(1000 * 2**i) for i in range(0, MAX_LOG_SIZE)] 
sketch_functions = [#(run_splineSketchUniform, "SplineSketch(Py)"),
                    (run_splinesketch_java, "SplineSketch"),
                    (run_kll, "KLL sketch"), 
                    (run_MomentSketch, "MomentSketch"),
                    (run_tdigest, "t-digest")]

# HERE IS THE MAIN PROGRAM
##########################

def one_experiment(dataFile, N, queriesFile, true_values, sketch_size, run_function, sketch_name, data_name):
    info = f"{sketch_name}_N{N}_size{sketch_size}_dataset_{data_name}"
    estimated_ranks, actual_sketch_size, query_time_ns, query_time_ns = run_function(dataFile, queriesFile, N, sketch_size, info)
    time_per_query_mus = query_time_ns / (1000.0 * N) 
    time_per_query_mus = query_time_ns / (1000.0 * len(true_values))

    # Calculate the errors
    if len(estimated_ranks) == len(true_values):
        errors = np.abs(np.array(estimated_ranks) - np.array(true_values))
    else: # in case of an error (e.g. from MomentSketch), inf is the estimate
        errors = np.array([float('inf') for _ in true_values])
        print(f"!!!!! {sketch_name}, N {N}, size {actual_sketch_size}, dataset {data_name} -- probably FAILED")

    average_error = np.mean(errors)
    max_error = max(errors)

    print(f"{sketch_name}, N {N}, size {actual_sketch_size}, dataset {data_name}. Average error {average_error}, max error {max_error}")


    return average_error, max_error, actual_sketch_size, true_values, errors, time_per_query_mus, time_per_query_mus

if __name__ == "__main__":
    # create necessary dirs.
    output_dir='./output_files/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    datasets_dir='./datasets/'
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
    plots_dir='./plots/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    print(f"data sources = {data_sources}")
    print(f"data_sizes = {data_sizes}")
    print(f"sketch_sizes = {sketch_sizes}")
    print(f"sketch_functions = {sketch_functions}")
    print(f"num queries = {NUM_QUERIES_LIST}")
    
    jobsData = []
    jobs = [] # jobs for sketches


    def gen_one_dataset(data_func, data_name, N):
        dataFile = datasets_dir+f"{data_name}_N{N}.data.txt"

        if not os.path.exists(dataFile):
            data = data_func(N)
            defk = 100


            write_floats_to_file(data, dataFile)

            print(f"created dataset {data_name} for N={N} in {dataFile})") # aspect ratio = {alpha}, refined aspect ratio = {alpha2} (k={defk})")
        else:
            data = load_floats_from_file(dataFile)
            print(f"loaded dataset {data_name} for N={N} in {dataFile})")
        
        queries_files = {}
        true_values_lists = {}
        for num_qs in NUM_QUERIES_LIST:
            queriesFile = datasets_dir+f"{data_name}_N{N}.queries.numQS{num_qs}.txt"
            queries = []
            if len(data) < num_qs:
                queries = sorted(data)
            else:
                queries = sorted(data)[::int(len(data)/num_qs)]
            write_floats_to_file(queries, queriesFile)
            true_values = compute_true_ranks(data, queries)
            queries_files[num_qs] = queriesFile
            true_values_lists[num_qs] = true_values

        jobs = [] # jobs for sketches
        for sketch_size in sketch_sizes:    
            for run_function, sketch_name in sketch_functions:
                for num_qs in NUM_QUERIES_LIST:
                    size = sketch_size
                    if sketch_name == "MomentSketch":
                        size = MomentSketch_ks[sketch_sizes.index(size)] # translating sketch_size to MomentSketch k
                    elif sketch_name == "KLL sketch":
                        size = KLL_ks[sketch_sizes.index(size)] # translating sketch_size to KLL k
                    queriesFile = queries_files[num_qs]
                    true_values = true_values_lists[num_qs]
                    jobs.append(delayed(one_experiment)(dataFile, N, queriesFile, true_values, size, run_function, sketch_name, data_name))
        
        #return data, queries, true_values # no need to return
        return jobs

    for data_func, data_name in data_sources:
        for N in data_sizes:
            jobsData.append(delayed(gen_one_dataset)(data_func, data_name, N))
            
    
    resJobs = Parallel(n_jobs=NUM_PARALLEL_JOBS)(jobsData)
    jobs = []
    [ jobs.extend(subjobs) for subjobs in resJobs] 
    #print(jobs)
    print(f"============== DATA LOADED ===================")
    results = Parallel(n_jobs=NUM_PARALLEL_JOBS)(jobs)


    ##################### PLOTTING SETUP ###########################

    plt.rcParams.update({'font.size': 12})
    plt.rcParams["figure.figsize"] = (6,4)

    # Get the default color cycle
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Define your desired linestyles
    linestyles = ['-', ':', '-.', '--']

    # Create a paired cycle: assign a linestyle to each default color, cycling through linestyles.
    paired_cycle = []
    for i, color in enumerate(default_colors):
        paired_cycle.append({'color': color, 'linestyle': linestyles[i % len(linestyles)]})

    # Apply the new property cycle to matplotlib's rcParams
    plt.rcParams['axes.prop_cycle'] = cycler(**{
        'color': [d['color'] for d in paired_cycle],
        'linestyle': [d['linestyle'] for d in paired_cycle]
    })

    idx = 0
    for data_func, data_name in data_sources:
        for sketch_size in sketch_sizes:
            for N in data_sizes:
                query_times = {fn.__name__: [] for fn,_ in sketch_functions}
                #actual_sketch_sizes = {fn.__name__: [] for fn,_ in sketch_functions}
                for run_function, sketch_name in sketch_functions:
                    for num_qs in NUM_QUERIES_LIST:
                        average_error, max_error, actual_sketch_size, true_values, errors, time_per_update_mus, time_per_query_mus = results[idx]
                        idx += 1
                        if average_error == float('inf'):
                            print(f"skipping plot for: {data_name}, N={N}, sketch={sketch_name}, size={sketch_size}")
                            continue
                        query_times[run_function.__name__].append(time_per_query_mus)
                    #actual_sketch_sizes[run_function.__name__].append(actual_sketch_size)

                fig_qtm, ax_qtm = plt.subplots()
                for run_function, sketch_name in sketch_functions:
                    ax_qtm.plot(NUM_QUERIES_LIST, query_times[run_function.__name__], label=f"{sketch_name}")
                ax_qtm.set_xlabel('number of queries')
                ax_qtm.set_ylabel('time per query [μs]')
                ax_qtm.grid(True)
                fig_qtm.tight_layout()
                ax_qtm.legend()
                # ax_qtm.set_title(f"{data_name}")
                fig_qtm.savefig(plots_dir+f"query_time_{data_name}_sketchSize{sketch_size}_N{N}_legend.pdf", format='pdf')
                ax_qtm.set_yscale('log')
                ax_qtm.set_xscale('log')
                ax_qtm.grid(True)
                fig_qtm.tight_layout()
                ax_qtm.legend()
                # ax_qtm.set_title(f"{data_name}")
                fig_qtm.savefig(plots_dir+f"query_time_{data_name}_sketchSize{sketch_size}_N{N}_LogLog_legend.pdf", format='pdf')


