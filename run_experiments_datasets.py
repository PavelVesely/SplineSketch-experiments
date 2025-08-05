import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

import os
from joblib import Parallel, delayed

from helper_funcs import *
from run_sketches_fncs import *

import csv

########################## PARAMETERS OF EXPERIMENTS ######################
###########################################################################
# for datasets, see below
sketch_sizes = [15,20,25,32,50,75,100,125,150,175,200,250] #,350,400,450,500]
MomentSketch_ks = [7,9,11,13,15,19,23,27,33,39,45,51,57]
KLL_ks = [8,24,42,52,62,72,82,92,102,112,122,132] #note: KLL size is also influenced by log(N) in the DataSketches implementation
NUM_QUERIES = 10000
NUM_PARTS_FOR_MERGEABILITY = 1 # should be 1 for the streaming setting, using 10000 for testing mergeability
MAX_ALLOWED_SIZE_IN_BYTES = 4500
TEST_SKEWED = True # whether to include the version with MG or not
sketch_functions = [#(run_splineSketchUniform, "SplineSketch"),
                    (run_splinesketch_java, "SplineSketch"),
                    (run_kll, "KLL sketch"), 
                    # (run_MomentSketch, "MomentSketch"),
                    (run_tdigest, "t-digest"),
                    # (run_GK, "GKAdaptive"),
                    (run_splinesketchLong_java, "SplineSketchLong"),
                    #(run_DDSketch, "DDSketch"),
                    ]
if TEST_SKEWED:
    sketch_functions.append((run_splinesketchMG_java, "SplineSketch+MG"))

NUM_PARALLEL_JOBS = 7 # number of jobs run in parallel

# FUNCTION FOR LOADING REAL-WORLD DATASETS
###############################

def load_hepmass_data(train_file='../../datasets/hepmass/all_train.csv', test_file='../../datasets/hepmass/all_test.csv'):
    data = []

    # Helper function to read and process a file
    def process_file(filename):
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for row in reader:
                data.append(float(row[1]))  # Append the first column as float

    # Process both training and testing files
    process_file(train_file)
    process_file(test_file)
    
    return data

def load_power_data(filename='../../datasets/household_power_consumption/household_power_consumption.txt'):
    data = []

    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)  # Skip the header
        for row in reader:
            try: # there are some rows with '?'
                data.append(float(row[2]))  # Append the second column as float 
            except:
                pass
    return data

def load_sosd_books_data(filename='../../datasets/SOSDdata/books_800M_uint64'):
    data = np.fromfile(filename, dtype=np.uint64)[1:] # based on https://github.com/learnedsystems/SOSD/blob/f52f4cba01dfcd37f1574551fccef00198863b88/downsample.py#L10
    return np.random.permutation(data)

def load_SOSD_data(filename, dir='../../datasets/SOSDdata/'): # we randomly permute the data
    filename = dir + filename
    data = np.fromfile(filename, dtype=np.uint64)[1:] # based on https://github.com/learnedsystems/SOSD/blob/f52f4cba01dfcd37f1574551fccef00198863b88/downsample.py#L10
    return np.random.permutation(data)

def load_sosd_wiki_data():
    return load_SOSD_data("wiki_ts_200M_uint64")

# def load_sosd_books_data():
#     return load_SOSD_data("books_800M_uint64")

def load_sosd_osm_data():
    return load_SOSD_data("osm_cellids_800M_uint64")



data_sources = [
                # (load_hepmass_data, "hepmass"),
                # (load_power_data, "power"),
                (load_sosd_books_data, "sosd_books"),
                (load_sosd_wiki_data, "sosd_wiki"),
                # (load_sosd_osm_data, "sosd_osm")
                ] 

# HERE IS THE MAIN PROGRAM
##########################

def one_experiment_dataset(dataFile, queriesFile, N, true_values, sketch_size, run_function, sketch_name, data_name):
    info = f"{sketch_name}, size {sketch_size}, dataset {data_name}"
    estimated_ranks, actual_sketch_size, update_time_ns, query_time_ns = run_function(dataFile, queriesFile, N, sketch_size, NUM_PARTS_FOR_MERGEABILITY, info)
    if (NUM_PARTS_FOR_MERGEABILITY == 1):
        time_per_update_mus = update_time_ns / (1000.0 * N) 
    else:
        time_per_update_mus = update_time_ns / (1000.0 * (NUM_PARTS_FOR_MERGEABILITY - 1)) 
    time_per_query_mus = query_time_ns / (1000.0 * len(true_values))

    # Calculate the errors
    if len(estimated_ranks) == len(true_values):
        errors = np.abs(np.array(estimated_ranks) - np.array(true_values))
    else: # in case of an error (e.g. from MomentSketch), inf is the estimate
        errors = np.array([float('inf') for _ in true_values])
        print(f"!!!!! {sketch_name}, size {actual_sketch_size}, dataset {data_name} -- probably FAILED")

    average_error = np.mean(errors) / N # normalize
    max_error = max(errors) / N # normalize

    print(f"{sketch_name}, size {actual_sketch_size}, dataset {data_name}. Average error {average_error}, max error {max_error}")


    return average_error, max_error, actual_sketch_size, true_values, errors, time_per_update_mus, time_per_query_mus, N

if __name__ == "__main__":
    # create necessary dirs.
    datasets_dir='./datasets/'
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
    plots_dir='./plots/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    print(f"data sources = {data_sources}")
    print(f"sketch_sizes = {sketch_sizes}")
    print(f"sketch_functions = {sketch_functions}")

    jobsData = []
    jobs = [] # jobs for sketches

    def gen_one_dataset(data_func, data_name):
        data = data_func()
        dataFile = datasets_dir+f"{data_name}.data.txt"
        queriesFile = datasets_dir+f"{data_name}.queries.txt"

        if not os.path.exists(dataFile) or not os.path.exists(queriesFile):
            queries = []
            if len(data) < 10000: # changed from if len(queries) < 10000:
                queries = sorted(data)
            else:
                queries = sorted(data)[::int(len(data)/10000)]
            write_floats_to_file(data, dataFile)
            write_floats_to_file(queries, queriesFile)
        else:
            data = load_floats_from_file(dataFile)
            queries = load_floats_from_file(queriesFile)

        n = len(data)
        true_values = compute_true_ranks(data, queries)
        print(f"loaded data and computed true ranks for {data_name}, n = {n}") #; aspect ratio = {alpha}, refined aspect ratio = {alpha2} (k={defk})")
        jobs = [] # jobs for sketches
        for sketch_size in sketch_sizes:    
            for run_function, sketch_name in sketch_functions:
                size = sketch_size
                if sketch_name == "MomentSketch":
                    size = MomentSketch_ks[sketch_sizes.index(size)] # translating sketch_size to MomentSketch k
                elif sketch_name == "KLL sketch":
                    size = KLL_ks[sketch_sizes.index(size)] # translating sketch_size to KLL k
                jobs.append(delayed(one_experiment_dataset)(dataFile, queriesFile, len(data), true_values, size, run_function, sketch_name, data_name))
        
        #return data, queries, true_values # no need to return
        return jobs

    for data_func, data_name in data_sources:
        jobsData.append(delayed(gen_one_dataset)(data_func, data_name))
            
    
    resJobs = Parallel(n_jobs=NUM_PARALLEL_JOBS)(jobsData)
    jobs = []
    [ jobs.extend(subjobs) for subjobs in resJobs] 
    print(f"============== DATA LOADED ===================")
    results = Parallel(n_jobs=NUM_PARALLEL_JOBS)(jobs)
    
    ###################### PLOTTING SETUP
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["figure.figsize"] = (6,4)

    # Get the default color cycle
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Define your desired linestyles
    linestyles = ['-', ':', '-.', '--', (5, (10, 3))] # the last one for GK: 
    if TEST_SKEWED:
        linestyles.append('-') # for SplineSketch+MG    
    linestyles.append((0, (1, 10))) # for error bound

    # Create a paired cycle: assign a linestyle to each default color, cycling through linestyles.
    paired_cycle = []
    for i, color in enumerate(default_colors):
        paired_cycle.append({'color': color, 'linestyle': linestyles[i % len(linestyles)]})

    # Apply the new property cycle to matplotlib's rcParams
    plt.rcParams['axes.prop_cycle'] = cycler(**{
        'color': [d['color'] for d in paired_cycle],
        'linestyle': [d['linestyle'] for d in paired_cycle]
    })
    minSize = sketch_sizes[0]*16
    maxSize = sketch_sizes[-1]*16
    idx = 0
    mergingSuffix = f"_mergingM{NUM_PARTS_FOR_MERGEABILITY}" if (NUM_PARTS_FOR_MERGEABILITY > 1) else ""
    if TEST_SKEWED:
        mergingSuffix += "_withMG"
    for data_func, data_name in data_sources:
        average_errors = {fn.__name__: [] for fn,_ in sketch_functions}
        max_errors = {fn.__name__: [] for fn,_ in sketch_functions}
        update_times = {fn.__name__: [] for fn,_ in sketch_functions}
        query_times = {fn.__name__: [] for fn,_ in sketch_functions}
        actual_sketch_sizes = {fn.__name__: [] for fn,_ in sketch_functions}
        improvementOverTD = {fn.__name__: [] for fn,_ in sketch_functions}

        for sketch_size in sketch_sizes:
            fig_re, ax_re = plt.subplots()
            ax_re.set_xlabel('true rank')
            ax_re.set_ylabel('error')
            #ax_re.set_yscale('log') # using default linear scale here as it doesn't look very good with log scale
            fig_re.tight_layout()
            dataSize = 0
            for run_function, sketch_name in sketch_functions:
                average_error, max_error, actual_sketch_size, true_values, errors, time_per_update_mus, time_per_query_mus, n = results[idx]
                idx += 1
                dataSize = n
                if average_error == float('inf') or actual_sketch_size > MAX_ALLOWED_SIZE_IN_BYTES:
                    print(f"skipping plot for: {data_name}, sketch={sketch_name}, size={sketch_size}, actual size={actual_sketch_size}, avg error={average_error}")
                    continue
                average_errors[run_function.__name__].append(average_error)
                max_errors[run_function.__name__].append(max_error)
                actual_sketch_sizes[run_function.__name__].append(actual_sketch_size)
                update_times[run_function.__name__].append(time_per_update_mus)
                query_times[run_function.__name__].append(time_per_query_mus)

                ax_re.plot(true_values, errors, label=f"{sketch_name}, size {actual_sketch_size}")


            # #fig_re.savefig(f"plots/{data_name}_size={sketch_size}_nolegend.pdf", format='pdf')
            # fig_re.legend()
            # #ax_re.set_title(f"{data_name}_size={sketch_size}")
            # ax_re.grid(True)
            # fig_re.savefig(f"plots/{data_name}{mergingSuffix}_size{sketch_size}_legend.pdf", format='pdf')
        fig_avg, ax_avg = plt.subplots()
        fig_max, ax_max = plt.subplots()
        fig_utm, ax_utm = plt.subplots()
        fig_qtm, ax_qtm = plt.subplots()
        i = 0
        #minSize = float('inf')
        #maxSize = 0
        for run_function, sketch_name in sketch_functions:
            actual_sketch_sizes[run_function.__name__].sort()
            # minSize = np.min(minSize, actual_sketch_sizes[run_function.__name__][0])
            # minSize = np.max(maxSize, actual_sketch_sizes[run_function.__name__][-1])
            ax_avg.plot(actual_sketch_sizes[run_function.__name__], average_errors[run_function.__name__], label=f"{sketch_name}")
            ax_max.plot(actual_sketch_sizes[run_function.__name__], max_errors[run_function.__name__], label=f"{sketch_name}")
            ax_utm.plot(actual_sketch_sizes[run_function.__name__], update_times[run_function.__name__], label=f"{sketch_name}")
            ax_qtm.plot(actual_sketch_sizes[run_function.__name__], query_times[run_function.__name__], label=f"{sketch_name}")
            i += 1
        xx = np.linspace(minSize, maxSize, 1000)
        yy = 4 * 16 / xx # 16 bytes per bucket/centroid; increased so that KLL and GK fit
        ax_avg.plot(xx, yy, label=f"error bound")
        ax_avg.set_yscale('log')
        ax_avg.grid(True)
        fig_avg.tight_layout()
        fig_avg.savefig(f"plots/{data_name}{mergingSuffix}_average_nolegend.pdf", format='pdf')
        ax_avg.set_xlabel('sketch size in bytes')
        ax_avg.set_ylabel('average rank error (log scale)')
        ax_avg.legend() 
        #ax_avg.set_title(f"{data_name}_average")
        # ax_avg.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=6, fancybox=False, shadow=False)
        fig_avg.savefig(f"plots/{data_name}{mergingSuffix}_average_legend.pdf", format='pdf')

        ax_max.plot(xx, yy, label=f"error bound")
        ax_max.set_yscale('log')
        ax_max.grid(True)
        fig_max.tight_layout()
        fig_max.savefig(f"plots/{data_name}{mergingSuffix}_max_nolegend.pdf", format='pdf')
        ax_max.set_xlabel('sketch size in bytes')
        ax_max.set_ylabel('max rank error (log scale)')
        ax_max.legend()
        fig_max.tight_layout()
        #ax_max.set_title(f"{data_name}_max")
        fig_max.savefig(f"plots/{data_name}{mergingSuffix}_max_legend.pdf", format='pdf')

        ax_utm.set_xlabel('sketch size in bytes')
        if NUM_PARTS_FOR_MERGEABILITY > 1:
            ax_utm.set_ylabel('time per merge operation [μs] (log scale)')
        else:
            ax_utm.set_ylabel('time per update [μs] (log scale)')
        ax_utm.set_yscale('log')
        ax_utm.grid(True)
        fig_utm.tight_layout()
        ax_utm.legend()
        fig_utm.savefig(plots_dir+f"{data_name}{mergingSuffix}_update_time_legend.pdf", format='pdf')

        ax_qtm.set_xlabel('sketch size in bytes')
        ax_qtm.set_ylabel('time per query [μs] (log scale)') #
        ax_qtm.set_yscale('log')
        ax_qtm.grid(True)
        fig_qtm.tight_layout()
        ax_qtm.legend()
        fig_qtm.savefig(plots_dir+f"{data_name}{mergingSuffix}_query_time_legend.pdf", format='pdf')
