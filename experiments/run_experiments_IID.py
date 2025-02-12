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

NUM_PARALLEL_JOBS = 4 # number of jobs run in parallel

data_sources = [(normal, "normal"),
                (uniform, "uniform"),
                (gumbel, "gumbel"),
                (lognormal, "lognormal"), (pareto, "pareto"), 
                (loguniform, "loguniform"), 
                (signed_lognormal, "signed_lognormal"),
                (signed_loguniform, "signed_loguniform"),
                (signed_loguniform_extreme, "signed_loguniform_extreme"),
                # (distinct_values_42, "distinct_values_42"),
                # (distinct_values_5, "distinct_values_5"),
                # (distinct_values_150, "distinct_values_150"),
                (normal_with_1_large_change, "normal_with_1_large_change"),
                (normal_with_1_small_change, "normal_with_1_small_change"),
                (normal_and_distinct_42, "normal_and_distinct_42"),
                ]
sketch_sizes = [15,20,25,32,50,75,100,125,150,175,200,250]
MomentSketch_ks = [7,9,11,13,15,19,23,27,33,39,45,51,57]
KLL_ks = [8,24,42,52,62,72,82,92,102,112,122,132] # KLL size is also influenced by log(N) in the DataSketches implementation
input_sizes_log10 = [8] 
NUM_QUERIES = 10000
sketch_functions = [#(run_splineSketchUniform, "SplineSketch(Py)"),
                    (run_splinesketch_java, "SplineSketch"),
                    (run_kll, "KLL sketch"), 
                    (run_MomentSketch, "MomentSketch"),
                    (run_tdigest, "t-digest")]


# HERE IS THE MAIN PROGRAM
##########################

def one_experiment(dataFile, N, logN, queriesFile, true_values, sketch_size, run_function, sketch_name, data_name):
    info = f"{sketch_name}_logN{logN}_size{sketch_size}_dataset_{data_name}"
    estimated_ranks, actual_sketch_size, update_time_ns, query_time_ns = run_function(dataFile, queriesFile, N, sketch_size, info)
    time_per_update_mus = update_time_ns / (1000.0 * N) 
    time_per_query_mus = query_time_ns / (1000.0 * len(true_values))

    # Calculate the errors
    if len(estimated_ranks) == len(true_values):
        errors = np.abs(np.array(estimated_ranks) - np.array(true_values))
    else: # in case of an error (e.g. from MomentSketch), inf is the estimate
        errors = np.array([float('inf') for _ in true_values])
        print(f"!!!!! {sketch_name}, logN {logN}, size {actual_sketch_size}, dataset {data_name} -- probably FAILED")

    average_error = np.mean(errors)
    max_error = max(errors)

    print(f"{sketch_name}, logN {logN}, size {actual_sketch_size}, dataset {data_name}. Average error {average_error}, max error {max_error}")


    return average_error, max_error, actual_sketch_size, true_values, errors, time_per_update_mus, time_per_query_mus

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
    print(f"input_sizes_log10 = {input_sizes_log10}")
    print(f"sketch_sizes = {sketch_sizes}")
    print(f"sketch_functions = {sketch_functions}")

    
    jobsData = []
    jobs = [] # jobs for sketches


    def gen_one_dataset(data_func, data_name, logN):
        dataFile = datasets_dir+f"{data_name}_logN{logN}.data.txt"
        queriesFile = datasets_dir+f"{data_name}_logN{logN}.queries.txt"

        if not os.path.exists(dataFile) or not os.path.exists(queriesFile):
            data = data_func(np.power(10, logN))

            queries = []
            queries = sorted(data)[::int(len(data)/NUM_QUERIES)]

            true_values = compute_true_ranks(data, queries)

            write_floats_to_file(data, dataFile)
            write_floats_to_file(queries, queriesFile)

            print(f"created dataset {data_name} for logN={logN} in {dataFile}, queries in {queriesFile})") # aspect ratio = {alpha}, refined aspect ratio = {alpha2} (k={defk})")
        else:
            data = load_floats_from_file(dataFile)
            queries = load_floats_from_file(queriesFile)
            true_values = compute_true_ranks(data, queries)
            print(f"loaded dataset {data_name} for logN={logN} in {dataFile}, queries in {queriesFile})")
        
        jobs = [] # jobs for sketches
        for sketch_size in sketch_sizes:    
            for run_function, sketch_name in sketch_functions:
                size = sketch_size
                if sketch_name == "MomentSketch":
                    size = MomentSketch_ks[sketch_sizes.index(size)] # translating sketch_size to MomentSketch k
                elif sketch_name == "KLL sketch":
                    size = KLL_ks[sketch_sizes.index(size)] # translating sketch_size to KLL k

                jobs.append(delayed(one_experiment)(dataFile, len(data), logN, queriesFile, true_values, size, run_function, sketch_name, data_name))
        
        #return data, queries, true_values # no need to return
        return jobs

    for data_func, data_name in data_sources:
        for logN in input_sizes_log10:
            if (np.power(10, logN) < NUM_QUERIES):
                print(f"SKIPPING logN={logN} as 10**{logN} < {NUM_QUERIES}")
                continue
            jobsData.append(delayed(gen_one_dataset)(data_func, data_name, logN))
            
    
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
        for logN in input_sizes_log10:
            n = 10**logN
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

                for run_function, sketch_name in sketch_functions:
                    average_error, max_error, actual_sketch_size, true_values, errors, time_per_update_mus, time_per_query_mus = results[idx]
                    idx += 1
                    if average_error == float('inf'):
                        print(f"skipping plot for: {data_name}, logN={logN}, sketch={sketch_name}, size={sketch_size}")
                        continue
                    average_errors[run_function.__name__].append(average_error)
                    max_errors[run_function.__name__].append(max_error)
                    update_times[run_function.__name__].append(time_per_update_mus)
                    query_times[run_function.__name__].append(time_per_query_mus)
                    
                    actual_sketch_sizes[run_function.__name__].append(actual_sketch_size)

                    # ax_re.plot(true_values, errors, label=f"{sketch_name}, size {actual_sketch_size}")


                #fig_re.savefig(plots_dir+f"{data_name}_logN={logN}_size={sketch_size}_nolegend.pdf", format='pdf')
                fig_re.legend()
                #ax_re.set_title(f"{data_name}_logN={logN}_size={sketch_size}") #TODO: to be removed (now for info which plot is which)
                ax_re.grid(True)
                fig_re.savefig(plots_dir+f"{data_name}_logN{logN}_size{sketch_size}_legend.pdf", format='pdf')

            fig_avg, ax_avg = plt.subplots()
            fig_max, ax_max = plt.subplots()
            fig_utm, ax_utm = plt.subplots()
            fig_qtm, ax_qtm = plt.subplots()
            for run_function, sketch_name in sketch_functions:
                ax_avg.plot(actual_sketch_sizes[run_function.__name__], average_errors[run_function.__name__], label=f"{sketch_name}")
                ax_max.plot(actual_sketch_sizes[run_function.__name__], max_errors[run_function.__name__], label=f"{sketch_name}")
                ax_utm.plot(actual_sketch_sizes[run_function.__name__], update_times[run_function.__name__], label=f"{sketch_name}")
                ax_qtm.plot(actual_sketch_sizes[run_function.__name__], query_times[run_function.__name__], label=f"{sketch_name}") - 8)) / max_errors["run_splinesketch_java"][i] if max_errors["run_splinesketch_java"][i] != 0 else 100 for i in range(len(actual_sketch_sizes["run_splinesketch_java"]))], label=f"(n / k) / splineSketch err.")

            ax_avg.set_yscale('log')
            ax_avg.grid(True)
            fig_avg.tight_layout()
            fig_avg.savefig(plots_dir+f"{data_name}_logN{logN}_average_nolegend.pdf", format='pdf')
            ax_avg.set_xlabel('sketch size in bytes')
            ax_avg.set_ylabel('average rank error (log scale)')
            fig_avg.tight_layout()
            ax_avg.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, fancybox=False, shadow=False)
            #ax_avg.set_title(f"{data_name}")
            fig_avg.savefig(plots_dir+f"{data_name}_logN{logN}_average_legend.pdf", format='pdf')

            ax_max.set_yscale('log')
            ax_max.grid(True)
            fig_max.tight_layout()
            fig_max.savefig(plots_dir+f"{data_name}_logN{logN}_max_nolegend.pdf", format='pdf')
            ax_max.set_xlabel('sketch size in bytes')
            ax_max.set_ylabel('maximum rank error (log scale)')
            fig_avg.tight_layout()
            ax_max.legend()
            # ax_max.set_title(f"{data_name}")
            fig_max.savefig(plots_dir+f"{data_name}_logN{logN}_max_legend.pdf", format='pdf')

            ax_utm.set_xlabel('sketch size in bytes')
            ax_utm.set_ylabel('time per update [μs] (log scale)')
            # ax_utm.set_yscale('log')
            ax_utm.grid(True)
            fig_utm.tight_layout()
            ax_utm.legend()
            fig_utm.savefig(plots_dir+f"{data_name}_logN{logN}_update_time_legend.pdf", format='pdf')

            ax_qtm.set_xlabel('sketch size in bytes')
            ax_qtm.set_ylabel('time per query [μs] (log scale)')
            ax_qtm.set_yscale('log')
            ax_qtm.grid(True)
            fig_qtm.tight_layout()
            ax_qtm.legend()
            fig_qtm.savefig(plots_dir+f"{data_name}_logN{logN}_query_time_legend.pdf", format='pdf')


