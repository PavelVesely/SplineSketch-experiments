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

# ONLY SPLINESKETCH

########################## PARAMETERS OF EXPERIMENTS ######################3

NUM_PARALLEL_JOBS = 8 # number of jobs run in parallel

data_sources = [(normal, "normal"),
                # (uniform, "uniform"),
                # (gumbel, "gumbel"),
                # (lognormal, "lognormal"), 
                # (pareto, "pareto"), 
                # (loguniform, "loguniform"), 
                # (signed_lognormal, "signed_lognormal"),
                # (signed_loguniform, "signed_loguniform"),
                # (signed_loguniform_extreme, "signed_loguniform_extreme"),
                # (normal_with_1_large_change, "normal_with_1_large_change"),
                # (normal_with_1_small_change, "normal_with_1_small_change"),
                ## with heavy hitters
                # # (distinct_values_42, "distinct_values_42"),
                # # (distinct_values_5, "distinct_values_5"),
                # # (distinct_values_150, "distinct_values_150"),
                #(normal_and_distinct_42, "normal_and_distinct_42"),
                #(sorted_with_frequent, "sorted_with_frequent")
                ]
sketch_sizes = [15,20,25,32,50,75,100,125,150,175,200,250]
input_sizes_log10 = [6] # 8 was used for 
NUM_QUERIES = 10000
MAX_ALLOWED_SIZE_IN_BYTES = 4500
NUM_PARTS_FOR_MERGEABILITY = 1 # should be 1 for the streaming setting, we use 10000 for mergeability testing

## ABLATION STUDY TYPE -- NEEDS ADJUSTMENT ALSO around LINE 138

# SETUP for interpolations
PARAM_VALS = [(3, "PCHIP interpolation"), (1, "linear interpolation")]
EXP_NAME = "interpolations"

# SETUP for heur. error type
# PARAM_VALS = [(2, "2nd deriv. of CDF"), (3, "3rd deriv. of CDF"), (1, "bucket counter"), (-1, "bucket length"), (0, "none")]
# EXP_NAME = "heuristicError"

# SETUP for defaultBucketBoundMult
# PARAM_VALS = [(x, f"defaultBucketBoundMult = {x}") for x in range(1, 8)]
# EXP_NAME = "defaultBucketBoundMult"

# SETUP for minFracBucketBoundToSplit
# PARAM_VALS = [(10**x, f"minFracBucketBoundToSplit = {10**x}") for x in range(-5, 0)]
# EXP_NAME = "minFracBucketBoundToSplit"

# SETUP for epochIncrFactor
# PARAM_VALS = [(x/10, f"epochIncrFactor = {x/10}") for x in range(11, 21)]
# EXP_NAME = "epochIncrFactor"

# HERE IS THE MAIN PROGRAM
##########################

def one_experiment(dataFile, N, logN, queriesFile, true_values, sketch_size, params, param_name, data_name):
    info = f"{param_name}_logN{logN}_size{sketch_size}_dataset_{data_name}"
    estimated_ranks, actual_sketch_size, update_time_ns, query_time_ns = run_splinesketchAdjustable_java(dataFile, queriesFile, N, sketch_size, NUM_PARTS_FOR_MERGEABILITY, params)
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
        print(f"!!!!! {param_name}, logN {logN}, size {actual_sketch_size}, dataset {data_name} -- probably FAILED")

    average_error = np.mean(errors) / N # normalize
    max_error = max(errors) / N # normalize

    print(f"{param_name}, logN {logN}, size {actual_sketch_size}, dataset {data_name}. Average error {average_error}, max error {max_error}")


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

    print(f"EXPERIMENT {EXP_NAME}")
    print(f"param vals = {PARAM_VALS}")
    print(f"data sources = {data_sources}")
    print(f"input_sizes_log10 = {input_sizes_log10}")
    print(f"sketch_sizes = {sketch_sizes}")
    if NUM_PARTS_FOR_MERGEABILITY == 1:
        print("STREAMING setting")
    else:
        print(f"MERGEABILITY with NUM_PARTS_FOR_MERGEABILITY={NUM_PARTS_FOR_MERGEABILITY}")
    
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
            for param_val, param_name in PARAM_VALS:
                size = sketch_size
                # heuristicErrorType, int interpolationDegree, double splitJoinRatio, double minRelativeBucketLength, double minFracBucketBoundToSplit, double epochIncrFactor, double defaultBucketBoundMult
                # DEFAULT: f"2 3 1.5 0.00000001 0.01 1.25 3.0"
                # this.splitJoinRatio = 1.5;
                # this.minRelativeBucketLength = 1e-8;
                # this.minFracBucketBoundToSplit = 0.01;
                # this.epochIncrFactor = 1.25;
                # this.defaultBucketBoundMult = 3.0;
                

                # TODO: switch here
                params = f"2 {param_val} 1.5 0.00000001 0.01 1.25 3.0" # for testing interpolations
                # params = f"{param_val} 3 1.5 0.00000001 0.01 1.25 3.0" # for testing heur. error type
                # params = f"2 3 {param_val} 0.00000001 0.01 1.25 3.0" # for testing gamma
                # params = f"2 3 1.5 0.00000001 0.01 1.25 {param_val}" # for testing defaultBucketBoundMult
                # params = f"2 3 1.5 0.00000001 {param_val} 1.25 3.0" # for testing minFracBucketBoundToSplit
                # params = f"2 3 1.5 0.00000001 0.01 {param_val} 3.0" # for testing epochIncrFactor
                jobs.append(delayed(one_experiment)(dataFile, len(data), logN, queriesFile, true_values, size, params, param_name, data_name))
        
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
    plt.rcParams["figure.figsize"] = (8,7) #(3.3,4)

    # Get the default color cycle
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Define your desired linestyles
    linestyles = ['-', ':', '-.', '--', (5, (10, 3))] 

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
    mergingSuffix = f"_mergingM{NUM_PARTS_FOR_MERGEABILITY}" if (NUM_PARTS_FOR_MERGEABILITY > 1) else ""
    for data_func, data_name in data_sources:
        for logN in input_sizes_log10:
            n = 10**logN
            average_errors = {param_name: [] for _,param_name in PARAM_VALS}
            max_errors = {param_name: [] for _,param_name in PARAM_VALS}
            update_times = {param_name: [] for _,param_name in PARAM_VALS}
            query_times = {param_name: [] for _,param_name in PARAM_VALS}
            actual_sketch_sizes = {param_name: [] for _,param_name in PARAM_VALS}
            improvementOverTD = {param_name: [] for _,param_name in PARAM_VALS}

            for sketch_size in sketch_sizes:
                fig_re, ax_re = plt.subplots()
                ax_re.set_xlabel('true rank')
                ax_re.set_ylabel('error')
                fig_re.tight_layout()

                for param_val, param_name in PARAM_VALS:
                    average_error, max_error, actual_sketch_size, true_values, errors, time_per_update_mus, time_per_query_mus = results[idx]
                    idx += 1
                    if average_error == float('inf') or actual_sketch_size > MAX_ALLOWED_SIZE_IN_BYTES:
                        print(f"skipping plot for: {data_name}, logN={logN}, sketch={param_name}, size={sketch_size}, actual size={actual_sketch_size}, avg error={average_error}")
                        continue
                    average_errors[param_name].append(average_error)
                    max_errors[param_name].append(max_error)
                    update_times[param_name].append(time_per_update_mus)
                    query_times[param_name].append(time_per_query_mus)
                    
                    actual_sketch_sizes[param_name].append(actual_sketch_size)

                    ax_re.plot(true_values, errors, label=f"{param_name}, size {actual_sketch_size}")


                # #fig_re.savefig(plots_dir+f"{EXP_NAME}_{data_name}_logN={logN}_size={sketch_size}{mergingSuffix}_nolegend.pdf", format='pdf')
                # fig_re.legend()
                # #ax_re.set_title(f"{data_name}_logN={logN}_size={sketch_size}") #TODO: to be removed (now for info which plot is which)
                # ax_re.grid(True)
                # fig_re.savefig(plots_dir+f"{EXP_NAME}_{data_name}_logN{logN}_size{sketch_size}{mergingSuffix}_legend.pdf", format='pdf')
            fig_avg, ax_avg = plt.subplots()
            fig_max, ax_max = plt.subplots()
            fig_utm, ax_utm = plt.subplots()
            fig_qtm, ax_qtm = plt.subplots()
            for param_val, param_name in PARAM_VALS:
                actual_sketch_sizes[param_name].sort()
                # minSize = np.min(minSize, actual_sketch_sizes[param_name][0])
                # minSize = np.max(maxSize, actual_sketch_sizes[param_name][-1])
                ax_avg.plot(actual_sketch_sizes[param_name], average_errors[param_name], label=f"{param_name}")
                ax_max.plot(actual_sketch_sizes[param_name], max_errors[param_name], label=f"{param_name}")
                ax_utm.plot(actual_sketch_sizes[param_name], update_times[param_name], label=f"{param_name}")
                ax_qtm.plot(actual_sketch_sizes[param_name], query_times[param_name], label=f"{param_name}")

            ax_avg.set_yscale('log')
            ax_avg.grid(True)
            fig_avg.tight_layout()
            # fig_avg.savefig(plots_dir+f"{EXP_NAME}_{data_name}_logN{logN}{mergingSuffix}_average_nolegend.pdf", format='pdf')
            ax_avg.set_xlabel('sketch size in bytes')
            ax_avg.set_ylabel('average rank error (log scale)')
            ax_avg.legend()
            fig_avg.tight_layout()
            #ax_avg.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, fancybox=False, shadow=False)
            #ax_avg.set_title(f"{data_name}")
            fig_avg.savefig(plots_dir+f"{EXP_NAME}_{data_name}_logN{logN}{mergingSuffix}_average_legend.pdf", format='pdf', bbox_inches='tight')

            ax_max.set_yscale('log')
            ax_max.grid(True)
            fig_max.tight_layout()
            # fig_max.savefig(plots_dir+f"{EXP_NAME}_{data_name}_logN{logN}{mergingSuffix}_max_nolegend.pdf", format='pdf')
            ax_max.set_xlabel('sketch size in bytes')
            ax_max.set_ylabel('maximum rank error (log scale)')
            ax_max.legend()
            fig_avg.tight_layout()
            # ax_max.set_title(f"{data_name}")
            fig_max.savefig(plots_dir+f"{EXP_NAME}_{data_name}_logN{logN}{mergingSuffix}_max_legend.pdf", format='pdf', bbox_inches='tight')

            ax_utm.set_xlabel('sketch size in bytes')
            if NUM_PARTS_FOR_MERGEABILITY > 1:
                ax_utm.set_ylabel('time per merge operation [μs] (log scale)')
            else:
                ax_utm.set_ylabel('time per update [μs] (log scale)')
            ax_utm.set_yscale('log')
            ax_utm.grid(True)
            fig_utm.tight_layout()
            ax_utm.legend()
            fig_utm.savefig(plots_dir+f"{EXP_NAME}_{data_name}_logN{logN}{mergingSuffix}_update_time_legend.pdf", format='pdf', bbox_inches='tight')

            ax_qtm.set_xlabel('sketch size in bytes')
            ax_qtm.set_ylabel('time per query [μs] (log scale)')
            ax_qtm.set_yscale('log')
            ax_qtm.grid(True)
            fig_qtm.tight_layout()
            ax_qtm.legend()
            fig_qtm.savefig(plots_dir+f"{EXP_NAME}_{data_name}_logN{logN}{mergingSuffix}_query_time_legend.pdf", format='pdf', bbox_inches='tight')


