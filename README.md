# SplineSketch: a new quantile sketch with uniform error guarantees and high accuracy in practice

Here, we provide its prototype implementation [in Python](spline_sketch_uniform.py) and [in Java](SplineSketch.java), and an experimental pipeline for evaluating its accuracy on synthetic and real-world datasets and also its update, merge, and query times, in comparison with [t-digest](https://github.com/tdunning/t-digest), [KLL](https://datasketches.apache.org/docs/KLL/KLLSketch.html), GKAdaptive, and [MomentSketch](https://github.com/stanford-futuredata/msketch).

## Running experiments

**Setup:** Clone the repository and then run `make` to compile the Java wrappers that run the individual skeches.

There are four experimental pipelines, with parameters adjusted in the individual Python source codes:
1. Accuracy and running time experiments on synthetic datasets: run with `python run_experiments_IID.py`
2. Accuracy and running time experiments on real-world datasets: download datasets as described below and then run with `python run_experiments_datasets.py` (optionally adjust the datasets in `load_<dataset>_data` functions)
3. Update time experiment:  run with `python run_experiments_update_time.py`
4. Query time experiment:  run with `python run_experiments_query_time.py`

All of these Python programs produce a set of plots with results into `plots/` directory.

### Downloading real-world datasets

- [HEPMASS dataset](https://archive.ics.uci.edu/dataset/347/hepmass) from UC Irvine ML Repository: download `all_train.csv.gz` and `all_test.csv.gz` and decompress both files into `datasets/hepmass/`
- [Power dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) from UC Irvine ML Repository: download into `datasets/household_power_consumption/household_power_consumption.txt`
- Books dataset from [SOSD](https://github.com/learnedsystems/SOSD) (a benchmark for learned indexes): download using `download_books_dataset.sh`.
