# SplineSketch: a new quantile sketch
SplineSketch: a new quantile sketch with uniform error guarantees and high accuracy in practice.

Here, we provide its prototype implementation in Python and in Java, and an experimental pipeline for evaluating its accuracy on synthetic and real-world datasets and also its update and query times, in comparison with t-digest, KLL, and MomentSketch.

Note: This repository accompanies a paper under submission that we will make available in about 3 months. The SplineSketch implementation is intended to perform accuracy and running time experiments; it is thus to some extent under development and may change. A SplineSketch version that is suitable for usage in production will appear later in a different repository.

## Running experiments

**Setup:** Clone the repository and then run `make` to compile the Java wrappers that run the individual skech.

There are four experimental pipelines, with parameters adjusted in the individual Python source codes:
1. Accuracy and running time experiments on synthetic datasets: run with `python run_experiments_IID.py`
2. Accuracy and running time experiments on real-world datasets: run with `python run_experiments_datasets.py`
3. Update time experiment:  run with `python run_experiments_update_time.py`
4. Query time experiment:  run with `python run_experiments_query_time.py`

All of these Python programs produce a set of plots with results into `plots/` directory.
