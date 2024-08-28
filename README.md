# ECL Suite

This repository hosts a suite of high-performance graph analytics codes for GPUs that have had their "benign" data races removed. A full description of the codes and methodology can be found in our paper (see below).

If you use any of the code or results in this repository, please cite the following publication:

>Yiqian Liu, Avery VanAusdal, and Martin Burtscher. Performance Impact of Removing Data Races from GPU Graph Analytics Programs. Proceedings of the IEEE International Symposium on Workload Characterization. September 2024.

As a side note, you may also be interested in the related suites [Indigo](https://cs.txstate.edu/~burtscher/research/IndigoSuite/), [Indigo2](https://cs.txstate.edu/~burtscher/research/Indigo2Suite/), and [Indigo3](https://github.com/burtscher/Indigo3Suite/).

This repository includes the codes, inputs, and scripts to run the experiments performed in the paper. The repository is laid out as follows.

* All the baseline codes are in the `src/baseline` directory
* All the race-free codes are in the `src/racefree` directory
* The `library` directory contains the header files
* The `scripts` directory contains Python scripts that help run the codes, read the run times, and analyze the results, as well as a program that converts MatrixMarket graphs to the ECL format
* The `download_inputs.sh` Bash script downloads and prepares the input graphs used in our experiments
* The `download_grid.sh` Bash script downloads input graphs for the APSP codes
* The `all_tests.sh` Bash script performs the experiments. It runs all codes and places their runtime output logs in a new `results/` directory. Then it uses `scripts/analyze_results.py` to calculate the speedups and creates tables and figures in a new `output/` directory.

## Installation and Setup

The Python scripts that automate compiling and running the codes require the numpy, matplotlib, and scipy packages. We recommend installing and managing these with [pip](https://pypi.org/project/pip/):

    pip3 install numpy matplotlib scipy

## Acquiring Input Graphs

To download and prepare all of the input graphs used in the paper's experiments, run:

    ./download_inputs.sh
    
The script creates the `inputs-undirected/` and `inputs-directed/` directories and places the input graphs in them.

## Run Paper Experiments

After acquiring the inputs, to compile and run all codes on all inputs, run:

    ./all_tests.sh

The script runs every baseline and race-free code on every appropriate input 9 times by default, then calculates the speedups from baseline to race-free. This takes 1-2 hours for a 2070 Super.

The speedups in CSV and graph form can be found in the `output/` directory.

## Experiment Customization

On a system with multiple GPUs, the `all_tests.sh` script defaults to using the fastest GPU. To change this, edit the value of `CUDA_VISIBLE_DEVICES` on line 4 of `all_tests.sh`.

By default, each code is run on each input 9 times, the same number of times we used to generate the paper's results. The median of those 9 runtimes is used for the speedup calculations. If time is a concern, this number can be edited on line 5 of the `all_tests.sh` script.

