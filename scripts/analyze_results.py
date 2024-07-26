#!/usr/bin/python3 -u
# This script will analyze generated results residing in the results/ directory and output 2 speedup tables and a geomean speedup bar chart to the output/ directory.

'''
This file is part of the ECL Suite version 1.0.

BSD 3-Clause License

Copyright (c) 2024, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Yiqian Liu, Avery VanAusdal, and Martin Burtscher

URL: The latest version of this code is available at https://github.com/burtscher/ECL-Suite.

Publication: This work is described in detail in the following paper.
Yiqian Liu, Avery VanAusdal, and Martin Burtscher. Performance Impact of Removing Data Races from GPU Programs. Proceedings of the IEEE International Symposium on Workload Characterization. September 2024.
'''


import numpy as np
import os
import re
import statistics
import csv
from scipy.stats import gmean

results_dir = "./results/"
output_dir = "./output/"
os.makedirs(output_dir, exist_ok=True)

display_names = {"wikipedia-20061104" : "wikipedia", "toroid-hex.mesh-M-4-idx-0" : "toroid-hex", "klein-bottle.mesh-M-4-idx-0" : "klein-bottle", \
                 "star.mesh" : "star", "toroid-wedge.mesh" : "toroid-wedge", "cold-flow.mesh" : "cold-flow"}

def parse_runtime_file(filename):
    # Regular expression to capture the input name and compute time
    input_pattern = re.compile(r'(?:inputs-undirected|inputs-directed)\/(\S+).egr')
    time_pattern = re.compile(r'time: +(\d+\.\d+) s')

    results = {}
    current_input = None
    num_inputs = None

    with open(filename, 'r') as file:
        for line in file:
            # find num inputs
            re_match = re.search(r'num_inputs:\s*(\d+)', line)
            if re_match:
              num_inputs = int(re_match.group(1))

            # Match input names
            input_match = input_pattern.search(line)
            if input_match:
                current_input = input_match.group(1)
                if current_input not in results:
                    results[current_input] = []

            # Match compute times
            line = re.sub(r'Dev[-+]', 'runtime', line) #for MST
            time_match = time_pattern.search(line)
            if time_match and current_input:
                compute_time = float(time_match.group(1))
                results[current_input].append(compute_time)

    if not num_inputs:
      raise ValueError("num_inputs not found")

    # Verify that there are num_inputs inputs with the same number of runtimes each
    if len(results) != num_inputs:
        raise ValueError(f"Expected {num_inputs} inputs, but found {len(results)}.")

    num_runs = -1
    for input_name, runtimes in results.items():
        if num_runs == -1:
            num_runs = len(runtimes)
        elif len(runtimes) != num_runs:
            raise ValueError(f"Expected {num_runs} runtimes for input '{input_name}' in '{filename}', but found {len(runtimes)}.")

    return results

def get_median_runtimes(runtime_dict):
    median_runtimes = {}
    for input_name, runtimes in runtime_dict.items():
        if runtimes:
            median_runtimes[input_name] = statistics.median(runtimes)
    return median_runtimes

def calculate_speedup(old_runtimes, new_runtimes):
    speedup = {}
    for input_name in old_runtimes:
        if input_name in new_runtimes:
            old_time = old_runtimes[input_name]
            new_time = new_runtimes[input_name]
            if new_time != 0:  # Avoid division by zero
                speedup[input_name] = old_time / new_time
            else:
                print(f'ERROR: Race-free time for {input_name} is 0')
        else:
            print(f'ERROR: Missing race-free runtime for {input_name}')
    return speedup

def process_directory(directory, algorithms):
    speedups = {}

    for algo in algorithms:
        #print(f'Processing {algo}')
        baseline_file = os.path.join(directory, f'baseline_{algo}.txt')
        racefree_file = os.path.join(directory, f'racefree_{algo}.txt')

        if not os.path.exists(baseline_file):
            raise FileNotFoundError(f"Required file for {algo} is missing: {baseline_file}")
        if not os.path.exists(racefree_file):
            raise FileNotFoundError(f"Required file for {algo} is missing: {racefree_file}")

        # Parse the files and calculate median runtimes
        baseline_runtimes = get_median_runtimes(parse_runtime_file(baseline_file))
        racefree_runtimes = get_median_runtimes(parse_runtime_file(racefree_file))

        # Calculate speedups
        algo_speedups = calculate_speedup(baseline_runtimes, racefree_runtimes)
        speedups[algo] = algo_speedups

    return speedups

def process_directory_egr(directory):
  algorithms = ['CC', 'GC', 'MIS', 'MST']
  return process_directory(directory, algorithms)

def process_directory_mesh(directory):
  algorithms = ['SCC']
  return process_directory(directory, algorithms)

def write_speedups_to_csv(speedups, filename, headers):
    # Collect all input names
    all_inputs = set()
    for algo_speedups in speedups.values():
        all_inputs.update(algo_speedups.keys())

    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)

        for input_name in sorted(all_inputs):
            display_name = input_name
            if display_name in display_names.keys():
              display_name = display_names[input_name]
            row = [display_name]
            for algo in headers[1:]:
                if algo in speedups and input_name in speedups[algo]:
                    row.append(speedups[algo][input_name])
                else:
                    row.append(None)  # Fill with None if data is missing
            csvwriter.writerow(row)

        # Add Min Speedup row
        row = ["Min Speedup"]
        for algo in headers[1:]:
            if algo in speedups and input_name in speedups[algo]:
                minimum = min(speedups[algo].values())
                row.append(minimum)
            else:
                row.append(None)
        csvwriter.writerow(row)
        
        # Add Geomean Speedup row
        row = ["Geomean Speedup"]
        for algo in headers[1:]:
            if algo in speedups and input_name in speedups[algo]:
                geomean = gmean(list(speedups[algo].values()))
                row.append(geomean)
            else:
                row.append(None)  # Fill with None if data is missing
        csvwriter.writerow(row)
        
        # Add Max Speedup row
        row = ["Max Speedup"]
        for algo in headers[1:]:
            if algo in speedups and input_name in speedups[algo]:
                maximum = max(speedups[algo].values())
                row.append(maximum)
            else:
                row.append(None)
        csvwriter.writerow(row)


def write_speedups_to_csv_egr(speedups, filename):
  headers = ['Input', 'CC', 'GC', 'MIS', 'MST']
  write_speedups_to_csv(speedups, filename, headers)

def write_speedups_to_csv_mesh(speedups, filename):
  headers = ['Input', 'SCC']
  write_speedups_to_csv(speedups, filename, headers)

# Generate tables
egr_speedups = process_directory_egr(results_dir)
mesh_speedups = process_directory_mesh(results_dir)

filename = os.path.join(output_dir, "undirected_speedups.csv")
print(f'Writing speedup table {filename}...')
write_speedups_to_csv_egr(egr_speedups, filename)

filename = os.path.join(output_dir, "directed_speedups.csv")
print(f'Writing speedup table {filename}...')
write_speedups_to_csv_mesh(mesh_speedups, filename)

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print("Matplotlib not available, skipping graph output.")
    exit(0)

def plot_geometric_mean_speedups(all_egr_speedups, all_mesh_speedups):
    algorithms = list(all_egr_speedups.keys())
    if 'SCC' in all_mesh_speedups:
        algorithms.append('SCC')

    num_algorithms = len(algorithms)

    # Initialize a list to hold geometric mean speedups for each algorithm
    geometric_mean_speedups = []

    for algo in algorithms:
        if algo == 'SCC':
            gm_speedup = gmean(list(all_mesh_speedups[algo].values()))
        else:
            gm_speedup = gmean(list(all_egr_speedups[algo].values()))
        geometric_mean_speedups.append(gm_speedup)

    fig, ax = plt.subplots(figsize=(12, 8))  # Increase figure size

    bar_container = ax.bar(algorithms, geometric_mean_speedups)
    plt.rcParams['font.size'] = 18
    ax.bar_label(bar_container, fmt='{:,.2f}')

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1)  # Add a line at y=1.0

    ax.set_ylabel('Geometric Mean Speedup', fontsize=22, labelpad=6.0)
    plt.xticks(fontsize=22)

    # Add more y-ticks
    start, end = ax.get_ylim()
    plt.yticks(np.arange(start, end, 0.1), fontsize=18)
    plt.grid(visible=True, axis='y', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "geometric_means_bar.svg"))  # Save figure

print(f'Writing speedup bar chart figure {os.path.join(output_dir, "geometric_means_bar.svg")}...')
plot_geometric_mean_speedups(egr_speedups, mesh_speedups)