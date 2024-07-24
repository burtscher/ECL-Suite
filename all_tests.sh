#!/bin/bash
# This will run all experiments, place runtime results in results/, place speedup CSVs and a figure in output/

export CUDA_VISIBLE_DEVICES=0   # Specifies which GPU is used; 0 is the fastest GPU when CUDA_DEVICE_ORDER is unset
runs=9                          # Specifies how many times to run each input on each code; median runtime taken
undir_input_dir=inputs-undirected
dir_input_dir=inputs-directed
result_dir=results

mkdir -p $result_dir
ulimit -s unlimited # for CC's recursive verifier

# Baseline code runs
python3 ./scripts/multi_run.py src/baseline/egr-input/CC $undir_input_dir/ ./library/ $runs >> $result_dir/baseline_CC.txt
python3 ./scripts/multi_run.py src/baseline/egr-input/GC $undir_input_dir/ ./library/ $runs >> $result_dir/baseline_GC.txt
python3 ./scripts/multi_run.py src/baseline/egr-input/MIS $undir_input_dir/ ./library/ $runs >> $result_dir/baseline_MIS.txt
python3 ./scripts/multi_run.py src/baseline/egr-input/MST $undir_input_dir/ ./library/ $runs >> $result_dir/baseline_MST.txt
python3 ./scripts/multi_run.py src/baseline/mesh-input/ $dir_input_dir/ ./library/ $runs >> $result_dir/baseline_SCC.txt

# Racefree code runs
python3 ./scripts/multi_run.py src/racefree/egr-input/CC $undir_input_dir/ ./library/ $runs >> $result_dir/racefree_CC.txt
python3 ./scripts/multi_run.py src/racefree/egr-input/GC $undir_input_dir/ ./library/ $runs >> $result_dir/racefree_GC.txt
python3 ./scripts/multi_run.py src/racefree/egr-input/MIS $undir_input_dir/ ./library/ $runs >> $result_dir/racefree_MIS.txt
python3 ./scripts/multi_run.py src/racefree/egr-input/MST $undir_input_dir/ ./library/ $runs >> $result_dir/racefree_MST.txt
python3 ./scripts/multi_run.py src/racefree/mesh-input/ $dir_input_dir/ ./library/ $runs >> $result_dir/racefree_SCC.txt

python3 ./scripts/analyze_results.py
