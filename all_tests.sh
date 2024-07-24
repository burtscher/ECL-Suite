#!/bin/bash
# This will run all experiments, place runtime results in results/, place speedup CSVs and a figure in output/

export CUDA_VISIBLE_DEVICES=0   # Specifies which GPU is used; 0 is the fastest GPU when CUDA_DEVICE_ORDER is unset
runs=9                          # Specifies how many times to run each input on each code; median runtime taken
undir_input_dir=inputs-undirected
dir_input_dir=inputs-directed
result_dir=results

if [ ! -d "$undir_input_dir" ]; then
  echo "$undir_input_dir/ does not exist, run ./download_inputs.sh first"
  exit 1
fi
if [ ! -d "$dir_input_dir" ]; then
  echo "$dir_input_dir/ does not exist, run ./download_inputs.sh first"
  exit 1
fi

mkdir -p $result_dir
ulimit -s unlimited # for CC's recursive verifier

# Baseline code runs
echo "Running baseline CC..."
python3 ./scripts/multi_run.py src/baseline/egr-input/CC $undir_input_dir/ ./library/ $runs >> $result_dir/baseline_CC.txt
echo "Running baseline GC..."
python3 ./scripts/multi_run.py src/baseline/egr-input/GC $undir_input_dir/ ./library/ $runs >> $result_dir/baseline_GC.txt
echo "Running baseline MIS..."
python3 ./scripts/multi_run.py src/baseline/egr-input/MIS $undir_input_dir/ ./library/ $runs >> $result_dir/baseline_MIS.txt
echo "Running baseline MST..."
python3 ./scripts/multi_run.py src/baseline/egr-input/MST $undir_input_dir/ ./library/ $runs >> $result_dir/baseline_MST.txt
echo "Running baseline SCC..."
python3 ./scripts/multi_run.py src/baseline/mesh-input/ $dir_input_dir/ ./library/ $runs >> $result_dir/baseline_SCC.txt

# Racefree code runs
echo "Running race-free CC..."
python3 ./scripts/multi_run.py src/racefree/egr-input/CC $undir_input_dir/ ./library/ $runs >> $result_dir/racefree_CC.txt
echo "Running race-free GC..."
python3 ./scripts/multi_run.py src/racefree/egr-input/GC $undir_input_dir/ ./library/ $runs >> $result_dir/racefree_GC.txt
echo "Running race-free MIS..."
python3 ./scripts/multi_run.py src/racefree/egr-input/MIS $undir_input_dir/ ./library/ $runs >> $result_dir/racefree_MIS.txt
echo "Running race-free MST..."
python3 ./scripts/multi_run.py src/racefree/egr-input/MST $undir_input_dir/ ./library/ $runs >> $result_dir/racefree_MST.txt
echo "Running race-free SCC..."
python3 ./scripts/multi_run.py src/racefree/mesh-input/ $dir_input_dir/ ./library/ $runs >> $result_dir/racefree_SCC.txt

echo "Calculating speedups..."
python3 ./scripts/analyze_results.py
