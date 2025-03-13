#!/bin/bash

base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

results_path=$base_path/results/semisynthetic_results.csv
if [ -f "$results_path" ]; then
  rm $results_path
fi

seeds=(0 1 2 3 4 5 6 7 8 9)
dataset_names=("metabric" "mimic_all" "mimic_hospital" "seer_brain" "seer_liver" "seer_stomach")
strategies=('original' 'top_5' 'top_10' 'random_25')

for seed in "${seeds[@]}"; do
    for dataset_name in "${dataset_names[@]}"; do
        for strategy in "${strategies[@]}"; do
            echo "Running with seed=$seed, dataset_name=$dataset_name, strategy=$strategy"
            python3 $base_path/src/experiments/train_semisynthetic_datasets.py --seed "$seed" --dataset_name "$dataset_name" --strategy "$strategy"
        done
    done
done
