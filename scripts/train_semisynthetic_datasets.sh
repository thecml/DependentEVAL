#!/bin/bash
set -euo pipefail

base_path="$(cd "$(dirname "$0")" && pwd)"
echo "$base_path"

results_path=$base_path/../results/semisynthetic_results.csv
if [ -f "$results_path" ]; then
  rm $results_path
fi

seeds=({0..9})
dataset_names=("metabric" "gbsg" "nacd" "support" "flchain" "whas" "employee" "churn" "mimic_all" "seer_brain" "seer_liver" "seer_stomach")
strategies=('original' 'top_5' 'top_10' 'random_25')

for seed in "${seeds[@]}"; do
    for dataset_name in "${dataset_names[@]}"; do
        for strategy in "${strategies[@]}"; do
            echo "Running with seed=$seed, dataset_name=$dataset_name, strategy=$strategy"
            python3 $base_path/../src/experiments/train_semisynthetic_datasets.py --seed "$seed" --dataset_name "$dataset_name" --strategy "$strategy"
        done
    done
done
