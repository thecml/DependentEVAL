#!/bin/bash
set -euo pipefail

base_path="$(cd "$(dirname "$0")" && pwd)"
echo "$base_path"

results_path="$base_path/../results"
mkdir -p "$results_path"

# Remove only synthetic outputs
rm -f "$results_path"/synthetic_*.csv

# Synthetic datasets
python3 "$base_path/../src/experiments/train_synthetic_censoring.py"
python3 "$base_path/../src/experiments/train_synthetic_correct_copula.py"
python3 "$base_path/../src/experiments/train_synthetic_correct_copula_cgq.py"
python3 "$base_path/../src/experiments/train_synthetic_wrong_copula.py"