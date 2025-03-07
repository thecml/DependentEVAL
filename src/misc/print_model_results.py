import pandas as pd
from pathlib import Path
import numpy as np
import config as cfg
from utility.data import get_dataset_info, map_dataset_name, map_strategy_name

N_DECIMALS = 2
                
def calculate_errors(results, dataset, strategy, model_names, metrics):
    true_metrics = {f"{metric}True": metric for metric in ["CI", "IBS", "MAE"]}
    
    # Initialize error storage
    mean_errors = {metric: [] for metric in metrics}
    std_errors = {metric: [] for metric in metrics}

    for model_name in model_names:
        # Extract true metrics
        true_values = {
            true_metric: results.loc[
                (results["Dataset"] == dataset) &
                (results["Strategy"] == strategy) &
                (results["ModelName"] == model_name), true_metric
            ].values for true_metric in true_metrics
        }

        # Extract predicted metrics
        for metric in metrics:
            predicted_values = results.loc[
                (results["Dataset"] == dataset) &
                (results["Strategy"] == strategy) &
                (results["ModelName"] == model_name), metric
            ].values
            
            # Match true metric and calculate errors
            true_metric_key = next((k for k, v in true_metrics.items() if metric.startswith(v)), None)
            if true_metric_key:
                true_values_for_metric = true_values[true_metric_key]
                errors = abs(true_values_for_metric - predicted_values)
                mean_errors[metric].append(np.mean(errors))
                std_errors[metric].append(np.std(errors))
    
    # Aggregate mean/std errors across models
    mean_errors = {k: np.nanmean(v) for k, v in mean_errors.items()}
    std_errors = {k: np.nanmean(v) for k, v in std_errors.items()}

    return mean_errors, std_errors

if __name__ == "__main__":
    results = pd.read_csv(Path.joinpath(cfg.RESULTS_DIR, "semisynthetic_results.csv"))
    metrics = ["CIHarrell", "CIUno", "CIDepIPCW", "IBSIPCW", "IBSDepBG",
               "MAEUncens", "MAEHinge", "MAEPseudo", "MAEMargin", "MAEDepBG"]
    
    # Scale metrics by percentage
    cols_to_scale = ["CITrue", "CIHarrell", "CIUno", "CIDepIPCW", "IBSTrue", "IBSIPCW", "IBSDepBG"]
    results[cols_to_scale] = results[cols_to_scale] * 100

    datasets = ["metabric", "mimic_all", "mimic_hospital", "seer_brain", "seer_liver", "seer_stomach"]
    strategies = ["original", "top_5", "top_10", "random_25"]
    model_names = ["coxph", "gbsa", "rsf", "deepsurv", "mtlr"]

for idx, dataset in enumerate(datasets):
    n_samples, censoring_rate = get_dataset_info(dataset)
    print(r"\multirow{4}{*}{\makecell{" + f"{map_dataset_name(dataset)} \\\ ($N$={n_samples}, $C$={censoring_rate}\%)" + r"}}")
    for strategy in strategies:
        mean_errors, std_errors = calculate_errors(results, dataset, strategy, model_names, metrics)

        # Format for printing
        formatted_errors = {
            k: f"%.{N_DECIMALS}f" % round(v, N_DECIMALS)
            for k, v in mean_errors.items()
        }
        formatted_std_errors = {
            k: f"%.{N_DECIMALS}f" % round(v, N_DECIMALS)
            for k, v in std_errors.items()
        }
        
        # Construct the text with mean and std errors
        text = f"& {map_strategy_name(strategy)}" + \
            "".join(f" & {formatted_errors[metric]}$\pm$\\scriptsize" + r"{" + f"{formatted_std_errors[metric]}" + r"}" for metric in mean_errors) + " \\\\"
        print(text)
    
    if idx != len(datasets) - 1:
        print(r"\cmidrule(lr){1-1}")
