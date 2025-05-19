import pandas as pd
from pathlib import Path
import numpy as np
import config as cfg
from utility.data import get_dataset_info, map_dataset_name, map_strategy_name
from utility.survival import theta_to_kendall_tau

N_DECIMALS = 3
SIGMA_LEVEL = 2
                
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
                std_errors[metric].append(SIGMA_LEVEL * np.std(errors))
    
    # Aggregate mean/std errors across models
    mean_errors = {k: np.nanmean(v) for k, v in mean_errors.items()}
    std_errors = {k: np.nanmean(v) for k, v in std_errors.items()}

    return mean_errors, std_errors

if __name__ == "__main__":
    results = pd.read_csv(Path.joinpath(cfg.RESULTS_DIR, "semisynthetic_results.csv"))
    
    # Clayton or Frank copula
    # CI: CIHarrell, CIUno, CIDepIPCW
    # IBS/MAE: IBSIPCW, IBSBG, IBSDepBG, MAEHinge, MAEPseudo, MAEMargin, MAEDepBG
    #metrics = ["CIHarrell", "CIUno", "CIDepIPCW"]
    metrics = ["IBSIPCW", "IBSBG", "IBSDepBG", "MAEHinge", "MAEPseudo", "MAEMargin", "MAEDepBG"]
    
    datasets = ["metabric", "mimic_all", "seer_liver"]
    #datasets = ["metabric", "mimic_all", "mimic_hospital", "seer_brain", "seer_liver", "seer_stomach"]
    strategies = ["original", "top_5", "top_10", "random_25"]
    model_names = ["coxph", "gbsa", "rsf", "deepsurv", "mtlr"]

for idx, dataset in enumerate(datasets):
    n_samples, censoring_rate = get_dataset_info(dataset)
    print(r"\multirow{4}{*}{\makecell{" + f"{map_dataset_name(dataset)} \\\ ($N$={n_samples}, $C$={censoring_rate}\%)" + r"}}")
    for strategy in strategies:
        mean_errors, std_errors = calculate_errors(results, dataset, strategy, model_names, metrics)
        
        data = results.loc[(results['Dataset'] == dataset) & (results['Strategy'] == strategy)]
        most_common_copula = data['BestCopulaName'].mode()[0]
        mean_theta = data[data['BestCopulaName'] == most_common_copula]['BestCopulaTheta'].mean()
        k_tau = round(theta_to_kendall_tau(most_common_copula, mean_theta), 2)

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
