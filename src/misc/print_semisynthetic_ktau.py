import pandas as pd
from pathlib import Path
import numpy as np
import config as cfg
from utility.data import get_dataset_info, map_dataset_name, map_strategy_name
from utility.survival import theta_to_kendall_tau

N_DECIMALS = 3
SIGMA_LEVEL = 2

if __name__ == "__main__":
    results = pd.read_csv(Path.joinpath(cfg.RESULTS_DIR, "semisynthetic_results.csv"))
    
    # Clayton or Frank copula
    # CI: CIHarrell, CIUno, CIDepIPCW
    # IBS/MAE: IBSIPCW, IBSBG, IBSDepBG, MAEHinge, MAEPseudo, MAEMargin, MAEDepBG
    #metrics = ["CIHarrell", "CIUno", "CIDepIPCW"]
    metrics = ["IBSIPCW", "IBSBG", "IBSDepBG", "MAEHinge", "MAEPseudo", "MAEMargin", "MAEDepBG"]
    
    datasets = ["metabric", "mimic_all", "mimic_hospital", "seer_brain", "seer_liver", "seer_stomach"]
    strategies = ["original", "top_5", "top_10", "random_25"]
    model_names = ["coxph", "gbsa", "rsf", "deepsurv", "mtlr"]

for idx, dataset in enumerate(datasets):
    n_samples, censoring_rate = get_dataset_info(dataset)
    print(r"\multirow{4}{*}{\makecell{" + f"{map_dataset_name(dataset)} \\\ ($N$={n_samples}, $C$={censoring_rate}\%)" + r"}}")
    for strategy in strategies:
        data = results.loc[(results['Dataset'] == dataset) & (results['Strategy'] == strategy)]
        most_common_copula = data['BestCopulaName'].mode()[0]
        mean_theta = data[data['BestCopulaName'] == most_common_copula]['BestCopulaTheta'].mean()
        k_tau = round(theta_to_kendall_tau(most_common_copula, mean_theta), 2)
        
        print(dataset)
        print(most_common_copula)
        print(k_tau)
