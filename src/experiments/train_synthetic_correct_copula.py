import argparse
import random
import torch
from data_loader import SingleEventSyntheticDataLoader
import pandas as pd
import numpy as np
import config as cfg
from SurvivalEVAL import SurvivalEvaluator
from sksurv.linear_model import CoxPHSurvivalAnalysis
from SurvivalEVAL.Evaluations.util import predict_median_survival_time
from scipy.interpolate import interp1d

from metrics import DependentEvaluator
from sota.sksurv import make_cox_model
from utility.data import dotdict
from utility.survival import (convert_to_structured, kendall_tau_to_theta,
                              make_stratified_split, make_time_bins)

from sksurv.metrics import concordance_index_ipcw

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

dtype = torch.float64
torch.set_default_dtype(dtype)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_cfg = {
    "alpha_e1": 19,
    "alpha_e2": 17,
    "gamma_e1": 6,
    "gamma_e2": 4,
    "n_samples": 1000,
    "n_features": 10,
}

SEEDS = list(range(0, 100))
COPULA_NAMES = ["clayton", "frank"]
K_TAU = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
DATA = [(10000, 10)]
LINEAR = True

if __name__ == "__main__":
    results_dict = {}
    for seed in SEEDS:
        for copula_name in COPULA_NAMES:
            for k_tau in K_TAU:
                for sample_cfg in DATA:
                    n_samples = sample_cfg[0]
                    n_features = sample_cfg[1]
                    print(f"Running: {seed}, {copula_name}, {k_tau}, {n_samples}, {n_features}")
                    
                    results_dict[(seed, copula_name, k_tau, n_samples, n_features)] = {} # tuple as key
                    
                    data_cfg['n_samples'] = n_samples
                    data_cfg['n_features'] = n_features
                    
                    # Load data
                    dl = SingleEventSyntheticDataLoader().load_data(data_cfg, k_tau=k_tau, copula_name=copula_name,
                                                                    linear=LINEAR, device=device, dtype=dtype)
                    df = dl.get_data()
                    df['true_time'] = dl.true_event_times
                    df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='time', frac_train=0.7,
                                                                        frac_valid=0.1, frac_test=0.2,
                                                                        random_state=0)
            
                    # Process data
                    data_train = df_train.drop(columns=["true_time"])
                    true_test_time = df_test.true_time.values
                    true_test_event = np.ones(df_test.shape[0])
                    data_valid = df_valid.drop(columns=["true_time"])
                    data_test = df_test.drop(columns=["true_time"])
                    X_train = data_train.drop(columns=["time", "event"])
                    X_valid = data_valid.drop(columns=["time", "event"])
                    X_test = data_test.drop(columns=["time", "event"])
                    
                    # Format data
                    y_train = convert_to_structured(df_train['time'], df_train['event'])
                    y_valid = convert_to_structured(df_valid['time'], df_valid['event'])
                    y_test = convert_to_structured(df_test['time'], df_test['event'])
            
                    # Make time bins
                    time_bins = make_time_bins(y_train['time'], event=y_train['event'], dtype=dtype).to(device)
                    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins)).cpu().numpy()
            
                    # Train Cox model
                    config = dotdict(cfg.COXPH_PARAMS)
                    model = make_cox_model(config)
                    model.fit(X_train, y_train)
                    survival_outputs = model.predict_survival_function(X_test)
                    survival_outputs = np.row_stack([fn(model.unique_times_) for fn in survival_outputs])
                    spline = interp1d(model.unique_times_,
                                    survival_outputs,
                                    kind='linear', fill_value='extrapolate')
                    survival_outputs = spline(time_bins)
            
                    # Make dataframe
                    survival_outputs = pd.DataFrame(survival_outputs, columns=time_bins)
                    survival_outputs[0] = 1
            
                    # Create true evaluator to calculate true metrics
                    true_evaluator = SurvivalEvaluator(survival_outputs, time_bins, true_test_time, true_test_event)
                    ci_true = true_evaluator.concordance()[0]
                    ibs_true = true_evaluator.integrated_brier_score(IPCW_weighted=False, num_points=10)
                    mae_true = true_evaluator.mae(method="Uncensored")
            
                    # Calculate censored metrics
                    censored_evaluator = SurvivalEvaluator(survival_outputs, time_bins, data_test.time.values, data_test.event.values,
                                                           data_train.time.values, data_train.event.values)
                    ci_harrell = censored_evaluator.concordance()[0]
                    predicted_times = censored_evaluator.predict_time_from_curve(predict_median_survival_time)
                    risks = -1 * predicted_times
                    ci_uno = concordance_index_ipcw(y_train, y_test, risks, tau=y_train['time'].max())[0]
                    ibs_ipcw = censored_evaluator.integrated_brier_score(num_points=10)
                    mae_margin = censored_evaluator.mae(method="Margin", weighted=True)
                    
                    # Calculate dependent metrics
                    theta = kendall_tau_to_theta(copula_name, k_tau)
                    dep_evaluator = DependentEvaluator(survival_outputs, time_bins, data_test.time.values, data_test.event.values,
                                                       data_train.time.values, data_train.event.values, copula_name=copula_name,
                                                       alpha=theta)
                    ci_dep_ipcw = dep_evaluator.concordance(method="IPCW")[0]
                    ibs_dep_bg = dep_evaluator.integrated_brier_score(method="BG", num_points=10)
                    mae_dep_bg = dep_evaluator.mae(method="BG")
                    mae_dep_ipcw = dep_evaluator.mae(method="IPCW")
                    
                    # Calculate errors
                    ci_harrell_error = abs(ci_true - ci_harrell)
                    ci_dep_ipcw_error = abs(ci_true - ci_dep_ipcw)
                    ci_uno_error = abs(ci_true - ci_uno)
                    ibs_ipcw_error = abs(ibs_true - ibs_ipcw)
                    ibs_dep_bg_error = abs(ibs_true - ibs_dep_bg)
                    mae_margin_error = abs(mae_true - mae_margin)
                    mae_dep_bg_error = abs(mae_true - mae_dep_bg)
                    mae_dep_ipcw_error = abs(mae_true - mae_dep_ipcw)
                    
                    # Store results in the dictionary
                    results_dict[(seed, copula_name, k_tau, n_samples, n_features)] = {
                        "ci_harrell_error": ci_harrell_error,
                        "ci_dep_ipcw_error": ci_dep_ipcw_error,
                        "ci_uno_error": ci_uno_error,
                        "ibs_ipcw_error": ibs_ipcw_error,
                        "ibs_dep_bg_error": ibs_dep_bg_error,
                        "mae_margin_error": mae_margin_error,
                        "mae_dep_bg_error": mae_dep_bg_error,
                        "mae_dep_ipcw_error": mae_dep_ipcw_error
                    }
                    
    # Flatten the nested dictionary into a list of rows
    flattened_results = []
    for (seed, copula_name, k_tau, n_samples, n_features), metrics in results_dict.items():
        flattened_results.append({
            "seed": seed,
            "copula_name": copula_name,
            "k_tau": k_tau,
            "n_samples": n_samples,
            "n_features": n_features,
            **metrics
        })
            
    results_df = pd.DataFrame(flattened_results)

    # Save results to a CSV file
    filename = f"{cfg.RESULTS_DIR}/synthetic_results_correct_copula.csv"
    results_df.to_csv(filename, index=False)
        
        