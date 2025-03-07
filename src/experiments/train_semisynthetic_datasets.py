import argparse
import os
import random
import torch
from copula import Clayton_Bivariate, Frank_Bivariate
from data_loader import get_data_loader
import pandas as pd
import numpy as np
import config as cfg
from metrics import DependentEvaluator
from sota.deepsurv import DeepSurv, make_deepsurv_prediction, train_deepsurv_model
from sota.mtlr import make_mtlr_prediction, mtlr, train_mtlr_model
from sota.sksurv import make_cox_model, make_gbsa_model, make_rsf_model
from utility.data import dotdict, fix_types
from SurvivalEVAL import SurvivalEvaluator
from SurvivalEVAL.Evaluations.util import predict_median_survival_time
from scipy.interpolate import interp1d

from models import Weibull_log_linear, Weibull_nonlinear
from strategies import combine_data_with_censor, make_synthetic_censoring
from utility.preprocessor import Preprocessor
from utility.survival import convert_to_structured, make_stratified_split, make_time_bins
from trainer import train_copula_model

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import concordance_index_ipcw

import time

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

dtype = torch.float64
torch.set_default_dtype(dtype)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS = ["coxph", "gbsa", "rsf", "deepsurv", "mtlr"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='metabric')
    parser.add_argument('--strategy', type=str, default='original')
    
    args = parser.parse_args()
    seed = args.seed
    dataset_name = args.dataset_name
    strategy = args.strategy
    
    # Load data
    dl = get_data_loader(dataset_name).load_data()
    df_full = dl.get_data().reset_index(drop=True)
    num_features, cat_features = dl.get_features()
    
    # Preprocess full dataset
    preprocessor = Preprocessor(cat_feat_strat='mode', num_feat_strat='mean', scaling_strategy="minmax")
    transformer = preprocessor.fit(df_full.drop(['time', 'event'], axis=1),
                                   cat_feats=cat_features, num_feats=num_features,
                                   one_hot=True, fill_value=-1)
    X = transformer.transform(df_full.drop(['time', 'event'], axis=1)).reset_index(drop=True)
    df_full = pd.concat([X, df_full[['time', 'event']]], axis=1)
    
    # Drop censored rows
    df = df_full.drop(df_full[df_full.event == 0].index)
    df.reset_index(drop=True, inplace=True)
    df.time = df.time.round().astype(int)
    
    # Make synthetic censoring time
    censor_times, selected_features = make_synthetic_censoring(strategy, df, df_full)
    censor_times = np.round(censor_times).astype(int)
    
    # Combine truth and censored data to make semi-synthetic dataset
    df = combine_data_with_censor(df, censor_times, selected_features)
    
    # Split data
    df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='both', frac_train=0.7,
                                                        frac_valid=0.1, frac_test=0.2,
                                                        random_state=seed)
    
    # Adjust types
    df_train, df_valid, df_test = fix_types(df_train, df_valid, df_test)
  
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
    train_dict, valid_dict, test_dict = dict(), dict(), dict()
    train_dict['X'] = torch.tensor(X_train.values, device=device, dtype=dtype)
    train_dict['T'] = torch.tensor(data_train['time'].values, device=device, dtype=dtype)
    train_dict['E'] = torch.tensor(data_train['event'].values, device=device, dtype=dtype)
    valid_dict['X'] = torch.tensor(X_valid.values, device=device, dtype=dtype)
    valid_dict['T'] = torch.tensor(data_valid['time'].values, device=device, dtype=dtype)
    valid_dict['E'] = torch.tensor(data_valid['event'].values, device=device, dtype=dtype)
    test_dict['X'] = torch.tensor(X_test.values, device=device, dtype=dtype)
    test_dict['T'] = torch.tensor(data_test['time'].values, device=device, dtype=dtype)
    test_dict['E'] = torch.tensor(data_test['event'].values, device=device, dtype=dtype)
    n_samples = train_dict['X'].shape[0]
    n_features = train_dict['X'].shape[1]
    X_train = pd.DataFrame(train_dict['X'].cpu().numpy(), columns=[f'X{i}' for i in range(n_features)])
    X_valid = pd.DataFrame(valid_dict['X'].cpu().numpy(), columns=[f'X{i}' for i in range(n_features)])
    X_test = pd.DataFrame(test_dict['X'].cpu().numpy(), columns=[f'X{i}' for i in range(n_features)])
    y_train = convert_to_structured(train_dict['T'].cpu().numpy(), train_dict['E'].cpu().numpy())
    y_valid = convert_to_structured(valid_dict['T'].cpu().numpy(), valid_dict['E'].cpu().numpy())
    y_test = convert_to_structured(test_dict['T'].cpu().numpy(), test_dict['E'].cpu().numpy())
    
    # Make time bins
    time_bins = make_time_bins(train_dict['T'].cpu(), event=train_dict['E'].cpu(), dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))
    
    # Estimate theta on the new dataset and find the best copula
    results_list = []
    for copula_name in ["clayton", "frank"]:
        # Reset seeds
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        random.seed(0)
        
        torch.cuda.empty_cache() # empty cache
        
        dep_model1 = Weibull_nonlinear(n_features, dtype=dtype, device=device) # censoring model
        dep_model2 = Weibull_nonlinear(n_features, dtype=dtype, device=device) # event model
        if copula_name == "clayton":
            copula = Clayton_Bivariate(2.0, 1e-4, dtype=dtype, device=device)
        elif copula_name == "frank":
            copula = Frank_Bivariate(2.0, 1e-4, dtype=dtype, device=device)
        dep_model1, dep_model2, copula, min_val_loss = train_copula_model(dep_model1, dep_model2, train_dict,
                                                                          valid_dict, copula=copula, n_epochs=10000,
                                                                          patience=100, lr=0.001, batch_size=1024,
                                                                          copula_name=copula_name, verbose=True)
        copula_theta = float(copula.parameters()[0][0])
        k = sum(param.numel() for param in dep_model1.parameters())
        k += sum(param.numel() for param in dep_model2.parameters())
        k += sum(param.numel() for param in copula.parameters())
        results_list.append({'copula_name': copula_name, 'copula_theta': copula_theta,
                             'min_val_loss': min_val_loss, 'num_params': k})
    results_df = pd.DataFrame(results_list)
    results_df['AIC'] = 2*results_df['num_params'] + 2*results_df['min_val_loss'] # AIC
    
    # Filter for copulas that capture dependence (theta > 0.001)
    threshold = 1e-3
    valid_copulas = results_df[(results_df['copula_theta'].abs() >= threshold) & results_df['copula_theta'].notna()]

    # Select the copula with the lowest AIC among valid copulas
    if not valid_copulas.empty:
        best_idx = valid_copulas['AIC'].idxmin()
        best_copula_name = valid_copulas.loc[best_idx, 'copula_name']
        best_copula_theta = valid_copulas.loc[best_idx, 'copula_theta']
    else:
        # No dependence found, assume independent copula
        best_copula_name = "clayton"
        best_copula_theta = 0.001
    
    for model_name in MODELS:
        # Reset seeds
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        random.seed(0)
        
        print(f"Started training model {model_name}")
        start_time = time.time()
        
        # Train base learners
        if model_name == "coxph":
            config = dotdict(cfg.COXPH_PARAMS)
            model = make_cox_model(config)
            model.fit(X_train, y_train)
        elif model_name == "gbsa":
            config = dotdict(cfg.GBSA_PARAMS)
            model = make_gbsa_model(config)
            model.fit(X_train, y_train)
        elif model_name == "rsf":
            config = dotdict(cfg.RSF_PARAMS)
            model = make_rsf_model(config)
            model.fit(X_train, y_train)
        elif model_name == "deepsurv":
            config = dotdict(cfg.DEEPSURV_PARAMS)
            model = DeepSurv(in_features=n_features, config=config)
            data_train = pd.DataFrame(train_dict['X'].cpu().numpy())
            data_train['time'] = train_dict['T'].cpu().numpy()
            data_train['event'] = train_dict['E'].cpu().numpy()
            data_valid = pd.DataFrame(valid_dict['X'].cpu().numpy())
            data_valid['time'] = valid_dict['T'].cpu().numpy()
            data_valid['event'] = valid_dict['E'].cpu().numpy()
            model = train_deepsurv_model(model, data_train, data_valid, time_bins, config=config,
                                         random_state=0, reset_model=True, device=device, dtype=dtype)
        elif model_name == "mtlr":
            data_train = X_train.copy()
            data_train["time"] = pd.Series(y_train['time'])
            data_train["event"] = pd.Series(y_train['event']).astype(int)
            data_valid = X_valid.copy()
            data_valid["time"] = pd.Series(y_valid['time'])
            data_valid["event"] = pd.Series(y_valid['event']).astype(int)
            config = dotdict(cfg.MTLR_PARAMS)
            num_time_bins = len(time_bins)
            model = mtlr(in_features=n_features, num_time_bins=num_time_bins, config=config)
            model = train_mtlr_model(model, data_train, data_valid, time_bins.cpu().numpy(),
                                     config, random_state=0, dtype=dtype,
                                     reset_model=True, device=device)
        else:
            raise NotImplementedError()
        end_time = time.time()
        elapsed_time = end_time - start_time
        end_time = time.time()
        
        print(f"Training time for model {model_name}: {elapsed_time:.2f} seconds")
        
        # Compute survival function
        if model_name in ["coxph", "gbsa", "rsf"]:
            survival_outputs = model.predict_survival_function(X_test)
            survival_outputs = np.row_stack([fn(time_bins.cpu().numpy()) for fn in survival_outputs])
        elif model_name == "deepsurv":
            survival_outputs, time_bins_deepsurv = make_deepsurv_prediction(model, test_dict['X'].to(device),
                                                                            config=config, dtype=dtype)
            spline = interp1d(time_bins_deepsurv.cpu().numpy(),
                              survival_outputs.cpu().numpy(),
                              kind='linear', fill_value='extrapolate')
            survival_outputs = spline(time_bins.cpu().numpy())
        elif model_name == "mtlr":
            data_test = X_test.copy()
            data_test["time"] = pd.Series(y_test['time'])
            data_test["event"] = pd.Series(y_test['event']).astype('int')
            mtlr_test_data = torch.tensor(data_test.drop(["time", "event"], axis=1).values,
                                          dtype=dtype, device=device)
            survival_outputs, _, _ = make_mtlr_prediction(model, mtlr_test_data, time_bins, config)
            survival_outputs = survival_outputs[:, 1:].cpu().numpy()
        else:
            raise NotImplementedError()
        
        # Make dataframe
        survival_outputs = pd.DataFrame(survival_outputs, columns=time_bins.cpu().numpy())
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
        mae_uncensored = censored_evaluator.mae(method="Uncensored")
        mae_hinge = censored_evaluator.mae(method="Hinge")
        mae_margin = censored_evaluator.mae(method="Margin", weighted=True)
        mae_ipcwv1 = censored_evaluator.mae(method="IPCW-v1", weighted=True)
        mae_ipcwv2 = censored_evaluator.mae(method="IPCW-v2", weighted=True)
        mae_pseudo = censored_evaluator.mae(method="Pseudo_obs", weighted=True)

        # Calculate dependent metrics
        dep_evaluator = DependentEvaluator(survival_outputs, time_bins, data_test.time.values, data_test.event.values,
                                           data_train.time.values, data_train.event.values, copula_name=best_copula_name,
                                           alpha=best_copula_theta)
        ci_dep_ipcw = dep_evaluator.concordance(method="IPCW")[0]
        ibs_dep_bg = dep_evaluator.integrated_brier_score(method="BG", num_points=10)
        ibs_dep_ipcw = dep_evaluator.integrated_brier_score(method="IPCW", num_points=10)
        mae_dep_bg = dep_evaluator.mae(method="BG")
        mae_dep_ipcw = dep_evaluator.mae(method="IPCW")

        # Create results
        model_results = pd.DataFrame()
        result_row = pd.Series([seed, model_name, best_copula_name, dataset_name, strategy, best_copula_theta,
                                ci_true, ibs_true, mae_true, ci_harrell, ci_uno, ibs_ipcw, mae_uncensored, mae_hinge, mae_margin,
                                mae_ipcwv1, mae_ipcwv2, mae_pseudo, ci_dep_ipcw, ibs_dep_bg, ibs_dep_ipcw,
                                mae_dep_bg, mae_dep_ipcw],
                                index=["Seed", "ModelName", "Copula", "Dataset", "Strategy", "Theta",
                                       "CITrue", "IBSTrue", "MAETrue", "CIHarrell", "CIUno", "IBSIPCW", "MAEUncens",
                                       "MAEHinge", "MAEMargin", "MAEIPCWV1", "MAEIPCWV2", "MAEPseudo",
                                       "CIDepIPCW", "IBSDepBG", "IBSDepIPCW", "MAEDepBG", "MAEDepIPCW"])
        print(result_row)
        model_results = pd.concat([model_results, result_row.to_frame().T], ignore_index=True)
        
        # Save results
        filename = f"{cfg.RESULTS_DIR}/semisynthetic_results.csv"
        if os.path.exists(filename):
            results = pd.read_csv(filename)
        else:
            results = pd.DataFrame(columns=model_results.columns)
        results = results.append(model_results, ignore_index=True)
        results.to_csv(filename, index=False)
        