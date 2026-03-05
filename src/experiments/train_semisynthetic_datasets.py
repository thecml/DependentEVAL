import argparse
import os
import random
import torch
from copula import Clayton_Bivariate, Frank_Bivariate
from data_loader import get_data_loader
import pandas as pd
import numpy as np
import config as cfg
from evaluator import DependentEvaluator
from sota.deepsurv import DeepSurv, make_deepsurv_prediction, train_deepsurv_model
from sota.mtlr import make_mtlr_prediction, mtlr, train_mtlr_model
from sota.sksurv import make_cox_model, make_gbsa_model, make_rsf_model, make_weibull_aft_model
from utility.data import dotdict, subsample_dataset, fix_types
from SurvivalEVAL import SurvivalEvaluator
from scipy.interpolate import interp1d

from models import Weibull_model
from strategies import make_semi_synth
from utility.preprocessor import Preprocessor
from utility.survival import convert_to_structured, make_stratified_split, make_time_bins
from trainer import train_copula_model

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
    
    # Make semi-synthetic dataset
    df_synth = make_semi_synth(df_full, strategy=strategy)
    
    # Subsample
    if dataset_name == "employee":
        df_subsample = subsample_dataset(df_synth.copy(), dataset_name, target_size=10000)
    elif dataset_name == "mimic_all":
        df_subsample = subsample_dataset(df_synth.copy(), dataset_name, target_size=10000)
    elif dataset_name in ["seer_brain", "seer_liver", "seer_stomach"]:
        df_subsample = subsample_dataset(df_synth.copy(), dataset_name, target_size=10000)
    else:
        df_subsample = df_synth
        
    # Split data
    df_train, df_valid, df_test = make_stratified_split(df_subsample, stratify_colname='both', frac_train=0.7,
                                                        frac_valid=0.1, frac_test=0.2,
                                                        random_state=seed)
    
    # Fix types
    df_train, df_valid, df_test = fix_types(df_train, df_valid, df_test)
  
    # Process data
    data_train = df_train.drop(columns=["true_time", "true_censor"])
    true_test_time = df_test.true_time.values
    true_test_event = np.ones(df_test.shape[0])
    data_valid = df_valid.drop(columns=["true_time", "true_censor"])
    data_test = df_test.drop(columns=["true_time", "true_censor"])
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
    copula_param_path = f"{cfg.RESULTS_DIR}/copula_parameters.csv"
    best_copula_name = None
    best_copula_theta = None
        
    # Try to load cached copula parameters
    if os.path.exists(copula_param_path):
        cop_df = pd.read_csv(copula_param_path)
        match = cop_df[
            (cop_df["Seed"] == seed) &
            (cop_df["Dataset"] == dataset_name) &
            (cop_df["Strategy"] == strategy)
        ]
        if len(match) == 1:
            best_copula_name = match.iloc[0]["BestCopulaName"]
            best_copula_theta = match.iloc[0]["BestCopulaTheta"]
            print(f"Loaded cached copula: {best_copula_name} (θ={best_copula_theta})")
            copula_runtime = 0.0
            copula_memory_used = 0.0
    
    # If not cached, then perform training
    if best_copula_name is None:
        copula_result = dict()
        best_loss = float("inf")

        # robust timing + peak memory (MiB)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            copula_mem_baseline = torch.cuda.memory_allocated(device)
        else:
            copula_mem_baseline = 0

        copula_start_time = time.time()

        for copula_name in ["clayton", "frank"]:
            # Reset seeds
            np.random.seed(0)
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            dep_model1 = Weibull_model(n_features, dtype=dtype, device=device)
            dep_model2 = Weibull_model(n_features, dtype=dtype, device=device)

            if copula_name == "clayton":
                copula = Clayton_Bivariate(2.0, 1e-4, dtype=dtype, device=device)
            elif copula_name == "frank":
                copula = Frank_Bivariate(2.0, 1e-4, dtype=dtype, device=device)
            else:
                raise ValueError(f"Unknown copula_name={copula_name}")

            dep_model1, dep_model2, copula, min_val_loss = train_copula_model(
                dep_model1, dep_model2, train_dict, valid_dict,
                copula=copula, n_epochs=30000, lr=0.01,
                batch_size=n_samples, copula_name=copula_name, verbose=False
            )

            copula_theta = float(copula.theta.item())
            copula_result[copula_name] = {"theta": copula_theta, "val_loss": float(min_val_loss)}

            if min_val_loss < best_loss:
                best_loss = float(min_val_loss)
                best_copula_name = copula_name
                best_copula_theta = copula_theta

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        copula_end_time = time.time()

        copula_runtime = copula_end_time - copula_start_time

        # Peak allocated memory during the whole copula-fitting block (MiB)
        if torch.cuda.is_available():
            copula_peak_alloc = torch.cuda.max_memory_allocated(device)
            copula_memory_used = (copula_peak_alloc - copula_mem_baseline) / (1024 ** 2)  # MiB
            # Guard against tiny negative due to allocator bookkeeping
            copula_memory_used = max(0.0, float(copula_memory_used))
        else:
            copula_memory_used = 0.0

        print(f"Best copula: {best_copula_name} with theta = {best_copula_theta} and val_loss = {best_loss}")

        # Save to copula_parameters.csv
        new_row = pd.DataFrame([{
            "Seed": seed,
            "Dataset": dataset_name,
            "Strategy": strategy,
            "BestCopulaName": best_copula_name,
            "BestCopulaTheta": best_copula_theta
        }])

        if os.path.exists(copula_param_path):
            existing = pd.read_csv(copula_param_path)
            existing = pd.concat([existing, new_row], ignore_index=True).drop_duplicates()
            existing.to_csv(copula_param_path, index=False)
        else:
            new_row.to_csv(copula_param_path, index=False)

    print(f"Copula fitting done. Time: {copula_runtime:.2f}s | Peak GPU Mem: {copula_memory_used:.2f} MiB")
    
    # Create results
    model_results = pd.DataFrame(columns=[
        "Seed", "ModelName", "Dataset", "Strategy",
        "BestCopulaName", "BestCopulaTheta",
        "IBSTrue", "IBSUncensored", "IBSIPCW",
        "IBSIndepBG", "IBSIndepBGUW", "IBSDepBG", "IBSDepBGUW",
    ])
    
    # Create runtime log
    runtime_log = pd.DataFrame(columns=[
        "Seed", "ModelName", "Dataset", "Strategy",
        "CopulaRuntime", "CopulaMemoryUsed",
        "IBSUncensTime", "IBSIPCWTime", "IBSIndepBGTime",
        "IBSIndepBGUWTime", "IBSDepBGTime", "IBSDepBGUWTime",
    ])
    
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
        elif model_name == "weibullaft":
            config = dotdict(cfg.WEIBULL_AFT_PARAMS)
            model = make_weibull_aft_model(config)
            model.fit(X_train, y_train)
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
        elif model_name == "weibullaft":
            times_numpy = time_bins.cpu().numpy()
            X_test_df = pd.DataFrame(test_dict['X'].cpu().numpy(),
                                     columns=model.feature_names_)
            surv_df = model.model.predict_survival_function(X_test_df, times=times_numpy)
            preds_array = np.minimum(np.asarray(surv_df.T), 1.0)
            survival_outputs = pd.DataFrame(preds_array, columns=times_numpy)
        else:
            raise NotImplementedError()
        
        # Make dataframe
        survival_outputs = pd.DataFrame(survival_outputs, columns=time_bins.cpu().numpy())
        survival_outputs[0] = 1
        
        # Create true evaluator to calculate true IBS
        true_evaluator = SurvivalEvaluator(survival_outputs, time_bins, true_test_time, true_test_event)
        ibs_true = true_evaluator.integrated_brier_score(IPCW_weighted=False, num_points=10)
        original_evaluator = SurvivalEvaluator(survival_outputs, time_bins,
                                               data_test.time.values, data_test.event.values,
                                               data_train.time.values, data_train.event.values)
        # Calculate IBS
        ibs_uncens_start_time = time.time()
        ibs_uncens = original_evaluator.integrated_brier_score(IPCW_weighted=False, num_points=10)
        ibs_uncens_end_time = time.time()
        ibs_uncens_time = ibs_uncens_end_time - ibs_uncens_start_time
        
        ibs_ipcw_start_time = time.time()
        ibs_ipcw = original_evaluator.integrated_brier_score(num_points=10)
        ibs_ipcw_end_time = time.time()
        ibs_ipcw_time = ibs_ipcw_end_time - ibs_ipcw_start_time
        
        # Calculate independent metrics CI/IBS/MAE
        indep_evaluator = DependentEvaluator(survival_outputs, time_bins, data_test.time.values, data_test.event.values,
                                             data_train.time.values, data_train.event.values, copula_name="clayton", alpha=0)
        
        ibs_indep_bg_start_time = time.time()
        ibs_indep_bg = indep_evaluator.integrated_brier_score(method="BG", num_points=10)
        ibs_indep_bg_end_time = time.time()
        ibs_indep_bg_time = ibs_indep_bg_end_time - ibs_indep_bg_start_time

        ibs_indep_bguw_start_time = time.time()
        ibs_indep_bguw = indep_evaluator.integrated_brier_score(method="BG_UW", num_points=10)
        ibs_indep_bguw_end_time = time.time()
        ibs_indep_bguw_time = ibs_indep_bguw_end_time - ibs_indep_bguw_start_time

        # Calculate dependent metrics
        dep_evaluator = DependentEvaluator(survival_outputs, time_bins, data_test.time.values, data_test.event.values,
                                           data_train.time.values, data_train.event.values, copula_name=best_copula_name,
                                           alpha=best_copula_theta)
            
        ibs_dep_bg_start_time = time.time()
        ibs_dep_bg = dep_evaluator.integrated_brier_score(method="BG", num_points=10)
        ibs_dep_bg_end_time = time.time()
        ibs_dep_bg_time = ibs_dep_bg_end_time - ibs_dep_bg_start_time

        ibs_dep_bguw_start_time = time.time()
        ibs_dep_bguw = dep_evaluator.integrated_brier_score(method="BG_UW", num_points=10)
        ibs_dep_bguw_end_time = time.time()
        ibs_dep_bguw_time = ibs_dep_bguw_end_time - ibs_dep_bguw_start_time
        
        # Create results
        result_row = pd.Series([
            seed, model_name, dataset_name, strategy,
            best_copula_name, best_copula_theta,
            ibs_true, ibs_uncens, ibs_ipcw,
            ibs_indep_bg, ibs_indep_bguw, ibs_dep_bg, ibs_dep_bguw,
        ], index=model_results.columns)
        model_results = pd.concat(
            [model_results, result_row.to_frame().T],
            ignore_index=True
        )
        
        # Create timing results
        runtime_row = pd.Series([
            seed, model_name, dataset_name, strategy,
            copula_runtime, copula_memory_used,
            ibs_uncens_time, ibs_ipcw_time, ibs_indep_bg_time,
            ibs_indep_bguw_time, ibs_dep_bg_time, ibs_dep_bguw_time,
        ], index=runtime_log.columns)

        runtime_log = pd.concat(
            [runtime_log, runtime_row.to_frame().T],
            ignore_index=True
        )
    
    results_path = f"{cfg.RESULTS_DIR}/semisynthetic_results.csv"
    runtime_log_path = f"{cfg.RESULTS_DIR}/semisynthetic_results_timing.csv"
    
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    
    # Save model results
    if os.path.exists(results_path):
        existing_results = pd.read_csv(results_path)
        model_results = pd.concat([existing_results, model_results], ignore_index=True)
    model_results.to_csv(results_path, index=False)

    # Save runtime logs
    if os.path.exists(runtime_log_path):
        existing_runtime = pd.read_csv(runtime_log_path)
        runtime_log = pd.concat([existing_runtime, runtime_log], ignore_index=True)
    runtime_log.to_csv(runtime_log_path, index=False)
