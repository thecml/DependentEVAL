import os
import random
import torch
from copula import Clayton_Bivariate, Frank_Bivariate
from data_loader import get_data_loader
import pandas as pd
import numpy as np
import config as cfg
from utility.data import subsample_dataset, fix_types
from models import Weibull_model
from strategies import make_semi_synth
from utility.preprocessor import Preprocessor
from utility.survival import make_stratified_split
from trainer import train_copula_model
import time

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

dtype = torch.float64
torch.set_default_dtype(dtype)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_names = ("seer_brain", "seer_liver", "seer_stomach")

if __name__ == "__main__":
    seed = 0
    strategy = "original"

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    copula_param_path = f"{cfg.RESULTS_DIR}/copula_parameters.csv"
    copula_log_path = f"{cfg.RESULTS_DIR}/copula_fitting_log.csv"

    for dataset_name in dataset_names:
        print(f"Now fitting copula for {dataset_name}")
        
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
        data_valid = df_valid.drop(columns=["true_time", "true_censor"])
        X_train = data_train.drop(columns=["time", "event"])
        X_valid = data_valid.drop(columns=["time", "event"])

        # Format data
        train_dict, valid_dict = dict(), dict()
        train_dict['X'] = torch.tensor(X_train.values, device=device, dtype=dtype)
        train_dict['T'] = torch.tensor(data_train['time'].values, device=device, dtype=dtype)
        train_dict['E'] = torch.tensor(data_train['event'].values, device=device, dtype=dtype)
        valid_dict['X'] = torch.tensor(X_valid.values, device=device, dtype=dtype)
        valid_dict['T'] = torch.tensor(data_valid['time'].values, device=device, dtype=dtype)
        valid_dict['E'] = torch.tensor(data_valid['event'].values, device=device, dtype=dtype)

        n_samples = train_dict['X'].shape[0]
        n_features = train_dict['X'].shape[1]
        best_loss = float("inf")

        # robust timing + peak GPU memory (MiB)
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

            dep_model1, dep_model2, copula, min_val_loss = train_copula_model(
                dep_model1, dep_model2, train_dict, valid_dict,
                copula=copula, n_epochs=30000, lr=0.01,
                batch_size=n_samples, copula_name=copula_name, verbose=False
            )

            copula_theta = float(copula.theta.item())

            if float(min_val_loss) < best_loss:
                best_loss = float(min_val_loss)
                best_copula_name = copula_name
                best_copula_theta = copula_theta

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        copula_end_time = time.time()

        copula_runtime = copula_end_time - copula_start_time

        if torch.cuda.is_available():
            copula_peak_alloc = torch.cuda.max_memory_allocated(device)
            copula_memory_used = (copula_peak_alloc - copula_mem_baseline) / (1024 ** 2)  # MiB
            copula_memory_used = max(0.0, float(copula_memory_used))
        else:
            copula_memory_used = 0.0

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
            existing = pd.concat([existing, new_row], ignore_index=True).drop_duplicates(
                subset=["Seed", "Dataset", "Strategy"], keep="last"
            )
            existing.to_csv(copula_param_path, index=False)
        else:
            new_row.to_csv(copula_param_path, index=False)

        print(f"Copula fitting done. Time: {copula_runtime:.2f}s | Peak GPU Mem: {copula_memory_used:.2f} MiB")

        # Save timing/memory log (one row per seed/dataset/strategy)
        log_row = pd.DataFrame([{
            "Seed": seed,
            "Dataset": dataset_name,
            "Strategy": strategy,
            "BestCopulaName": best_copula_name,
            "BestCopulaTheta": best_copula_theta,
            "CopulaRuntimeSec": float(copula_runtime),
            "CopulaPeakMemMiB": float(copula_memory_used),
        }])

        if os.path.exists(copula_log_path):
            existing_log = pd.read_csv(copula_log_path)
            existing_log = pd.concat([existing_log, log_row], ignore_index=True).drop_duplicates(
                subset=["Seed", "Dataset", "Strategy"], keep="last"
            )
            existing_log.to_csv(copula_log_path, index=False)
        else:
            log_row.to_csv(copula_log_path, index=False)
