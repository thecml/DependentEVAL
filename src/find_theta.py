import argparse
import random
import torch
from copula import Clayton_Bivariate, Frank_Bivariate
from data_loader import SingleEventSyntheticDataLoader
import pandas as pd
import numpy as np
import config as cfg
from models import Weibull_log_linear
from utility.survival import theta_to_kendall_tau
from trainer import train_copula_model

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

dtype = torch.float64
torch.set_default_dtype(dtype)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--k_tau', type=float, default=0.25)
    parser.add_argument('--copula_name', type=str, default="clayton")
    parser.add_argument('--linear', action='store_true') # store_true = False by default
    
    args = parser.parse_args()
    seed = args.seed
    k_tau = args.k_tau
    copula_name = args.copula_name
    linear = args.linear
    
    dl = SingleEventSyntheticDataLoader().load_data(cfg.data_cfg, k_tau=k_tau, copula_name=copula_name,
                                                    linear=linear, device=device, dtype=dtype)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=seed)
    
    for dataset in [train_dict, valid_dict, test_dict]: # put on device
        for key in ['X', 'T', 'E']:
            dataset[key] = dataset[key].to(device)
    
    # Estimate theta
    n_features = train_dict['X'].shape[1]
    dep_model1 = Weibull_log_linear(n_features, dtype=dtype, device=device) # censoring model
    dep_model2 = Weibull_log_linear(n_features, dtype=dtype, device=device) # event model
    if copula_name == "clayton":
        copula = Clayton_Bivariate(2.0, 1e-4, dtype=dtype, device=device)
    elif copula_name == "frank":
        copula = Frank_Bivariate(2.0, 1e-4, dtype=dtype, device=device)
    dep_model1, dep_model2, copula = train_copula_model(dep_model1, dep_model2, train_dict,
                                                        valid_dict, copula=copula, n_epochs=10000,
                                                        patience=1000, lr=1e-4, batch_size=1024, verbose=True)
    
    estimated_theta = float(copula.parameters()[0])
    estimated_k_tau = theta_to_kendall_tau(copula_name, estimated_theta)
    
    result_row = pd.Series([seed, linear, copula_name, k_tau, estimated_theta, estimated_k_tau],
                            index=["Seed", "Linear", "Copula", "KTau", "EstTheta", "EstKTau"])
    
    