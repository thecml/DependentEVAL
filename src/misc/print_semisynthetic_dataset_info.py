import numpy as np
import pandas as pd
import config as cfg

from data_loader import get_data_loader
from strategies import combine_data_with_censor, make_synthetic_censoring
from utility.preprocessor import Preprocessor

datasets = ["gbsg", "metabric", "mimic", "nacd", "support", "whas",
            "seer_brain", "seer_breast", "seer_liver", "seer_prostate", "seer_stomach"]
dataset_info = []

# Iterate over datasets and axes
for dataset in datasets:
    # Load data
    dl = get_data_loader(dataset).load_data()
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
    censor_times, selected_features = make_synthetic_censoring("original", df, df_full)
    censor_times = np.round(censor_times).astype(int)
    
    # Combine truth and censored data to make semi-synthetic dataset
    df = combine_data_with_censor(df, censor_times, selected_features)

    # Get times    
    event_times = df.time.values[df.event.values == 1]
    censor_times = df.time.values[df.event.values == 0]

    # Calculate dataset statistics
    n_samples = df.shape[0]
    censor_rate = len(censor_times) / n_samples * 100
    
    dataset_info.append((dataset.upper(), n_samples, censor_rate))

# Sort datasets by the number of samples (n_samples)
dataset_info_sorted = sorted(dataset_info, key=lambda x: x[1])

# Print the sorted dataset information in LaTeX format
for info in dataset_info_sorted:
    print(f"{info[0]} & {info[1]} & {info[2]:.1f}\% \\\\")