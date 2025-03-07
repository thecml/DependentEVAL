import numpy as np
import pandas as pd
import config as cfg

from data_loader import get_data_loader
from utility.preprocessor import Preprocessor

datasets = ["metabric", "mimic_all", "mimic_hospital",
            "seer_brain", "seer_liver", "seer_stomach"]
dataset_info = []

# Iterate over datasets and axes
for data_name in datasets:
    dl = get_data_loader(data_name).load_data()
    data = dl.get_data().reset_index(drop=True)

    event_times = data.time.values[data.event.values == 1]
    censor_times = data.time.values[data.event.values == 0]

    # Calculate dataset statistics
    n_samples = data.shape[0]
    n_features = data.shape[1] - 2  # Exclude 'time' and 'event' columns
    n_events = np.sum(data.event)  # Count the number of True in the 'event' column
    censor_rate = len(censor_times) / n_samples * 100
    max_time = data.time.values[data.event == True].max()  # Maximum time for event=True
    
    num_features, cat_features = dl.get_features()
    preprocessor = Preprocessor(cat_feat_strat='mode', num_feat_strat='mean', scaling_strategy="standard")
    transformer = preprocessor.fit(data.drop(['time', 'event'], axis=1),
                                   cat_feats=cat_features, num_feats=num_features,
                                   one_hot=True, fill_value=-1)
    X = transformer.transform(data.drop(['time', 'event'], axis=1)).reset_index(drop=True)
    n_features_after = X.shape[1]

    dataset_info.append((data_name.upper(), n_samples, n_features, n_features_after,
                         censor_rate,  n_events, max_time))

# Sort datasets by the number of samples (n_samples)
dataset_info_sorted = sorted(dataset_info, key=lambda x: x[1])

# Print the sorted dataset information in LaTeX format
for info in dataset_info_sorted:
    print(f"{info[0]} & {info[1]} & {info[2]} ({info[3]}) & {info[4]:.1f}\% & "
          f"{info[5]:.0f} & {info[6]:.0f} \\\\")