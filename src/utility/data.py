import pandas as pd

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def fix_types(df_train, df_valid, df_test):
    df_train = df_train.astype({col: float for col in df_train.columns if col not in ["time", "true_time", "event"]})
    df_train["time"] = df_train["time"].astype(int)
    df_train["true_time"] = df_train["true_time"].astype(int)
    df_train["event"] = df_train["event"].astype(bool)
    df_valid = df_valid.astype({col: float for col in df_valid.columns if col not in ["time", "true_time", "event"]})
    df_valid["time"] = df_valid["time"].astype(int)
    df_valid["true_time"] = df_valid["true_time"].astype(int)
    df_valid["event"] = df_valid["event"].astype(bool)
    df_test = df_test.astype({col: float for col in df_train.columns if col not in ["time", "true_time", "event"]})
    df_test["time"] = df_test["time"].astype(int)
    df_test["true_time"] = df_test["true_time"].astype(int)
    df_test["event"] = df_test["event"].astype(bool)
    return df_train, df_valid, df_test

def get_dataset_info(dataset_name):
    return {
        "metabric": ("1,102", "42.1"),
        "mimic_all": ("12,845", "66.7"),
        "mimic_hospital": ("6,780", "97.7"),
        "seer_brain": ("44,137", "40.1"),
        "seer_liver": ("51,704", "37.6"),
        "seer_stomach": ("56,807", "43.4"),
    }.get(dataset_name, dataset_name)

def map_strategy_name(strategy):
    return {
        "original": "Original",
        "top_5": "Top 5",
        "top_10": "Top 10",
        "random_25": "Rand. 25\\%"
    }.get(strategy, strategy)
    
def map_dataset_name(dataset_name):
    return {
        "gbsg": "GBSG",
        "aids": "AIDS",
        "metabric": "METABRIC",
        "mimic_all": "MIMIC-IV",
        "mimic_hospital": "MIMIC-IV",
        "nacd": "NACD",
        "support": "SUPPORT",
        "whas": "WHAS",
        "churn": "Churn",
        "employee": "Employee",
        "flchain": "FLCHAIN",
        "seer_brain": "SEER-brain",
        "seer_liver": "SEER-liver",
        "seer_stomach": "SEER-stomach",
    }.get(dataset_name, dataset_name)
    
def subsample_dataset(df, name, time_col="time", event_col="event",
                      n_bins=10, censor_ratio=5, target_size=None, random_state=42):
    """
    Downsample survival datasets according to predefined rules.
    Dataset names supported:
    - "metabric"
    - "mimic_all"
    - "mimic_hospital"
    - "seer_brain", "seer_liver", "seer_stomach"
    """

    df = df.copy()
    df["time_bin"] = pd.qcut(df[time_col], q=n_bins, duplicates="drop")

    # --- Rules by dataset ---
    if name == "metabric":
        out = df

    elif name == "mimic_hospital":
        # Keep all events, sample censored up to ratio
        events = df[df[event_col] == 1]
        cens   = df[df[event_col] == 0]

        n_events = len(events)
        n_censor_target = min(len(cens), censor_ratio * n_events)

        cens_keep = cens.groupby("time_bin", group_keys=False).apply(
            lambda x: x.sample(
                n=max(1, int(len(x) * n_censor_target / len(cens))),
                random_state=random_state
            )
        )
        out = pd.concat([events, cens_keep], axis=0)

        # --- Enforce target_size if given ---
        if target_size is not None and len(out) > target_size:
            grouped = out.groupby([event_col, "time_bin"], group_keys=False)
            out = grouped.apply(
                lambda x: x.sample(
                    n=max(1, int(len(x) * target_size / len(out))),
                    random_state=random_state
                )
            )

    elif name in ["employee", "mimic_all"]:
        if target_size is None:
            out = df
        else:
            grouped = df.groupby([event_col, "time_bin"], group_keys=False)
            out = grouped.apply(
                lambda x: x.sample(
                    n=max(1, int(len(x) * target_size / len(df))),
                    random_state=random_state
                )
            )

    elif name in ["seer_brain", "seer_liver", "seer_stomach"]:
        if target_size is None:
            target_size = 20000
        grouped = df.groupby([event_col, "time_bin"], group_keys=False)
        out = grouped.apply(
            lambda x: x.sample(
                n=max(1, int(len(x) * target_size / len(df))),
                random_state=random_state
            )
        )

    else:
        raise ValueError(f"Unknown dataset name: {name}")

    # Drop helper column and shuffle
    out = out.drop(columns=["time_bin"]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return out

