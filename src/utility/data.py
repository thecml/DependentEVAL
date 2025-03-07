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
        "metabric": ("1,102", "3.8"),
        "mimic_all": ("12,845", "10.6"),
        "mimic_hospital": ("6,780", "74.3"),
        "seer_brain": ("44,137", "9.0"),
        "seer_liver": ("51,704", "20.5"),
        "seer_stomach": ("56,807", "21.9"),
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
        "metabric": "METABRIC",
        "mimic_all": "MIMIC-IV (all)",
        "mimic_hospital": "MIMIC-IV (hospital)",
        "nacd": "NACD",
        "support": "SUPPORT",
        "whas": "WHAS",
        "seer_brain": "SEER (brain)",
        "seer_breast": "SEER (breast)",
        "seer_liver": "SEER (liver)",
        "seer_prostate": "SEER (prostate)",
        "seer_stomach": "SEER (stomach)",
    }.get(dataset_name, dataset_name)