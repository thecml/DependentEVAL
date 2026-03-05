from typing import List
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.inspection import permutation_importance
import config as cfg

from sota.sksurv import make_cox_model
from utility.data import dotdict
from utility.survival import convert_to_structured

def combine_data_with_censor(
        df: pd.DataFrame,
        censor_time: np.ndarray,
        selected_features: List
) -> pd.DataFrame:
    event_status = df.event.values
    true_times = df.time.values
    times = np.copy(true_times)
    event_status[censor_time < true_times] = 0
    times[event_status == 0] = censor_time[event_status == 0]
    df.drop(columns=["time", "event"], inplace=True)
    df["time"] = times
    df["event"] = event_status
    df["true_time"] = true_times
    df = df[df.time != 0]  # Drop all patients with censor time 0
    df = df[['time'] + ['event'] + ['true_time'] + list(selected_features)]
    df.reset_index(drop=True, inplace=True)
    return df

def calculate_pdf(cdf):
    cdf_extra = np.ones((cdf.shape[0], cdf.shape[1] + 1))
    cdf_extra[:, :-1] = cdf
    pdf = np.diff(cdf_extra, axis=1)
    pdf /= pdf.sum(axis=1)[:, None]
    return pdf

def sample_time_from_curve(
        survival_curve: pd.DataFrame,
        n_samples: int,
) -> np.ndarray:
    """
    Sample time from a survival curve.
    The survival curve is a DataFrame with time as rows and samples as columns.
    """
    uniq_times = survival_curve.index.values
    end_time = uniq_times[-1]
    if survival_curve.shape[1] == 1:
        end_prob = survival_curve.iloc[-1, 0]
        extrapolation_t = end_time / (1 - end_prob)
        cdf = 1 - survival_curve.values.flatten()
        cdf_full = np.zeros(cdf.shape[0] + 2) # +2 because we add 0 at the start and 1 at the end
        cdf_full[1:-1] = cdf
        cdf_full[-1] = 1
        pdf = np.diff(cdf_full)
        # pdf /= pdf.sum()
        uniq_times = np.append(uniq_times, extrapolation_t)
        times = np.random.choice(uniq_times, size=n_samples, p=pdf)
    elif survival_curve.shape[1] > 1:
        survival = survival_curve.values.T
        cdf = 1 - survival
        cdf_full = np.zeros((cdf.shape[0], cdf.shape[1] + 2)) # +2 because we add 0 at the start and 1 at the end
        cdf_full[:, 1:-1] = cdf
        cdf_full[:, -1] = 1
        pdf = np.diff(cdf_full, axis=1)
        # pdf /= pdf.sum(axis=1)[:, None]
        times = np.empty(n_samples)
        uniq_ti = np.empty(uniq_times.shape[0] + 1)
        uniq_ti[:-1] = uniq_times
        for i in range(n_samples):
            end_prob = survival[i, -1]
            extrapolation_t = end_time / (1 - end_prob)
            uniq_ti[-1] = extrapolation_t
            times[i] = np.random.choice(uniq_ti, p=pdf[i, :])
    else:
        raise ValueError("Survival curve must have at least one column.")
    return times

def make_synthetic_censoring(strategy: str,
                             df_event: pd.DataFrame,
                             df_all: pd.DataFrame):
    """
    Build synthetic dependent censoring times
    :param strategy: type of censoring strategy
    :param df_event: dataframe with all event patients
    :param df_all: dataframe with all patients
    :return: synthetic censoring times
    """
    if strategy == "original":
        # Use original censoring distribution from the dataset. Assumes cond. indep censoring
        df_all_copy = df_all.copy()  # Make a copy to avoid changing the original dataset
        df_all_copy.event = 1 - df_all_copy.event
        X = df_all_copy.drop(['event', 'time'], axis=1)
        y_cens = convert_to_structured(df_all_copy['time'], df_all_copy['event'])
        config = dotdict(cfg.COXPH_PARAMS)
        cph_cens = make_cox_model(config)
        cph_cens = CoxPHSurvivalAnalysis(alpha=0.0001)
        cph_cens.fit(X, y_cens)
        censor_curves = cph_cens.predict_survival_function(df_event.drop(['event', 'time'], axis=1))
        censor_curves = pd.DataFrame(np.row_stack([fn(cph_cens.unique_times_) for fn in censor_curves]), columns=cph_cens.unique_times_)
        uniq_times = cph_cens.unique_times_
        censor_cdf = 1 - censor_curves.values
        censor_pdf = calculate_pdf(censor_cdf)
        censor_times = np.zeros(censor_pdf.shape[0])
        for i in range(censor_pdf.shape[0]):
            censor_times[i] = uniq_times[np.argmax(censor_pdf[i,:])] # use the max prob
        selected_features = df_all_copy.drop(columns=['time', 'event']).columns
    elif strategy == "top_5":
        # Find top 5 features using feature importances
        df_all_copy = df_all.copy()
        X = df_all_copy.drop(['event', 'time'], axis=1)
        y = convert_to_structured(df_all_copy['time'], df_all_copy['event'])
        config = dotdict(cfg.COXPH_PARAMS)
        cph_features = make_cox_model(config)
        cph_features.fit(X, y)
        result = permutation_importance(cph_features, X, y, n_jobs=-1,
                                        max_samples=0.25, random_state=0)
        importances_perm = result.importances_mean
        feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances_perm})
        top_5_fts = feature_importance_df.sort_values(by="Importance", ascending=False).head(5)['Feature']
       
        # Train CPH model on censoring dist
        df_all_copy = df_all.copy()
        df_all_copy.event = 1 - df_all_copy.event
        X = df_all_copy[df_all_copy.columns].drop(['time', 'event'], axis=1)
        y_cens = convert_to_structured(df_all_copy['time'], df_all_copy['event'])
        config = dotdict(cfg.COXPH_PARAMS)
        cph_cens = make_cox_model(config)
        cph_cens.fit(X, y_cens)
        
        # Predict the censoring
        censor_curves = cph_cens.predict_survival_function(df_event.drop(['event', 'time'], axis=1))
        censor_curves = pd.DataFrame(np.row_stack([fn(cph_cens.unique_times_) for fn in censor_curves]), columns=cph_cens.unique_times_)
        uniq_times = cph_cens.unique_times_
        censor_cdf = 1 - censor_curves.values
        censor_pdf = calculate_pdf(censor_cdf)
        censor_times = np.empty(censor_pdf.shape[0])
        for i in range(censor_pdf.shape[0]):
            censor_times[i] = uniq_times[np.argmax(censor_pdf[i, :])] # use the max prob
            
        selected_features = top_5_fts
    elif strategy == "top_10":
        # Find top 10 features
        df_all_copy = df_all.copy()
        X = df_all_copy.drop(['event', 'time'], axis=1)
        y = convert_to_structured(df_all_copy['time'], df_all_copy['event'])
        config = dotdict(cfg.COXPH_PARAMS)
        cph_features = make_cox_model(config)
        cph_features.fit(X, y)
        result = permutation_importance(cph_features, X, y, n_jobs=-1,
                                        max_samples=0.25, random_state=0)
        importances_perm = result.importances_mean
        feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances_perm})
        top_10_fts = feature_importance_df.sort_values(by="Importance", ascending=False).head(10)['Feature']
        
        # Train CPH model on censoring dist
        df_all_copy = df_all.copy()
        df_all_copy.event = 1 - df_all_copy.event
        X = df_all_copy[df_all_copy.columns].drop(['time', 'event'], axis=1)
        y_cens = convert_to_structured(df_all_copy['time'], df_all_copy['event'])
        config = dotdict(cfg.COXPH_PARAMS)
        cph_cens = make_cox_model(config)
        cph_cens.fit(X, y_cens)
        
        # Predict the censoring
        censor_curves = cph_cens.predict_survival_function(df_event.drop(['event', 'time'], axis=1))
        censor_curves = pd.DataFrame(np.row_stack([fn(cph_cens.unique_times_) for fn in censor_curves]), columns=cph_cens.unique_times_)
        uniq_times = cph_cens.unique_times_
        censor_cdf = 1 - censor_curves.values
        censor_pdf = calculate_pdf(censor_cdf)
        censor_times = np.empty(censor_pdf.shape[0])
        for i in range(censor_pdf.shape[0]):
            censor_times[i] = uniq_times[np.argmax(censor_pdf[i, :])] # use the max prob
            
        selected_features = top_10_fts
    elif strategy == "random_25":
        # Keep random 25% of the features
        df_all_copy = df_all.copy()  # Make a copy to avoid changing the original dataset
        all_features = df_all_copy.drop(columns=['time', 'event']).columns # Exclude time and event columns
        random_25_fts = np.random.choice(all_features, size=int(len(all_features) * 0.25), replace=False)
        
        # Train CPH model on censoring dist
        df_all_copy = df_all.copy()
        df_all_copy.event = 1 - df_all_copy.event
        X = df_all_copy[df_all_copy.columns].drop(['time', 'event'], axis=1)
        y_cens = convert_to_structured(df_all_copy['time'], df_all_copy['event'])
        config = dotdict(cfg.COXPH_PARAMS)
        cph_cens = make_cox_model(config)
        cph_cens.fit(X, y_cens)
        
        # Predict the censoring
        censor_curves = cph_cens.predict_survival_function(df_event.drop(['event', 'time'], axis=1))
        censor_curves = pd.DataFrame(np.row_stack([fn(cph_cens.unique_times_) for fn in censor_curves]), columns=cph_cens.unique_times_)
        uniq_times = cph_cens.unique_times_
        censor_cdf = 1 - censor_curves.values
        censor_pdf = calculate_pdf(censor_cdf)
        censor_times = np.empty(censor_pdf.shape[0])
        for i in range(censor_pdf.shape[0]):
            censor_times[i] = uniq_times[np.argmax(censor_pdf[i, :])] # use the max prob
        
        selected_features = random_25_fts
    else:
        raise NotImplementedError()    
    
    return censor_times, selected_features

def make_synthetic_censoring_top_k(df_event: pd.DataFrame,
                                   df_all: pd.DataFrame,
                                   top_k: int):
    """
    Build synthetic dependent censoring times
    :param strategy: type of censoring strategy
    :param df_event: dataframe with all event patients
    :param df_all: dataframe with all patients
    :return: synthetic censoring times
    """
    df_all_copy = df_all.copy()
    df_all_copy.event = 1 - df_all_copy.event
    X = df_all_copy[df_all_copy.columns].drop(['time', 'event'], axis=1)
    y = convert_to_structured(df_all_copy['time'], df_all_copy['event'])
    gbsa = GradientBoostingSurvivalAnalysis(random_state=0)
    gbsa.fit(X, y)
    importances = gbsa.feature_importances_
    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    top_k_fts = list(feature_importances.sort_values(by='Importance', ascending=False)[:top_k]['Feature'])
    cph = CoxPHSurvivalAnalysis(alpha=0.0001)
    X = df_all_copy.drop(['event', 'time'], axis=1)
    y = convert_to_structured(df_all_copy['time'], df_all_copy['event'])
    cph.fit(X, y)
    censor_curves = cph.predict_survival_function(df_event.drop(['event', 'time'], axis=1))
    censor_curves = pd.DataFrame(np.row_stack([fn(cph.unique_times_) for fn in censor_curves]), columns=cph.unique_times_)
    uniq_times = cph.unique_times_
    censor_cdf = 1 - censor_curves.values
    censor_pdf = calculate_pdf(censor_cdf)
    censor_times = np.empty(censor_pdf.shape[0])
    for i in range(censor_pdf.shape[0]):
        censor_times[i] = uniq_times[np.argmax(censor_pdf[i, :])] # use the max prob
    selected_features = top_k_fts
    
    return censor_times, selected_features

def make_semi_synth(
        dataset: pd.DataFrame,
        censoring_assumption: str = "cond_ind",
        strategy: str = "original",
        seed: int = 0
) -> tuple[pd.DataFrame, list[str]]:
    np.random.seed(seed)
    df_original = dataset
    df_orig_censor = df_original.copy()
    df_orig_censor.event = 1 - df_orig_censor.event  # invert event status for censoring
    
    if strategy == "original":
        features = df_original.drop(columns=["time", "event"]).columns.tolist()
    elif strategy == "top_1":
        df_all_copy = df_original.copy()
        X = df_all_copy.drop(['event', 'time'], axis=1)
        y = convert_to_structured(df_all_copy['time'], df_all_copy['event'])
        config = dotdict(cfg.COXPH_PARAMS)
        cph_features = make_cox_model(config)
        cph_features.fit(X, y)
        result = permutation_importance(cph_features, X, y, n_jobs=-1,
                                        max_samples=0.25, random_state=0)
        importances_perm = result.importances_mean
        feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances_perm})
        features = feature_importance_df.sort_values(by="Importance", ascending=False).head(1)['Feature']
    elif strategy == "top_5":
        df_all_copy = df_original.copy()
        X = df_all_copy.drop(['event', 'time'], axis=1)
        y = convert_to_structured(df_all_copy['time'], df_all_copy['event'])
        config = dotdict(cfg.COXPH_PARAMS)
        cph_features = make_cox_model(config)
        cph_features.fit(X, y)
        result = permutation_importance(cph_features, X, y, n_jobs=-1,
                                        max_samples=0.25, random_state=0)
        importances_perm = result.importances_mean
        feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances_perm})
        features = feature_importance_df.sort_values(by="Importance", ascending=False).head(5)['Feature']
    elif strategy == "top_10":
        df_all_copy = df_original.copy()
        X = df_all_copy.drop(['event', 'time'], axis=1)
        y = convert_to_structured(df_all_copy['time'], df_all_copy['event'])
        config = dotdict(cfg.COXPH_PARAMS)
        cph_features = make_cox_model(config)
        cph_features.fit(X, y)
        result = permutation_importance(cph_features, X, y, n_jobs=-1,
                                        max_samples=0.25, random_state=0)
        importances_perm = result.importances_mean
        feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances_perm})
        features = feature_importance_df.sort_values(by="Importance", ascending=False).head(10)['Feature']
    elif strategy == "random_25":
        df_all_copy = df_original.copy()
        all_features = df_all_copy.drop(columns=['time', 'event']).columns
        features = np.random.choice(all_features, size=int(len(all_features) * 0.25), replace=False)
    else:
        raise ValueError("Invalid strategy")

    event_generator = CoxPHFitter(penalizer=0.01)
    event_generator.fit(df_original, duration_col='time', event_col='event')
    event_curves = event_generator.predict_survival_function(df_original)

    if censoring_assumption == "cond_ind":
        censoring_generator = CoxPHFitter(penalizer=0.01)
        censoring_generator.fit(df_orig_censor, duration_col='time', event_col='event')
        censor_curves = censoring_generator.predict_survival_function(df_original)
    elif censoring_assumption == "random":
        censoring_generator = KaplanMeierFitter()
        censoring_generator.fit(durations=df_orig_censor['time'], event_observed=df_orig_censor['event'])
        # Use Kaplan-Meier estimator for random censoring
        censor_curves = censoring_generator.survival_function_
    else:
        raise ValueError("Unknown censoring assumption. Use 'cond_ind' or 'random'.")

    # Sample times from the survival curves
    e = sample_time_from_curve(event_curves, n_samples=df_original.shape[0])
    c = sample_time_from_curve(censor_curves, n_samples=df_original.shape[0])
    e = np.round(e).astype(int)
    c = np.round(c).astype(int)

    times = np.minimum(e, c)
    delta = e < c  # event indicator, 1 if event occurred, 0 if censored
    
    # Create a DataFrame with the sampled times and event indicators
    df = pd.DataFrame({
        "time": times,
        "event": delta,
        "true_time": e,
        "true_censor": c,
        **{f"{feature}": df_original[feature].values for feature in features}
    })
    
    # Replace invalid or negative times
    df["time"] = np.where(df["time"] <= 0, 1, df["time"])
    df["time"] = np.where(~np.isfinite(df["time"]), np.nan, df["time"])
    
    # Drop any rows that still have NaN times
    df = df.dropna(subset=["time"]).reset_index(drop=True)
    
    # Clip extremely large times to avoid outliers
    df["time"] = np.clip(df["time"], a_min=1, a_max=np.percentile(df["time"], 99.9))
    
    df = df[df.time != 0]  # Drop all patients with censor time 0
    df.reset_index(drop=True, inplace=True)
    
    return df
