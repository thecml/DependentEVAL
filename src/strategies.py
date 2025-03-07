from typing import List
from data_loader import MetabricDataLoader
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, WeibullAFTFitter
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sklearn.inspection import permutation_importance
import config as cfg

from misc.plot_km_curves import compare_km_curves
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

if __name__ == "__main__":
    # Load data
    dl = MetabricDataLoader().load_data()
    num_features, cat_features = dl.get_features()
    df_full = dl.get_data()
    
    # Drop censored rows
    df = df_full.drop(df_full[df_full.event == 0].index)
    df.reset_index(drop=True, inplace=True)
    df.time = df.time.round().astype(int)
    
    # Make synthetic censoring time
    strategy = "feature_importance" # original, best, feature_importance
    censor_times = make_synthetic_censoring(strategy, df, df_full)
    censor_times = np.round(censor_times).astype(int)
    
    # Combine truth and censored data
    df = combine_data_with_censor(df, censor_times)
    compare_km_curves(df_full, df, show=True)
    