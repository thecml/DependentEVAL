import torch
import numpy as np
import pandas as pd
import math
from sklearn.utils import shuffle
from skmultilearn.model_selection import iterative_train_test_split
from typing import Union, Tuple, Optional, List, Any
from utility.preprocessor import Preprocessor
from statsmodels.distributions.copula.api import ClaytonCopula, FrankCopula, GumbelCopula

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.optimize import root_scalar

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

def encode_survival(
        time: Union[float, int, NumericArrayLike],
        event: Union[int, bool, NumericArrayLike],
        bins: NumericArrayLike
) -> torch.Tensor:
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    # TODO this should handle arrays and (CUDA) tensors
    if isinstance(time, (float, int, np.ndarray)):
        time = np.atleast_1d(time)
        time = torch.tensor(time)
    if isinstance(event, (int, bool, np.ndarray)):
        event = np.atleast_1d(event)
        event = torch.tensor(event)

    if isinstance(bins, np.ndarray):
        bins = torch.tensor(bins)

    try:
        device = bins.device
    except AttributeError:
        device = "cpu"

    time = np.clip(time, 0, bins.max())
    # add extra bin [max_time, inf) at the end
    y = torch.zeros((time.shape[0], bins.shape[0] + 1),
                    dtype=torch.float,
                    device=device)
    # For some reason, the `right` arg in torch.bucketize
    # works in the _opposite_ way as it does in numpy,
    # so we need to set it to True
    bin_idxs = torch.bucketize(time, bins, right=True)
    for i, (bin_idx, e) in enumerate(zip(bin_idxs, event)):
        if e == 1:
            y[i, bin_idx] = 1
        else:
            y[i, bin_idx:] = 1
    return y.squeeze()

def reformat_survival(
        dataset: pd.DataFrame,
        time_bins: NumericArrayLike,
        dtype: torch.dtype
) -> (torch.Tensor, torch.Tensor):
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    x = torch.tensor(dataset.drop(["time", "event"], axis=1).values, dtype=dtype)
    y = encode_survival(dataset["time"].values, dataset["event"].values, time_bins)
    return x, y

def convert_to_structured(T, E):
    default_dtypes = {"names": ("event", "time"), "formats": ("bool", "f8")}
    concat = list(zip(E, T))
    return np.array(concat, dtype=default_dtypes)

def preprocess_data(X_train, X_valid, X_test, cat_features,
                    num_features, as_array=False) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    preprocessor = Preprocessor(cat_feat_strat='mode', num_feat_strat='mean', scaling_strategy="standard")
    transformer = preprocessor.fit(X_train, cat_feats=cat_features, num_feats=num_features,
                                   one_hot=True, fill_value=-1)
    X_train = transformer.transform(X_train)
    X_valid = transformer.transform(X_valid)
    X_test = transformer.transform(X_test)
    if as_array:
        return (np.array(X_train), np.array(X_valid), np.array(X_test))
    return (X_train, X_valid, X_test)

def compute_l1_difference(truth_preds, model_preds, n_samples, steps, device='cpu'):
    t_m = steps.max().to(device)
    surv1 = truth_preds.to(device)
    surv2 = model_preds.to(device)
    steps = steps.to(device)
    integ = torch.sum(torch.diff(torch.cat([torch.zeros(1, device=device), steps])) * torch.abs(surv1 - surv2))
    result = (integ / t_m / n_samples).detach().cpu().numpy()
    return result

def make_time_bins(
        times: NumericArrayLike,
        num_bins: Optional[int] = None,
        use_quantiles: bool = True,
        event: Optional[NumericArrayLike] = None,
        dtype=torch.float64
) -> torch.Tensor:
    """
    Courtesy of https://ieeexplore.ieee.org/document/10158019
    
    Creates the bins for survival time discretisation.

    By default, sqrt(num_observation) bins corresponding to the quantiles of
    the survival time distribution are used, as in https://github.com/haiderstats/MTLR.

    Parameters
    ----------
    times
        Array or tensor of survival times.
    num_bins
        The number of bins to use. If None (default), sqrt(num_observations)
        bins will be used.
    use_quantiles
        If True, the bin edges will correspond to quantiles of `times`
        (default). Otherwise, generates equally-spaced bins.
    event
        Array or tensor of event indicators. If specified, only samples where
        event == 1 will be used to determine the time bins.

    Returns
    -------
    torch.Tensor
        Tensor of bin edges.
    """
    # TODO this should handle arrays and (CUDA) tensors
    if event is not None:
        times = times[event == 1]
    if num_bins is None:
        num_bins = math.ceil(math.sqrt(len(times)))
    if use_quantiles:
        # NOTE we should switch to using torch.quantile once it becomes
        # available in the next version
        bins = np.unique(np.quantile(times, np.linspace(0, 1, num_bins)))
    else:
        bins = np.linspace(times.min(), times.max(), num_bins)
    bins = torch.tensor(bins, dtype=dtype)
    return bins

def Survival(model, estimate, x, steps=200):
    device = x.device
    u = torch.ones((x.shape[0],), device=device)*0.001
    time_steps = torch.linspace(1e-4,1,steps,device=device).reshape(1,-1).repeat(x.shape[0],1)
    t_max_model = model.rvs(x, u)
    #t_max_estimate = estimate.rvs(x, u)#.reshape(-1,1).repeat(1,100)
    #e = (t_max_model < t_max_estimate).type(torch.float32)
    t_max = t_max_model#e * t_max_estimate + (1-e) * t_max_model
    
    t_max = t_max.reshape(-1,1).repeat(1,steps)
    
    time_steps = t_max * time_steps
    surv1 = torch.zeros((x.shape[0], steps),device=device)
    surv2 = torch.zeros((x.shape[0], steps),device=device)
    for i in range(steps):
        surv1[:,i] = model.survival(time_steps[:,i],x)
        surv2[:,i] = estimate.survival(time_steps[:,i], x)
    
    return surv1, surv2, time_steps, t_max_model

def surv_diff(model, estimate, x, steps):
    surv1, surv2, time_steps, t_m = Survival(model, estimate, x, steps)
    
    
    integ = torch.sum(torch.diff(torch.cat([torch.zeros((surv1.shape[0],1), device=x.device), time_steps], dim=1))*(torch.abs(surv1-surv2)), dim=1)
    #integ2 = torch.sum(torch.diff(torch.cat([torch.zeros(surv1.shape[0],1), time_steps], dim=1))*(torch.abs(surv1)), dim=1)
    #return torch.mean(integ/integ2)
    #print(torch.std(integ/t_m))
    #print(integ.shape)
    #print((integ/t_m).shape)
    return torch.mean(integ/t_m)#time_steps[:,-1])#add std

def kendall_tau_to_theta(copula_name, k_tau):
    if copula_name == "clayton":
        return ClaytonCopula().theta_from_tau(k_tau)
    elif copula_name == "frank":
        return FrankCopula().theta_from_tau(k_tau)
    elif copula_name == "gumbel":
        return GumbelCopula().theta_from_tau(k_tau)
    else:
        raise NotImplementedError('Copula not implemented')
    
def theta_to_kendall_tau(copula_name, theta):
    if copula_name == "clayton":
        return ClaytonCopula().tau(theta)
    elif copula_name == "frank":
        return FrankCopula().tau(theta)
    elif copula_name == "gumbel":
        return GumbelCopula().tau(theta)
    else:
        raise NotImplementedError('Copula not implemented')
    
def multilabel_train_test_split(X, y, test_size, random_state=None):
    """Iteratively stratified train/test split
    (Add random_state to scikit-multilearn iterative_train_test_split function)
    See this paper for details: https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10
    """
    X, y = shuffle(X, y, random_state=random_state)
    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=test_size)
    return X_train, y_train, X_test, y_test

def make_stratified_split(
        df: pd.DataFrame,
        stratify_colname: str = 'event',
        frac_train: float = 0.5,
        frac_valid: float = 0.0,
        frac_test: float = 0.5,
        random_state: int = None
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    assert frac_train >= 0 and frac_valid >= 0 and frac_test >= 0, "Check train validation test fraction."
    frac_sum = frac_train + frac_valid + frac_test
    frac_train = frac_train / frac_sum
    frac_valid = frac_valid / frac_sum
    frac_test = frac_test / frac_sum

    X = df.values  # Contains all columns.
    columns = df.columns
    if stratify_colname == 'event':
        stra_lab = df[stratify_colname]
    elif stratify_colname == 'time':
        stra_lab = df[stratify_colname]
        bins = np.linspace(start=stra_lab.min(), stop=stra_lab.max(), num=20)
        stra_lab = np.digitize(stra_lab, bins, right=True).reshape(-1, 1)
    elif stratify_colname == "both":
        t = df["time"]
        bins = np.linspace(start=t.min(), stop=t.max(), num=20)
        t = np.digitize(t, bins, right=True)
        e = df["event"]
        stra_lab = np.stack([t, e], axis=1)
    else:
        raise ValueError("unrecognized stratify policy")

    x_train, _, x_temp, y_temp = multilabel_train_test_split(X, y=stra_lab, test_size=(1.0 - frac_train),
                                                             random_state=random_state)
    if frac_valid == 0:
        x_val, x_test = [], x_temp
    else:
        x_val, _, x_test, _ = multilabel_train_test_split(x_temp, y=y_temp,
                                                          test_size=frac_test / (frac_valid + frac_test),
                                                          random_state=random_state)
    df_train = pd.DataFrame(data=x_train, columns=columns)
    df_val = pd.DataFrame(data=x_val, columns=columns)
    df_test = pd.DataFrame(data=x_test, columns=columns)
    assert len(df) == len(df_train) + len(df_val) + len(df_test)
    return df_train, df_val, df_test

def compute_unique_counts(
        event: torch.Tensor,
        time: torch.Tensor,
        order: Optional[torch.Tensor] = None):
    n_samples = event.shape[0]

    if order is None:
        order = torch.argsort(time)

    uniq_times = torch.empty(n_samples, dtype=time.dtype, device=time.device)
    uniq_events = torch.empty(n_samples, dtype=torch.int, device=time.device)
    uniq_counts = torch.empty(n_samples, dtype=torch.int, device=time.device)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    uniq_times = uniq_times[:j]
    uniq_events = uniq_events[:j]
    uniq_counts = uniq_counts[:j]
    n_censored = uniq_counts - uniq_events

    # offset cumulative sum by one
    total_count = torch.cat([torch.tensor([0], device=uniq_counts.device), uniq_counts], dim=0)
    n_at_risk = n_samples - torch.cumsum(total_count, dim=0)

    return uniq_times, uniq_events, n_at_risk[:-1], n_censored

def make_monotonic(
        array: Union[torch.Tensor, np.ndarray, list]
):
    for i in range(len(array) - 1):
        if not array[i] >= array[i + 1]:
            array[i + 1] = array[i]
    return array