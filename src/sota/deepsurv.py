from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from pycox.models import DeepHitSingle
import torchtuples as tt
from pycox.models import DeepHit
from auton_survival.models.dsm import DeepSurvivalMachines
import torch
import numpy as np
import torch
import torch.nn as nn
import argparse
import pandas as pd
from typing import List, Tuple, Union
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
from utility.survival import compute_unique_counts, make_monotonic

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

class DeepSurv(nn.Module):
    def __init__(self, in_features: int, config: argparse.Namespace):
        super().__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        n_hidden=100
        
        # Shared parameters
        self.hidden = nn.Sequential(
            nn.Linear(in_features, n_hidden),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(n_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared embedding
        hidden = self.hidden(x)
        return self.fc1(hidden)

    def calculate_baseline_survival(self, x, t, e):
        outputs = self.forward(x)
        self.time_bins, self.cum_baseline_azhard, self.baseline_survival = calculate_baseline_hazard(outputs, t, e)

    def reset_parameters(self):
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}"

    def get_name(self):
        return self._get_name()

def train_deepsurv_model(
        model: nn.Module,
        data_train: pd.DataFrame,
        data_valid: pd.DataFrame,
        time_bins: NumericArrayLike,
        config: argparse.Namespace,
        random_state: int,
        reset_model: bool = True,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float64
) -> nn.Module:
    if config.verbose:
        print(f"Training {model.get_name()}: reset mode is {reset_model}, number of epochs is {config.num_epochs}, "
              f"learning rate is {config.lr}, C1 is {config.c1}, "
              f"batch size is {config.batch_size}, device is {device}.")
    
    batch_size = config.batch_size
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if reset_model:
        model.reset_parameters()

    model = model.to(device)
    model.train()
    best_val_nll = np.inf
    best_ep = -1

    pbar = trange(config.num_epochs, disable=not config.verbose)

    start_time = datetime.now()
    x_train, t_train, e_train = (torch.tensor(data_train.drop(["time", "event"], axis=1).values, dtype=dtype),
                                 torch.tensor(data_train["time"].values, dtype=dtype),
                                 torch.tensor(data_train["event"].values, dtype=dtype))
    x_val, t_val, e_val = (torch.tensor(data_valid.drop(["time", "event"], axis=1).values, dtype=dtype).to(device),
                           torch.tensor(data_valid["time"].values, dtype=dtype).to(device),
                           torch.tensor(data_valid["event"].values, dtype=dtype).to(device))

    train_loader = DataLoader(TensorDataset(x_train, t_train, e_train), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(x_val, t_val, e_val), batch_size=batch_size, shuffle=False)
    
    model.config.batch_size = batch_size

    for i in pbar:
        
        # Training
        total_nll_loss = 0
        num_batches = 0
        for xi, ti, ei in train_loader:
            xi, ti, ei = xi.to(device), ti.to(device), ei.to(device)
            
            # Check if there is at least one event (1) in ei
            if (ei == 1).sum() == 0:
                continue  # Skip the batch
    
            optimizer.zero_grad()
            y_pred = model.forward(xi)
            batch_loss = cox_nll(y_pred, 1, 0, ti, ei, model, C1=config.c1)

            batch_loss.backward()
            optimizer.step()
            
            total_nll_loss += batch_loss.item()
            num_batches += 1
        
        avg_nll_loss = total_nll_loss / num_batches if num_batches > 0 else float("inf")
        
        # Validation
        total_val_loss = 0
        num_val_batches = 0
        with torch.no_grad():
            for xi_val, ti_val, ei_val in valid_loader:
                xi_val, ti_val, ei_val = xi_val.to(device), ti_val.to(device), ei_val.to(device)

                # Check if there is at least one event (1) in ei_val
                if (ei_val == 1).sum() == 0:
                    continue  # Skip the batch

                logits_outputs = model.forward(xi_val)
                batch_val_loss = cox_nll(logits_outputs, 1, 0, ti_val, ei_val, model, C1=0)

                total_val_loss += batch_val_loss.item()
                num_val_batches += 1
            
        # Compute the average validation loss over all batches
        avg_val_nll = total_val_loss / num_val_batches if num_val_batches > 0 else float("inf")

        # Update progress bar
        pbar.set_description(f"[epoch {i + 1: 4}/{config.num_epochs}]")
        pbar.set_postfix_str(f"nll-loss = {avg_nll_loss:.4f}; "
                            f"Validation nll = {avg_val_nll:.4f};")
    
        # Early stopping logic
        if config.early_stop:
            if best_val_nll > avg_val_nll:
                best_val_nll = avg_val_nll
                best_ep = i
            if (i - best_ep) > config.patience:
                print(f"Validation loss converges at {best_ep}-th epoch.")
                break

    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training time: {training_time.total_seconds()}")
    # model.eval()
    model.calculate_baseline_survival(x_train.to(device), t_train.to(device), e_train.to(device))
    return model

def make_deepsurv_prediction(
        model: DeepSurv,
        x: torch.Tensor,
        config: argparse.Namespace,
        dtype: torch.dtype
):
    model.eval()
    start_time = datetime.now()
    with torch.no_grad():
        pred = model.forward(x)
        end_time = datetime.now()
        inference_time = end_time - start_time
        if config.verbose:
            print(f"Inference time: {inference_time.total_seconds()}")
        survival_curves = cox_survival(model.baseline_survival, pred, dtype)
        survival_curves = survival_curves.squeeze()

    time_bins = model.time_bins
    return survival_curves, time_bins

def cox_nll(
        risk_pred: torch.Tensor,
        precision: torch.Tensor,
        log_var: torch.Tensor,
        true_times: torch.Tensor,
        true_indicator: torch.Tensor,
        model: torch.nn.Module,
        C1: float
) -> torch.Tensor:
    """Computes the negative log-likelihood of a batch of model predictions.

    Parameters
    ----------
    risk_pred : torch.Tensor, shape (num_samples, )
        Risk prediction from Cox-based model. It means the relative hazard ratio: \beta * x.
    true_times : torch.Tensor, shape (num_samples, )
        Tensor with the censor/event time.
    true_indicator : torch.Tensor, shape (num_samples, )
        Tensor with the censor indicator.
    model
        PyTorch Module with at least `MTLR` layer.
    C1
        The L2 regularization strength.

    Returns
    -------
    torch.Tensor
        The negative log likelihood.
    """
    eps = 1e-20
    risk_pred = risk_pred.reshape(-1, 1)
    true_times = true_times.reshape(-1, 1)
    true_indicator = true_indicator.reshape(-1, 1)
    mask = torch.ones(true_times.shape[0], true_times.shape[0]).to(true_times.device)
    mask[(true_times.T - true_times) > 0] = 0
    max_risk = risk_pred.max()
    log_loss = torch.exp(risk_pred - max_risk) * mask
    log_loss = torch.sum(log_loss, dim=0)
    log_loss = torch.log(log_loss + eps).reshape(-1, 1) + max_risk
    # Sometimes in the batch we got all censoring data, so the denominator gets 0 and throw nan.
    # Solution: Consider increase the batch size. Afterall the nll should performed on the whole dataset.
    # Based on equation 2&3 in https://arxiv.org/pdf/1606.00931.pdf
    neg_log_loss = -torch.sum(precision * (risk_pred - log_loss) * true_indicator) / torch.sum(true_indicator) + log_var

    # L2 regularization
    for k, v in model.named_parameters():
        if "weight" in k:
            neg_log_loss += C1/2 * torch.norm(v, p=2)

    return neg_log_loss

def cox_survival(
        baseline_survival: torch.Tensor,
        linear_predictor: torch.Tensor,
        dtype: torch.dtype
) -> torch.Tensor:
    """
    Calculate the individual survival distributions based on the baseline survival curves and the liner prediction values.
    :param baseline_survival: (n_time_bins, )
    :param linear_predictor: (n_samples, n_data)
    :return:
    The invidual survival distributions. shape = (n_samples, n_time_bins)
    """
    n_sample = linear_predictor.shape[0]
    n_data = linear_predictor.shape[1]
    risk_score = torch.exp(linear_predictor)
    survival_curves = torch.empty((n_sample, n_data, baseline_survival.shape[0]), dtype=dtype).to(linear_predictor.device)
    for i in range(n_sample):
        for j in range(n_data):
            survival_curves[i, j, :] = torch.pow(baseline_survival, risk_score[i, j])
    return survival_curves

def calculate_baseline_hazard(
        logits: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Calculate the baseline cumulative hazard function and baseline survival function using Breslow estimator
    :param logits: logit outputs calculated from the Cox-based network using training data.
    :param time: Survival time of training data.
    :param event: Survival indicator of training data.
    :return:
    uniq_times: time bins correspond of the baseline hazard/survival.
    cum_baseline_hazard: cumulative baseline hazard
    baseline_survival: baseline survival curve.
    """
    risk_score = torch.exp(logits)
    order = torch.argsort(time)
    risk_score = risk_score[order]
    uniq_times, n_events, n_at_risk, _ = compute_unique_counts(event, time, order)

    divisor = torch.empty(n_at_risk.shape, dtype=torch.float, device=n_at_risk.device)
    value = torch.sum(risk_score)
    divisor[0] = value
    k = 0
    for i in range(1, len(n_at_risk)):
        d = n_at_risk[i - 1] - n_at_risk[i]
        value -= risk_score[k:(k + d)].sum()
        k += d
        divisor[i] = value

    assert k == n_at_risk[0] - n_at_risk[-1]

    hazard = n_events / divisor
    # Make sure the survival curve always starts at 1
    if 0 not in uniq_times:
        uniq_times = torch.cat([torch.tensor([0]).to(uniq_times.device), uniq_times], 0)
        hazard = torch.cat([torch.tensor([0]).to(hazard.device), hazard], 0)
    # TODO: torch.cumsum with cuda array will generate a non-monotonic array. Need to update when torch fix this bug
    # See issue: https://github.com/pytorch/pytorch/issues/21780
    baseline_hazard = hazard.cpu()
    cum_baseline_hazard = torch.cumsum(hazard.cpu(), dim=0).to(hazard.device)
    baseline_survival = torch.exp(- cum_baseline_hazard)
    if baseline_survival.isinf().any():
        print(f"Baseline survival contains \'inf\', need attention. \n"
              f"Baseline survival distribution: {baseline_survival}")
        last_zero = torch.where(baseline_survival == 0)[0][-1].item()
        baseline_survival[last_zero + 1:] = 0
    baseline_survival = make_monotonic(baseline_survival)
    return uniq_times, cum_baseline_hazard, baseline_survival