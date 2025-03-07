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

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

class CauseSpecificNet(torch.nn.Module):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """
    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                out_features, batch_norm=True, dropout=None):
        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features, num_nodes_shared[:-1], num_nodes_shared[-1],
            batch_norm, dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net = tt.practical.MLPVanilla(
                num_nodes_shared[-1], num_nodes_indiv, out_features,
                batch_norm, dropout,
            )
            self.risk_nets.append(net)

    def forward(self, input):
        out = self.shared_net(input)
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)
        return out

def make_deephit_single(in_features, out_features, time_bins, device, config):
    num_nodes = config['num_nodes_shared']
    batch_norm = config['batch_norm']
    dropout = config['dropout']
    labtrans = DeepHitSingle.label_transform(time_bins)
    net = tt.practical.MLPVanilla(in_features=in_features, num_nodes=num_nodes,
                                  out_features=labtrans.out_features, batch_norm=batch_norm,
                                  dropout=dropout)
    model = DeepHitSingle(net, tt.optim.Adam, device=device, alpha=0.2, sigma=0.1,
                          duration_index=labtrans.cuts)
    model.label_transform = labtrans
    return model

def train_deephit_model(model, x_train, y_train, valid_data, config):
    epochs = config['epochs']
    batch_size = config['batch_size']
    verbose = config['verbose']
    if config['early_stop']:
        callbacks = [tt.callbacks.EarlyStopping(patience=config['patience'])]
    else:
        callbacks = []
    model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=valid_data)
    return model