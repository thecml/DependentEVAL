import numpy as np
import torch
import random
from pycop import simulation
from utility.survival import kendall_tau_to_theta

def _uv_seed(seed: int, tau: float, copula_name: str) -> int:
    # include copula_name so clayton/frank don't share the same uv stream
    cop_hash = abs(hash(copula_name)) % 1_000_000
    return int(seed * 1_000_003 + round(float(tau) * 10_000) + cop_hash * 31)

def _set_global_seeds(seed: int):
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    # optional determinism (can slow down on GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _simulate_uv_archimedean(copula_name: str, n: int, tau: float, seed: int):
    # deterministic for this (seed,tau,copula)
    np.random.seed(int(seed))
    theta = kendall_tau_to_theta(copula_name, float(tau))
    u_np, v_np = simulation.simu_archimedean(copula_name, 2, n, theta=theta)
    return u_np.reshape(-1), v_np.reshape(-1)
