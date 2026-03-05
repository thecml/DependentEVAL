import os
import random
import torch
import pandas as pd
import numpy as np
import config as cfg
from SurvivalEVAL import SurvivalEvaluator
from scipy.interpolate import interp1d
from scipy.stats import norm

from dgp import DGP_Weibull_linear
from evaluator import DependentEvaluator
from sota.sksurv import make_cox_model
from utility.data import dotdict
from utility.experiment import _set_global_seeds, _simulate_uv_archimedean, _uv_seed
from utility.survival import convert_to_structured, kendall_tau_to_theta, make_time_bins

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_cfg = {
    "alpha_e1": 19,
    "alpha_e2": 17,
    "gamma_e1": 6,
    "gamma_e2": 4,
    "n_samples": 10000,
    "n_features": 10,
}

def tau_to_rho_gaussian(k_tau: float) -> float:
    k_tau = float(k_tau)
    # valid for k_tau in [-1,1]
    rho = np.sin(np.pi * k_tau / 2.0)
    return float(np.clip(rho, -0.999, 0.999))

def kendall_to_pearson(tau: float) -> float:
    if not -1 <= tau <= 1:
        raise ValueError("Kendall's tau must be between -1 and 1.")
    rho = np.sin(np.pi * tau / 2)
    return rho

def make_train_test_split_indices(n: int, train_frac: float, split_seed: int):
    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    return perm[:n_train], perm[n_train:]

def make_dep_censor_df_for_setting(*, X, dgp_event, dgp_cens, u, v):
    t_c = dgp_cens.rvs(X, u)  # numpy
    t_e = dgp_event.rvs(X, v)  # numpy

    T = np.minimum(t_e, t_c)
    E = (t_e < t_c).astype(int)

    n_features = X.shape[1]
    df = pd.DataFrame(X.detach().cpu().numpy(), columns=[f"X{i}" for i in range(n_features)])
    df["time"] = np.where(T <= 0, 1.0, T)
    df["event"] = E
    df["true_time"] = np.where(t_e <= 0, 1.0, t_e)
    df["true_censor"] = np.where(t_c <= 0, 1.0, t_c)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["time", "true_time"]).reset_index(drop=True)
    return df

def sample_uv(*, copula_name: str, k_tau: float, seed: int, n: int, device, dtype):
    copula_name = None if copula_name is None else str(copula_name)
    k_tau = float(k_tau)

    # One deterministic seed per (seed, tau, copula)
    base_seed = int(_uv_seed(int(seed), float(k_tau), str(copula_name)))
    rng = np.random.default_rng(base_seed)

    # Independence branch (still seeded by base_seed above)
    if copula_name is None or k_tau == 0.0:
        u_np = rng.uniform(0.0, 1.0, n)
        v_np = rng.uniform(0.0, 1.0, n)

    elif copula_name in ["clayton", "frank"]:
        # keep your existing archimedean simulator
        uv_seed = _uv_seed(int(seed), float(k_tau), copula_name)
        u_np, v_np = _simulate_uv_archimedean(copula_name, n, float(k_tau), uv_seed)

    elif copula_name == "gaussian":
        rho = tau_to_rho_gaussian(k_tau)

        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)
        x = z1
        y = rho * z1 + np.sqrt(1.0 - rho * rho) * z2

        u_np = norm.cdf(x)
        v_np = norm.cdf(y)

    else:
        raise ValueError(f"Unknown copula_name={copula_name}")

    u = torch.from_numpy(np.asarray(u_np)).to(device=device, dtype=dtype)
    v = torch.from_numpy(np.asarray(v_np)).to(device=device, dtype=dtype)
    return u, v

def tau_to_rho_gaussian(k_tau: float) -> float:
    k_tau = float(k_tau)
    rho = np.sin(np.pi * k_tau / 2.0)
    return float(np.clip(rho, -0.999, 0.999))

def sample_uv_gaussian(*, n: int, k_tau: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    rho = tau_to_rho_gaussian(k_tau)

    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    x = z1
    y = rho * z1 + np.sqrt(1.0 - rho * rho) * z2

    u = norm.cdf(x)
    v = norm.cdf(y)
    return u, v

def assumed_setting_for_experiment(exp: str, dgp_copula: str, dgp_tau: float):
    exp = str(exp)
    dgp_copula = str(dgp_copula)
    dgp_tau = float(dgp_tau)

    if exp == "family":
        if dgp_copula == "clayton":
            return "frank", dgp_tau
        if dgp_copula == "frank":
            return "clayton", dgp_tau
        # do NOT run family for gaussian DGP
        return None, None

    if exp == "dep":
        if dgp_copula in ["clayton", "frank"]:
            return dgp_copula, 0.8 - dgp_tau
        # do NOT run dep for gaussian DGP
        return None, None

    if exp == "gaussian":
        if dgp_copula == "gaussian":
            return "clayton", dgp_tau
        return None, None

    raise ValueError(exp)

def calibrate_alpha_c_mults_by_tau(
    *,
    data_cfg,
    pilot_seeds,
    copula_names,
    k_taus,
    mult_grid,
    target_censoring,   # e.g. 0.50 or 0.70
    device,
    dtype,
    linear=True,
    hidden_dim=32,
):
    """
    For each (copula, k_tau), choose alpha_c_mult that makes censoring_rate close to target_censoring
    averaged across pilot_seeds.
    Returns:
      chosen[(copula, k_tau)] = best_mult
      calib_df with mean/std censoring per candidate mult
    """
    assert linear, "Nonlinear version not implemented here (same idea, but reuse weights and change alpha)."

    n_samples  = int(data_cfg["n_samples"])
    n_features = int(data_cfg["n_features"])

    alpha_c_base = float(data_cfg["alpha_e1"])
    gamma_c      = float(data_cfg["gamma_e1"])
    alpha_e      = float(data_cfg["alpha_e2"])
    gamma_e      = float(data_cfg["gamma_e2"])

    rows = []

    for seed in pilot_seeds:
        _set_global_seeds(int(seed))

        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

        X = torch.rand((n_samples, n_features), generator=g, device=device, dtype=dtype)

        beta_event = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1
        beta_cens  = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1

        dgp_event = DGP_Weibull_linear(
            n_features, alpha_e, gamma_e, use_x=True,
            device=device, dtype=dtype, coeff=beta_event
        )

        for copula_name in copula_names:
            for k_tau in k_taus:
                # sample (u,v)
                u, v = sample_uv(
                    copula_name=str(copula_name),
                    k_tau=float(k_tau),
                    seed=int(seed),
                    n=n_samples,
                    device=device,
                    dtype=dtype,
                )

                # event times are fixed for this (seed,copula,tau)
                t_e = dgp_event.rvs(X, v)

                for mult in mult_grid:
                    alpha_c = alpha_c_base * float(mult)
                    dgp_cens = DGP_Weibull_linear(
                        n_features, alpha_c, gamma_c, use_x=True,
                        device=device, dtype=dtype, coeff=beta_cens
                    )
                    t_c = dgp_cens.rvs(X, u)

                    E = (t_e < t_c).astype(np.float64)
                    censoring_rate = float(1.0 - E.mean())

                    rows.append({
                        "seed": int(seed),
                        "copula_name": str(copula_name),
                        "k_tau": float(k_tau),
                        "alpha_c_mult": float(mult),
                        "alpha_c_used": float(alpha_c),
                        "censoring_rate": censoring_rate,
                        "target": float(target_censoring),
                        "abs_err_to_target": abs(censoring_rate - float(target_censoring)),
                    })

    calib = pd.DataFrame(rows)

    calib_df = (
        calib.groupby(["copula_name", "k_tau", "alpha_c_mult"], as_index=False)
             .agg(
                 censor_rate_mean=("censoring_rate", "mean"),
                 censor_rate_std=("censoring_rate", "std"),
                 abs_err_mean=("abs_err_to_target", "mean"),
                 alpha_c_used_mean=("alpha_c_used", "mean"),
             )
             .sort_values(["copula_name", "k_tau", "abs_err_mean", "alpha_c_mult"])
             .reset_index(drop=True)
    )

    chosen = {}
    for copula_name in copula_names:
        for k_tau in k_taus:
            sub = calib_df[(calib_df["copula_name"] == str(copula_name)) & (calib_df["k_tau"] == float(k_tau))]
            best = float(sub.iloc[0]["alpha_c_mult"])
            chosen[(str(copula_name), float(k_tau))] = best

    return chosen, calib_df

def run_wrong_copula_experiment(
    *,
    data_cfg,
    seeds,
    dgp_copulas,
    k_taus,
    alpha_c_mult_by_setting,
    experiments,
    device,
    dtype,
    train_frac=0.7,
    split_seed=0,
    num_points=10,
):
    rows = []

    n_samples = int(data_cfg["n_samples"])
    n_features = int(data_cfg["n_features"])

    alpha_c_base = float(data_cfg["alpha_e1"])
    gamma_c = float(data_cfg["gamma_e1"])
    alpha_e = float(data_cfg["alpha_e2"])
    gamma_e = float(data_cfg["gamma_e2"])

    features = [f"X{i}" for i in range(n_features)]
    train_idx, test_idx = make_train_test_split_indices(n_samples, train_frac, split_seed)

    for seed in seeds:
        _set_global_seeds(int(seed))
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

        X = torch.rand((n_samples, n_features), generator=g, device=device, dtype=dtype)

        beta_event = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1
        beta_cens  = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1

        dgp_event = DGP_Weibull_linear(
            n_features, alpha_e, gamma_e, use_x=True,
            device=device, dtype=dtype, coeff=beta_event
        )

        for dgp_copula in dgp_copulas:
            for k_tau in k_taus:
                dgp_copula = str(dgp_copula)
                k_tau = float(k_tau)

                # calibrated censoring multiplier per DGP setting (recommended)
                mult = float(alpha_c_mult_by_setting[(dgp_copula, k_tau)])
                alpha_c = alpha_c_base * mult

                dgp_cens = DGP_Weibull_linear(
                    n_features, alpha_c, gamma_c, use_x=True,
                    device=device, dtype=dtype, coeff=beta_cens
                )

                # sample (u,v) from the DGP copula
                u, v = sample_uv(
                    copula_name=dgp_copula, k_tau=k_tau, seed=seed, n=n_samples,
                    device=device, dtype=dtype
                )

                df = make_dep_censor_df_for_setting(X=X, dgp_event=dgp_event, dgp_cens=dgp_cens, u=u, v=v)

                # keep split logic identical to correct experiment
                if len(df) != n_samples:
                    tr, te = make_train_test_split_indices(len(df), train_frac, split_seed)
                    df_train = df.iloc[tr].copy()
                    df_test  = df.iloc[te].copy()
                else:
                    df_train = df.iloc[train_idx].copy()
                    df_test  = df.iloc[test_idx].copy()

                censoring_rate = float(1.0 - df["event"].mean())

                true_test_time = df_test["true_time"].values
                true_test_event = np.ones(df_test.shape[0], dtype=int)

                y_train = convert_to_structured(df_train["time"], df_train["event"])
                X_train = df_train[features]
                X_test  = df_test[features]

                time_bins = make_time_bins(df_train["true_time"].values, event=None, dtype=dtype).to(device)
                time_bins = torch.cat((torch.tensor([0.0], device=device, dtype=dtype), time_bins)).cpu().numpy()
                t_star = np.quantile(df_test["true_time"].values, 0.9)
                time_bins = time_bins[time_bins <= t_star]

                config = dotdict(cfg.COXPH_PARAMS)
                model = make_cox_model(config)
                model.fit(X_train, y_train)

                surv_fns = model.predict_survival_function(X_test)
                surv = np.row_stack([fn(model.unique_times_) for fn in surv_fns])

                spline = interp1d(
                    model.unique_times_, surv, kind="linear",
                    bounds_error=False,
                    fill_value=(1.0, surv[:, -1]),
                )
                S = np.clip(spline(time_bins), 0.0, 1.0)
                surv_on_grid = pd.DataFrame(S, columns=time_bins)
                surv_on_grid[0.0] = 1.0

                # "Truth" uses true event times (no censoring)
                true_eval = SurvivalEvaluator(surv_on_grid, time_bins, true_test_time, true_test_event)
                ibs_true = float(true_eval.integrated_brier_score(IPCW_weighted=False, num_points=num_points))

                # IPCW baseline
                ipcw_eval = SurvivalEvaluator(
                    surv_on_grid, time_bins,
                    df_test["time"].values, df_test["event"].values,
                    df_train["time"].values, df_train["event"].values,
                )
                ibs_ipcw = float(ipcw_eval.integrated_brier_score(num_points=num_points))

                for exp in experiments:
                    assumed_copula, assumed_tau = assumed_setting_for_experiment(
                        exp=exp, dgp_copula=dgp_copula, dgp_tau=k_tau
                    )
                    
                    if assumed_copula is None:
                        continue

                    # NOTE: this is the *assumed* copula parameterization, even if wrong
                    theta = kendall_tau_to_theta(str(assumed_copula), float(assumed_tau))

                    dep_eval = DependentEvaluator(
                        surv_on_grid, time_bins,
                        df_test["time"].values, df_test["event"].values,
                        df_train["time"].values, df_train["event"].values,
                        copula_name=str(assumed_copula),
                        alpha=theta,
                    )
                    ibs_dep_bguw = float(dep_eval.integrated_brier_score(method="BG_UW", num_points=num_points))

                    rows.append({
                        "experiment": str(exp),
                        "seed": int(seed),
                        "dgp_copula": str(dgp_copula),
                        "k_tau": float(k_tau),
                        "assumed_copula": str(assumed_copula),
                        "assumed_k_tau": float(assumed_tau),
                        "alpha_c_mult": float(mult),
                        "alpha_c_used": float(alpha_c),
                        "censoring_rate": float(censoring_rate),
                        "err_ipcw": abs(ibs_true - ibs_ipcw),
                        "err_dep_bguw": abs(ibs_true - ibs_dep_bguw),
                    })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    SEEDS = list(range(0, 10))
    PILOT_SEEDS = list(range(0, 10))
    COPULA_NAMES = ["clayton", "frank", "gaussian"]
    K_TAU = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    DGP_COPULAS = ["clayton", "frank", "gaussian"]
    experiments = ["family", "dep", "gaussian"]

    # use a target censoring rate
    TARGET_CENSOR = 0.50

    # grid of multipliers to search over
    mult_grid = np.concatenate([
        np.linspace(0.05, 0.50, 40),
        np.linspace(0.50, 2.00, 60),
        np.linspace(2.00, 6.00, 40),
    ]).tolist()

    alpha_c_mult_by_setting, calib_df = calibrate_alpha_c_mults_by_tau(
        data_cfg=data_cfg,
        pilot_seeds=PILOT_SEEDS,
        copula_names=COPULA_NAMES,
        k_taus=K_TAU,
        mult_grid=mult_grid,
        target_censoring=TARGET_CENSOR,
        device=device,
        dtype=dtype,
        linear=True,
    )

    results_df = run_wrong_copula_experiment(
        data_cfg=data_cfg,
        seeds=SEEDS,
        dgp_copulas=DGP_COPULAS,
        k_taus=K_TAU,
        alpha_c_mult_by_setting=alpha_c_mult_by_setting,
        experiments=experiments,
        device=device,
        dtype=dtype,
        train_frac=0.7,
        split_seed=0,
        num_points=10,
    )
    
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    filename = f"{cfg.RESULTS_DIR}/synthetic_results_wrong_copula.csv"
    results_df.to_csv(filename, index=False)
