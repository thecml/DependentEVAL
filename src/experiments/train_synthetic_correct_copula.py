import os
import random

import numpy as np
import pandas as pd
import torch
import config as cfg
from SurvivalEVAL import SurvivalEvaluator
from scipy.interpolate import interp1d
from scipy.stats import norm

from dgp import DGP_Weibull_linear
from evaluators import DependentEvaluator
from sota.sksurv import make_cox_model
from utility.data import dotdict
from utility.experiment import _set_global_seeds, _uv_seed
from utility.survival import convert_to_structured, kendall_tau_to_theta, make_time_bins

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_cfg = {
    "alpha_e1": 19,  # censor alpha base
    "alpha_e2": 17,  # event alpha
    "gamma_e1": 6,   # censor gamma
    "gamma_e2": 4,   # event gamma
    "n_samples": 10000,
    "n_features": 10,
}


def tau_to_rho_gaussian(k_tau: float) -> float:
    """Kendall's tau to Gaussian copula Pearson rho."""
    k_tau = float(k_tau)
    rho = np.sin(np.pi * k_tau / 2.0)
    return float(np.clip(rho, -0.999, 0.999))


def make_train_test_split_indices(n: int, train_frac: float, split_seed: int):
    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    return perm[:n_train], perm[n_train:]


def make_event_uniforms(*, seed: int, n: int, device, dtype):
    """
    Fixed event uniforms per seed.

    These are reused across all copula families and all Kendall tau values.
    Hence changing tau changes only the censoring mechanism, not event times.
    """
    rng = np.random.default_rng(int(seed) + 123_456_789)
    v_np = rng.uniform(0.0, 1.0, int(n))
    v_np = np.clip(v_np, 1e-12, 1.0 - 1e-12)
    return torch.from_numpy(v_np).to(device=device, dtype=dtype)


def sample_censor_uniform_given_event_uniform(
    *,
    copula_name: str,
    k_tau: float,
    seed: int,
    event_v: torch.Tensor,
    device,
    dtype,
):
    """
    Sample censoring uniforms U conditional on fixed event uniforms V.

    V is fixed across tau; U changes with the copula dependence. This isolates
    the effect of event--censor dependence while keeping event times fixed.
    """
    copula_name = str(copula_name)
    k_tau = float(k_tau)
    n = int(event_v.numel())

    v = event_v.detach().cpu().numpy().astype(float)
    v = np.clip(v, 1e-12, 1.0 - 1e-12)

    rng = np.random.default_rng(int(_uv_seed(int(seed), float(k_tau), copula_name)) + 987_654_321)
    w = rng.uniform(0.0, 1.0, n)
    w = np.clip(w, 1e-12, 1.0 - 1e-12)

    if k_tau == 0.0:
        u = w

    elif copula_name == "gaussian":
        rho = tau_to_rho_gaussian(k_tau)
        z_v = norm.ppf(v)
        z_w = norm.ppf(w)
        z_u = rho * z_v + np.sqrt(1.0 - rho * rho) * z_w
        u = norm.cdf(z_u)

    elif copula_name == "clayton":
        theta = float(kendall_tau_to_theta("clayton", k_tau))
        theta = max(theta, 1e-12)
        # w = dC(u,v)/dv = v^(-theta-1) * A^(-1/theta-1),
        # A = u^(-theta) + v^(-theta) - 1. Solve for u.
        A = (w * (v ** (theta + 1.0))) ** (-theta / (1.0 + theta))
        u_neg_theta = A - (v ** (-theta)) + 1.0
        u_neg_theta = np.maximum(u_neg_theta, 1e-12)
        u = u_neg_theta ** (-1.0 / theta)

    elif copula_name == "frank":
        theta = float(kendall_tau_to_theta("frank", k_tau))
        theta = max(theta, 1e-12)
        # For Frank, solve w = dC(u,v)/dv for a = exp(-theta*u).
        b = np.exp(-theta * v)
        d = np.exp(-theta) - 1.0
        denom = b - w * (b - 1.0)
        denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12, denom)
        a = 1.0 + (w * d / denom)
        a = np.clip(a, np.exp(-theta) + 1e-12, 1.0 - 1e-12)
        u = -np.log(a) / theta

    else:
        raise ValueError(f"Unknown copula_name={copula_name}")

    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return torch.from_numpy(u).to(device=device, dtype=dtype)


def make_observed_df(*, X, true_event_time, dgp_cens, u_censor):
    t_c = dgp_cens.rvs(X, u_censor)
    t_e = np.asarray(true_event_time, dtype=float)

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


def calibrate_alpha_c_mults_by_tau(
    *,
    data_cfg,
    pilot_seeds,
    copula_names,
    k_taus,
    mult_grid,
    target_censoring,
    device,
    dtype,
    linear=True,
    hidden_dim=32,
):
    """
    Seed-specific calibration with fixed event uniforms.

    Returns:
      chosen[(seed, copula, k_tau)] = best_mult
      calib_df = aggregated calibration summary
      selected_calib_df = chosen row per seed/copula/tau
    """
    assert linear, "Nonlinear version not implemented here."

    n_samples = int(data_cfg["n_samples"])
    n_features = int(data_cfg["n_features"])

    alpha_c_base = float(data_cfg["alpha_e1"])
    gamma_c = float(data_cfg["gamma_e1"])
    alpha_e = float(data_cfg["alpha_e2"])
    gamma_e = float(data_cfg["gamma_e2"])

    rows = []

    for seed in pilot_seeds:
        _set_global_seeds(int(seed))

        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

        X = torch.rand((n_samples, n_features), generator=g, device=device, dtype=dtype)
        beta_event = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1
        beta_cens = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1

        dgp_event = DGP_Weibull_linear(
            n_features, alpha_e, gamma_e, use_x=True,
            device=device, dtype=dtype, coeff=beta_event,
        )

        v_event = make_event_uniforms(seed=int(seed), n=n_samples, device=device, dtype=dtype)
        true_event_time = dgp_event.rvs(X, v_event)

        for copula_name in copula_names:
            for k_tau in k_taus:
                u_censor = sample_censor_uniform_given_event_uniform(
                    copula_name=str(copula_name),
                    k_tau=float(k_tau),
                    seed=int(seed),
                    event_v=v_event,
                    device=device,
                    dtype=dtype,
                )

                for mult in mult_grid:
                    alpha_c = alpha_c_base * float(mult)
                    dgp_cens = DGP_Weibull_linear(
                        n_features, alpha_c, gamma_c, use_x=True,
                        device=device, dtype=dtype, coeff=beta_cens,
                    )
                    t_c = dgp_cens.rvs(X, u_censor)
                    E = (true_event_time < t_c).astype(np.float64)
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
    selected_rows = []
    for seed in pilot_seeds:
        for copula_name in copula_names:
            for k_tau in k_taus:
                sub = calib[
                    (calib["seed"] == int(seed))
                    & (calib["copula_name"] == str(copula_name))
                    & (calib["k_tau"] == float(k_tau))
                ].copy()
                if sub.empty:
                    raise RuntimeError(
                        f"Missing calibration rows for seed={seed}, copula={copula_name}, tau={k_tau}"
                    )
                best_row = sub.sort_values(["abs_err_to_target", "alpha_c_mult"]).iloc[0]
                best = float(best_row["alpha_c_mult"])
                chosen[(int(seed), str(copula_name), float(k_tau))] = best
                selected_rows.append(best_row.to_dict())

    selected_calib_df = pd.DataFrame(selected_rows)
    return chosen, calib_df, selected_calib_df


def fit_oracle_cox_and_predict(*, X, true_event_time, train_idx, test_idx, features, dtype, device, num_points):
    """Fit one Cox model per seed on oracle uncensored training event times."""
    df_oracle = pd.DataFrame(X.detach().cpu().numpy(), columns=features)
    df_oracle["true_time"] = np.where(np.asarray(true_event_time) <= 0, 1.0, true_event_time)
    df_oracle["event"] = np.ones(df_oracle.shape[0], dtype=int)

    df_train = df_oracle.iloc[train_idx].copy()
    df_test = df_oracle.iloc[test_idx].copy()

    y_train_oracle = convert_to_structured(df_train["true_time"], df_train["event"])

    config = dotdict(cfg.COXPH_PARAMS)
    model = make_cox_model(config)
    model.fit(df_train[features], y_train_oracle)

    time_bins = make_time_bins(df_train["true_time"].values, event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0.0], device=device, dtype=dtype), time_bins)).cpu().numpy()
    t_star = np.quantile(df_test["true_time"].values, 0.9)
    time_bins = time_bins[time_bins <= t_star]

    surv_fns = model.predict_survival_function(df_test[features])
    surv = np.row_stack([fn(model.unique_times_) for fn in surv_fns])

    spline = interp1d(
        model.unique_times_, surv, kind="linear",
        bounds_error=False,
        fill_value=(1.0, surv[:, -1]),
    )
    S = np.clip(spline(time_bins), 0.0, 1.0)
    surv_on_grid = pd.DataFrame(S, columns=time_bins)
    surv_on_grid[0.0] = 1.0

    true_test_time = df_test["true_time"].values
    true_test_event = np.ones(df_test.shape[0], dtype=int)
    true_eval = SurvivalEvaluator(surv_on_grid, time_bins, true_test_time, true_test_event)
    ibs_true = float(true_eval.integrated_brier_score(IPCW_weighted=False, num_points=num_points))

    return surv_on_grid, time_bins, ibs_true


def run_bias_vs_tau_experiment(
    *,
    data_cfg,
    seeds,
    copula_names,
    k_taus,
    alpha_c_mult_by_setting,
    device,
    dtype,
    train_frac=0.7,
    split_seed=0,
    num_points=10,
    linear=True,
    hidden_dim=32,
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

        if linear:
            beta_event = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1
            beta_cens = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1
            dgp_event = DGP_Weibull_linear(
                n_features, alpha_e, gamma_e, use_x=True,
                device=device, dtype=dtype, coeff=beta_event,
            )
        else:
            raise NotImplementedError("Add the nonlinear version later.")

        v_event = make_event_uniforms(seed=int(seed), n=n_samples, device=device, dtype=dtype)
        true_event_time = dgp_event.rvs(X, v_event)

        surv_on_grid, time_bins, ibs_true = fit_oracle_cox_and_predict(
            X=X,
            true_event_time=true_event_time,
            train_idx=train_idx,
            test_idx=test_idx,
            features=features,
            dtype=dtype,
            device=device,
            num_points=num_points,
        )

        for copula_name in copula_names:
            for k_tau in k_taus:
                mult = float(alpha_c_mult_by_setting[(int(seed), str(copula_name), float(k_tau))])
                alpha_c = alpha_c_base * mult

                dgp_cens = DGP_Weibull_linear(
                    n_features, alpha_c, gamma_c, use_x=True,
                    device=device, dtype=dtype, coeff=beta_cens,
                )

                u_censor = sample_censor_uniform_given_event_uniform(
                    copula_name=str(copula_name),
                    k_tau=float(k_tau),
                    seed=int(seed),
                    event_v=v_event,
                    device=device,
                    dtype=dtype,
                )

                df = make_observed_df(X=X, true_event_time=true_event_time, dgp_cens=dgp_cens, u_censor=u_censor)
                if len(df) != n_samples:
                    raise RuntimeError(
                        f"Unexpected row drop after simulation: len(df)={len(df)} != n_samples={n_samples}. "
                        "This would break the fixed train/test split."
                    )

                df_train = df.iloc[train_idx].copy()
                df_test = df.iloc[test_idx].copy()

                censoring_rate = float(1.0 - df["event"].mean())

                ipcw_eval = SurvivalEvaluator(
                    surv_on_grid, time_bins,
                    df_test["time"].values, df_test["event"].values,
                    df_train["time"].values, df_train["event"].values,
                )
                ibs_ipcw = float(ipcw_eval.integrated_brier_score(num_points=num_points))

                theta = kendall_tau_to_theta(str(copula_name), float(k_tau))
                dep_eval = DependentEvaluator(
                    surv_on_grid, time_bins,
                    df_test["time"].values, df_test["event"].values,
                    df_train["time"].values, df_train["event"].values,
                    copula_name=str(copula_name), alpha=theta,
                )
                ibs_dep_bguw = float(dep_eval.integrated_brier_score(method="BG_UW", num_points=num_points))

                bias_ipcw = ibs_ipcw - ibs_true
                bias_dep_bguw = ibs_dep_bguw - ibs_true

                rows.append({
                    "prediction_model": "oracle_fixed_per_seed",
                    "seed": int(seed),
                    "copula_name": str(copula_name),
                    "k_tau": float(k_tau),
                    "alpha_c_mult": float(mult),
                    "alpha_c_used": float(alpha_c),
                    "censoring_rate": float(censoring_rate),
                    "ibs_true": ibs_true,
                    "ibs_ipcw": ibs_ipcw,
                    "ibs_dep_bguw": ibs_dep_bguw,
                    "bias_ipcw": bias_ipcw,
                    "bias_dep_bguw": bias_dep_bguw,
                    "err_ipcw": abs(bias_ipcw),
                    "err_dep_bguw": abs(bias_dep_bguw),
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    SEEDS = list(range(0, 10))
    PILOT_SEEDS = list(range(0, 10))
    COPULA_NAMES = ["clayton", "frank"]
    K_TAU = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    TARGET_CENSOR = 0.50

    mult_grid = np.concatenate([
        np.linspace(0.05, 0.50, 40),
        np.linspace(0.50, 2.00, 60),
        np.linspace(2.00, 6.00, 40),
    ]).tolist()

    alpha_c_mult_by_setting, calib_df, selected_calib_df = calibrate_alpha_c_mults_by_tau(
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

    print("Seed-specific censoring calibration summary:")
    print(
        selected_calib_df.groupby(["copula_name", "k_tau"], as_index=False)
        .agg(
            censor_rate_mean=("censoring_rate", "mean"),
            censor_rate_std=("censoring_rate", "std"),
            abs_err_mean=("abs_err_to_target", "mean"),
            alpha_c_mult_min=("alpha_c_mult", "min"),
            alpha_c_mult_max=("alpha_c_mult", "max"),
        )
        .to_string(index=False)
    )

    results_df = run_bias_vs_tau_experiment(
        data_cfg=data_cfg,
        seeds=SEEDS,
        copula_names=COPULA_NAMES,
        k_taus=K_TAU,
        alpha_c_mult_by_setting=alpha_c_mult_by_setting,
        device=device,
        dtype=dtype,
        train_frac=0.7,
        split_seed=0,
        num_points=10,
        linear=True,
    )

    print("Actual censoring rates in generated results:")
    print(
        results_df.groupby(["copula_name", "k_tau"], as_index=False)
        .agg(
            censoring_rate_mean=("censoring_rate", "mean"),
            censoring_rate_std=("censoring_rate", "std"),
            censoring_rate_min=("censoring_rate", "min"),
            censoring_rate_max=("censoring_rate", "max"),
            ibs_true_min=("ibs_true", "min"),
            ibs_true_max=("ibs_true", "max"),
        )
        .to_string(index=False)
    )

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    results_df.to_csv(f"{cfg.RESULTS_DIR}/synthetic_results_correct_copula.csv", index=False)
