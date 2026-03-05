import os
import random
import torch
import pandas as pd
import numpy as np
import config as cfg
from SurvivalEVAL import SurvivalEvaluator
from scipy.interpolate import interp1d

from dgp import DGP_Weibull_linear, DGP_Weibull_nonlinear
from evaluator import DependentEvaluator
from sota.sksurv import make_cox_model
from utility.data import dotdict
from utility.experiment import _set_global_seeds, _simulate_uv_archimedean, _uv_seed
from utility.survival import (convert_to_structured, kendall_tau_to_theta, make_time_bins)

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

dtype = torch.float64
torch.set_default_dtype(dtype)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_cfg = {
    "alpha_e1": 19,
    "alpha_e2": 17,
    "gamma_e1": 6,
    "gamma_e2": 4,
    "n_samples": 10000,
    "n_features": 10,
}

def make_dep_censor_df_for_setting(
    *,
    X: torch.Tensor,
    dgp_event,
    dgp_cens,
    u: torch.Tensor,
    v: torch.Tensor):
    # sample times
    t_c = dgp_cens.rvs(X, u)          # numpy
    t_e = dgp_event.rvs(X, v)         # numpy

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

def make_train_test_split_indices(n: int, train_frac: float, split_seed: int):
    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    return perm[:n_train], perm[n_train:]

def censoring_bin(cr: float, bins, labels):
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if (cr >= lo) and (cr < hi or (i == len(bins) - 2 and cr <= hi)):
            return labels[i]
    return None

def pre_calibrate_alpha_c_mults_global(
    *,
    data_cfg,
    pilot_seeds,
    copula_names,
    k_tau,
    mult_grid,
    device,
    dtype,
    linear,
    hidden_dim,
    bins
):
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
        beta_cens = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1

        if linear:
            dgp_event = DGP_Weibull_linear(
                n_features, alpha_e, gamma_e, use_x=True,
                device=device, dtype=dtype, coeff=beta_event
            )
            cens_ctor = lambda alpha_c: DGP_Weibull_linear(
                n_features, alpha_c, gamma_c, use_x=True,
                device=device, dtype=dtype, coeff=beta_cens
            )
        else:
            g_event = torch.Generator(device=device); g_event.manual_seed(int(seed) + 10_000)
            dgp_event = DGP_Weibull_nonlinear(
                n_features=n_features, alpha=alpha_e, gamma=gamma_e,
                hidden_dim=hidden_dim, use_x=True, device=device, dtype=dtype,
                generator=g_event
            )
            g_cens = torch.Generator(device=device); g_cens.manual_seed(int(seed) + 20_000)
            dgp_cens_base = DGP_Weibull_nonlinear(
                n_features=n_features, alpha=alpha_c_base, gamma=gamma_c,
                hidden_dim=hidden_dim, use_x=True, device=device, dtype=dtype,
                generator=g_cens
            )
            cens_sd = dgp_cens_base.net.state_dict()
            cens_ctor = lambda alpha_c: DGP_Weibull_nonlinear(
                n_features=n_features, alpha=alpha_c, gamma=gamma_c,
                hidden_dim=hidden_dim, use_x=True, device=device, dtype=dtype,
                coeff=cens_sd
            )

        for copula_name in copula_names:
            uv_seed = _uv_seed(int(seed), float(k_tau), str(copula_name))
            u_np, v_np = _simulate_uv_archimedean(str(copula_name), n_samples, float(k_tau), uv_seed)
            u = torch.from_numpy(u_np).to(device=device, dtype=dtype)
            v = torch.from_numpy(v_np).to(device=device, dtype=dtype)

            t_e = dgp_event.rvs(X, v)

            for mult in mult_grid:
                alpha_c = alpha_c_base * float(mult)
                dgp_cens = cens_ctor(alpha_c)
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
                })

    calib = pd.DataFrame(rows)
    calib_df = (
        calib.groupby(["copula_name", "alpha_c_mult"], as_index=False)
             .agg(
                 censor_rate_mean=("censoring_rate", "mean"),
                 censor_rate_std=("censoring_rate", "std"),
                 alpha_c_used_mean=("alpha_c_used", "mean"),
             )
             .sort_values(["copula_name", "alpha_c_mult"])
             .reset_index(drop=True)
    )

    bins = list(bins)

    chosen_by_copula = {}
    for copula_name in copula_names:
        sub = calib_df[calib_df["copula_name"] == str(copula_name)].copy()

        chosen = []
        used = set()

        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            mid = (lo + hi) / 2.0

            cand = sub.copy()
            cand["in_bin"] = (cand["censor_rate_mean"] >= lo) & (
                (cand["censor_rate_mean"] < hi) if i < len(bins) - 2 else (cand["censor_rate_mean"] <= hi)
            )
            cand["dist_mid"] = (cand["censor_rate_mean"] - mid).abs()

            # IMPORTANT: sort within this copula only
            cand = cand.sort_values(
                ["in_bin", "dist_mid", "alpha_c_mult"],
                ascending=[False, True, True]
            )

            picked = None
            for m in cand["alpha_c_mult"].tolist():
                if m not in used:
                    picked = float(m)
                    break
            if picked is None:
                picked = float(cand.iloc[0]["alpha_c_mult"])

            chosen.append(picked)
            used.add(picked)

        chosen_by_copula[str(copula_name)] = chosen

    return chosen_by_copula, calib_df

def run_synthetic_metric_error_experiment(
    *,
    data_cfg,
    seeds,
    copula_names,
    k_taus,
    alpha_c_mults,
    device,
    dtype,
    train_frac=0.7,
    split_seed=0,
    num_points=10,
    linear=True,
    bins=(0.0, 0.10, 0.20, 0.30, 0.40, 0.50),
    labels=("0-10", "11-20", "21-30", "31-40", "41-50"),
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
        beta_cens = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1
        
        if linear:
            dgp_event = DGP_Weibull_linear(
                n_features, alpha_e, gamma_e,
                use_x=True, device=device, dtype=dtype,
                coeff=beta_event
            )
        else:
            g_event = torch.Generator(device=device); g_event.manual_seed(int(seed) + 10_000)
            dgp_event = DGP_Weibull_nonlinear(
                n_features=n_features, alpha=alpha_e, gamma=gamma_e,
                hidden_dim=32, use_x=True, device=device, dtype=dtype,
                generator=g_event
            )

        for copula_name in copula_names:
            for k_tau in k_taus:
                uv_seed = _uv_seed(int(seed), float(k_tau), str(copula_name))
                u_np, v_np = _simulate_uv_archimedean(str(copula_name), n_samples, float(k_tau), uv_seed)
                u = torch.from_numpy(u_np).to(device=device, dtype=dtype)
                v = torch.from_numpy(v_np).to(device=device, dtype=dtype)

                for mult in alpha_c_mults:
                    alpha_c = alpha_c_base * float(mult)

                    if linear:
                        dgp_cens = DGP_Weibull_linear(
                            n_features, alpha_c, gamma_c,
                            use_x=True, device=device, dtype=dtype,
                            coeff=beta_cens
                        )
                    else:
                        raise NotImplementedError("Nonlinear censor sweep: reuse same net weights, change alpha only.")

                    df = make_dep_censor_df_for_setting(X=X, dgp_event=dgp_event, dgp_cens=dgp_cens, u=u, v=v)

                    if len(df) != n_samples:
                        tr, te = make_train_test_split_indices(len(df), train_frac, split_seed)
                        df_train = df.iloc[tr].copy()
                        df_test  = df.iloc[te].copy()
                    else:
                        df_train = df.iloc[train_idx].copy()
                        df_test  = df.iloc[test_idx].copy()

                    censoring_rate = float(1.0 - df["event"].mean())
                    cbin = censoring_bin(censoring_rate, list(bins), list(labels))

                    true_test_time = df_test["true_time"].values
                    true_test_event = np.ones(df_test.shape[0], dtype=int)

                    y_train = convert_to_structured(df_train["time"], df_train["event"])
                    y_test  = convert_to_structured(df_test["time"],  df_test["event"])

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
                        fill_value=(1.0, surv[:, -1])  # left of min -> 1, right of max -> last value
                    )
                    S = spline(time_bins)
                    S = np.clip(S, 0.0, 1.0)
                    surv_on_grid = pd.DataFrame(S, columns=time_bins)
                    surv_on_grid[0.0] = 1.0
                                        
                    true_eval = SurvivalEvaluator(surv_on_grid, time_bins, true_test_time, true_test_event)
                    ibs_true = float(true_eval.integrated_brier_score(IPCW_weighted=False, num_points=num_points))

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
                        copula_name=str(copula_name), alpha=theta
                    )
                    ibs_dep_bguw = float(dep_eval.integrated_brier_score(num_points=num_points))

                    rows.append({
                        "seed": int(seed),
                        "copula_name": str(copula_name),
                        "k_tau": float(k_tau),
                        "alpha_c_mult": float(mult),
                        "alpha_c_used": float(alpha_c),
                        "gamma_c": float(gamma_c),
                        "n_samples": int(n_samples),
                        "n_features": int(n_features),
                        "censoring_rate": float(censoring_rate),
                        "censoring_bin": cbin,
                        "err_ipcw": abs(ibs_true - ibs_ipcw),
                        "err_dep_bguw": abs(ibs_true - ibs_dep_bguw),
                    })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    SEEDS = list(range(0, 10))
    COPULA_NAMES = ["clayton", "frank"]
    K_TAU = 0.5

    bins   = [i/10 for i in range(0, 11)]
    labels = [f"{10*i}-{10*(i+1)}" for i in range(0, 10)]
    
    mult_grid = np.concatenate([
        np.linspace(0.02, 0.20, 40),
        np.linspace(0.20, 1.00, 40),
        np.linspace(1.00, 4.00, 40),
    ]).tolist()

    pilot_seeds = list(range(0, 100))

    chosen_by_copula, calib_df = pre_calibrate_alpha_c_mults_global(
        data_cfg=data_cfg,
        pilot_seeds=pilot_seeds,
        copula_names=COPULA_NAMES,
        k_tau=K_TAU,
        mult_grid=mult_grid,
        device=device,
        dtype=dtype,
        linear=True,
        hidden_dim=32,
        bins=bins,
    )

    print("Chosen alpha_c_mults by copula:")
    for c in COPULA_NAMES:
        print(f"  {c}: {chosen_by_copula[c]}")

    # optional: quick check table for each copula
    for c in COPULA_NAMES:
        sub = calib_df[calib_df["copula_name"] == c].set_index("alpha_c_mult")
        check = sub.loc[chosen_by_copula[c]][["censor_rate_mean", "censor_rate_std"]]
        print(f"\nCheck bins for {c}:\n{check}")

    all_results = []
    for c in COPULA_NAMES:
        df_res = run_synthetic_metric_error_experiment(
            data_cfg=data_cfg,
            seeds=SEEDS,
            copula_names=[c],
            k_taus=[K_TAU],
            alpha_c_mults=chosen_by_copula[c],
            device=device,
            dtype=dtype,
            train_frac=0.7,
            split_seed=0,
            num_points=10,
            linear=True,
            bins=bins,
            labels=labels,
        )
        all_results.append(df_res)

    results_df = pd.concat(all_results, ignore_index=True)
    
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    out_path = f"{cfg.RESULTS_DIR}/synthetic_results_censoring.csv"
    results_df.to_csv(out_path, index=False)
    