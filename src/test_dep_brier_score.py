#!/usr/bin/env python3
"""Synthetic smoke test for ``DependentEvaluator.brier_score``.

This file is both:

1. a small command-line diagnostic::

       python test_dependent_brier_score.py --n-samples 2000 --tau 0.5

2. a pytest-compatible smoke test::

       pytest -q test_dependent_brier_score.py

It follows the same data-generating process as
``train_synthetic_correct_copula.py`` but runs only one seed, one copula and
one dependence level. Predictions are fitted using the uncensored event times,
so the Brier score computed from the latent event times is an oracle reference.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.integrate import trapezoid

from dgp import DGP_Weibull_linear
from evaluators import DependentEvaluator
from experiments.train_synthetic_correct_copula import (
    make_event_uniforms,
    make_observed_df,
    make_train_test_split_indices,
    sample_censor_uniform_given_event_uniform,
)
from utility.metrics import predict_multi_probabilities_from_curve
from utility.survival import convert_to_structured, kendall_tau_to_theta
from sota.sksurv import make_cox_model
from utility.data import dotdict
import config as cfg

def _fit_oracle_cox_and_predict(
    X: torch.Tensor,
    true_event_time: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the oracle Cox model without assuming wrapper attributes.

    ``make_cox_model`` currently returns ``RobustCoxPHSurvivalAnalysis``.
    Its fitted scikit-survival model owns ``unique_times_``; the wrapper does
    not expose that attribute.  We therefore construct an explicit grid from
    the oracle training event times and evaluate the returned survival
    functions directly on that grid.
    """
    true_event_time = np.asarray(true_event_time, dtype=float)
    X_np = X.detach().cpu().numpy()

    oracle = pd.DataFrame(X_np, columns=features)
    oracle["true_time"] = np.maximum(true_event_time, 1e-12)
    oracle["event"] = 1

    train = oracle.iloc[train_idx]
    test = oracle.iloc[test_idx]
    y_train = convert_to_structured(train["true_time"], train["event"])

    model = make_cox_model(dotdict(cfg.COXPH_PARAMS))
    model.fit(train[features], y_train)

    # Stay inside the fitted baseline-hazard support and keep the grid modest.
    train_times = np.sort(train["true_time"].to_numpy(dtype=float))
    upper = float(min(train_times[-1], np.quantile(test["true_time"], 0.90)))
    supported = train_times[train_times <= upper]
    if supported.size > 250:
        q = np.linspace(0.0, 1.0, 250)
        supported = np.quantile(supported, q)
    time_coordinates = np.unique(np.concatenate(([0.0], supported)))

    survival_functions = model.predict_survival_function(test[features])
    predicted_curves = np.row_stack([
        np.asarray(fn(time_coordinates), dtype=float)
        for fn in survival_functions
    ])
    predicted_curves = np.clip(predicted_curves, 0.0, 1.0)
    predicted_curves[:, 0] = 1.0
    return predicted_curves, time_coordinates

@dataclass
class SmokeTestResult:
    censoring_rate: float
    target_times: np.ndarray
    oracle_bs: np.ndarray
    dependent_bs: Dict[str, np.ndarray]
    oracle_ibs: float
    dependent_ibs: Dict[str, float]

def _choose_censor_scale(
    X: torch.Tensor,
    true_event_time: np.ndarray,
    beta_censor: torch.Tensor,
    censor_uniform: torch.Tensor,
    alpha_c_base: float,
    gamma_c: float,
    target_censoring: float,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Choose a coarse censoring scale that is close to the requested rate."""
    grid = np.geomspace(0.08, 6.0, 50)
    best_mult = None
    best_error = np.inf

    for mult in grid:
        dgp_censor = DGP_Weibull_linear(
            X.shape[1],
            alpha_c_base * float(mult),
            gamma_c,
            use_x=True,
            device=device,
            dtype=dtype,
            coeff=beta_censor,
        )
        censor_time = dgp_censor.rvs(X, censor_uniform)
        censoring_rate = float(np.mean(true_event_time >= censor_time))
        error = abs(censoring_rate - target_censoring)
        if error < best_error:
            best_error = error
            best_mult = float(mult)

    if best_mult is None:
        raise RuntimeError("Failed to choose a censoring scale.")
    return best_mult

def _oracle_brier_scores(
    predicted_curves: np.ndarray,
    time_coordinates: np.ndarray,
    true_event_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Ordinary Brier scores using fully observed latent event times."""
    predicted_survival = predict_multi_probabilities_from_curve(
        np.asarray(predicted_curves),
        np.asarray(time_coordinates),
        np.asarray(target_times),
        "Linear",
    )
    true_survival_status = (
        np.asarray(true_event_times)[:, None] > np.asarray(target_times)[None, :]
    ).astype(float)
    return np.mean(np.square(true_survival_status - predicted_survival), axis=0)

def run_smoke_test(
    n_samples: int = 2000,
    n_features: int = 10,
    seed: int = 0,
    copula_name: str = "clayton",
    tau: float = 0.5,
    target_censoring: float = 0.5,
    train_fraction: float = 0.7,
    num_ibs_points: int = 20,
    methods: Iterable[str] = ("BG", "BG_UW", "CG_Q"),
    verbose: bool = True,
) -> SmokeTestResult:
    """Run one synthetic oracle-vs-dependent Brier-score experiment."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one.")
    if not 0.0 <= target_censoring < 1.0:
        raise ValueError("target_censoring must lie in [0, 1).")
    if not 0.0 <= tau < 1.0:
        raise ValueError("tau must lie in [0, 1).")

    dtype = torch.float64
    device = torch.device("cpu")
    torch.set_default_dtype(dtype)
    np.random.seed(seed)
    torch.manual_seed(seed)

    alpha_c_base = 19.0
    alpha_event = 17.0
    gamma_c = 6.0
    gamma_event = 4.0

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    X = torch.rand(
        (n_samples, n_features),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    beta_event = 2.0 * torch.rand(
        (n_features,), generator=generator, device=device, dtype=dtype
    ) - 1.0
    beta_censor = 2.0 * torch.rand(
        (n_features,), generator=generator, device=device, dtype=dtype
    ) - 1.0

    event_dgp = DGP_Weibull_linear(
        n_features,
        alpha_event,
        gamma_event,
        use_x=True,
        device=device,
        dtype=dtype,
        coeff=beta_event,
    )
    event_uniform = make_event_uniforms(
        seed=seed, n=n_samples, device=device, dtype=dtype
    )
    true_event_time = np.asarray(event_dgp.rvs(X, event_uniform), dtype=float)

    censor_uniform = sample_censor_uniform_given_event_uniform(
        copula_name=copula_name,
        k_tau=tau,
        seed=seed,
        event_v=event_uniform,
        device=device,
        dtype=dtype,
    )
    censor_scale = _choose_censor_scale(
        X=X,
        true_event_time=true_event_time,
        beta_censor=beta_censor,
        censor_uniform=censor_uniform,
        alpha_c_base=alpha_c_base,
        gamma_c=gamma_c,
        target_censoring=target_censoring,
        device=device,
        dtype=dtype,
    )
    censor_dgp = DGP_Weibull_linear(
        n_features,
        alpha_c_base * censor_scale,
        gamma_c,
        use_x=True,
        device=device,
        dtype=dtype,
        coeff=beta_censor,
    )
    observed = make_observed_df(
        X=X,
        true_event_time=true_event_time,
        dgp_cens=censor_dgp,
        u_censor=censor_uniform,
    )
    if len(observed) != n_samples:
        raise AssertionError("Simulation unexpectedly dropped rows.")

    train_idx, test_idx = make_train_test_split_indices(
        n_samples, train_fraction, split_seed=seed
    )
    train = observed.iloc[train_idx].copy()
    test = observed.iloc[test_idx].copy()
    features = [f"X{i}" for i in range(n_features)]

    predicted_curves, time_coordinates = _fit_oracle_cox_and_predict(
        X=X,
        true_event_time=true_event_time,
        train_idx=train_idx,
        test_idx=test_idx,
        features=features,
    )
    predicted_curves = np.asarray(predicted_curves, dtype=float)
    time_coordinates = np.asarray(time_coordinates, dtype=float)

    # Interior quantiles avoid a trivial score at time zero and unstable tail
    # behavior where almost nobody remains at risk.
    true_test_times = test["true_time"].to_numpy(dtype=float)
    target_times = np.quantile(true_test_times, [0.25, 0.50, 0.75])
    oracle_bs = _oracle_brier_scores(
        predicted_curves,
        time_coordinates,
        true_test_times,
        target_times,
    )

    theta = float(kendall_tau_to_theta(copula_name, tau))
    evaluator = DependentEvaluator(
        predicted_curves,
        time_coordinates,
        test["time"].to_numpy(dtype=float),
        test["event"].to_numpy(dtype=int),
        train["time"].to_numpy(dtype=float),
        train["event"].to_numpy(dtype=int),
        copula_name=copula_name,
        alpha=theta,
    )

    dependent_bs: Dict[str, np.ndarray] = {}
    dependent_ibs: Dict[str, float] = {}
    max_observed_time = float(
        max(train["time"].max(), test["time"].max())
    )
    ibs_grid = np.linspace(0.0, max_observed_time, num_ibs_points)
    oracle_ibs_scores = _oracle_brier_scores(
        predicted_curves,
        time_coordinates,
        true_test_times,
        ibs_grid,
    )
    oracle_ibs = float(trapezoid(oracle_ibs_scores, ibs_grid) / max_observed_time)

    for method in methods:
        vector_scores = evaluator.brier_score_multiple_points(method, target_times)
        scalar_scores = np.array(
            [evaluator.brier_score(method, float(t)) for t in target_times]
        )

        # Main API regression test: scalar and vector calls must be identical.
        np.testing.assert_allclose(
            scalar_scores,
            vector_scores,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Single- and multi-time BS disagree for {method}.",
        )
        if not np.all(np.isfinite(vector_scores)):
            raise AssertionError(f"{method} returned a non-finite Brier score.")
        if np.any(vector_scores < 0.0):
            raise AssertionError(f"{method} returned a negative Brier score.")

        # IBS must equal numerical integration of the same BS implementation.
        grid_scores = evaluator.brier_score_multiple_points(method, ibs_grid)
        manual_ibs = float(trapezoid(grid_scores, ibs_grid) / max_observed_time)
        api_ibs = float(
            evaluator.integrated_brier_score(method, num_points=num_ibs_points)
        )
        np.testing.assert_allclose(
            api_ibs,
            manual_ibs,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"IBS integration disagrees with pointwise BS for {method}.",
        )

        dependent_bs[method] = vector_scores
        dependent_ibs[method] = api_ibs

    result = SmokeTestResult(
        censoring_rate=float(1.0 - observed["event"].mean()),
        target_times=target_times,
        oracle_bs=oracle_bs,
        dependent_bs=dependent_bs,
        oracle_ibs=float(oracle_ibs),
        dependent_ibs=dependent_ibs,
    )

    if verbose:
        rows = []
        for j, target_time in enumerate(target_times):
            row = {
                "time": target_time,
                "oracle": oracle_bs[j],
            }
            for method in dependent_bs:
                row[method] = dependent_bs[method][j]
                row[f"abs_err_{method}"] = abs(
                    dependent_bs[method][j] - oracle_bs[j]
                )
            rows.append(row)

        print(
            f"Synthetic dependent-BS smoke test: copula={copula_name}, "
            f"tau={tau:.3f}, theta={theta:.6g}"
        )
        print(
            f"n={n_samples}, train={len(train)}, test={len(test)}, "
            f"censoring={result.censoring_rate:.1%}, "
            f"censor-scale={censor_scale:.4g}"
        )
        print("\nPointwise Brier scores (oracle uses latent uncensored event times):")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

        ibs_row = {"oracle": result.oracle_ibs, **result.dependent_ibs}
        print("\nIntegrated scores:")
        print(pd.DataFrame([ibs_row]).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        print("\nPASS: scalar BS = vector BS, and IBS = integrated pointwise BS.")

    return result

def test_dependent_brier_score_smoke() -> None:
    """Small pytest regression test using the real synthetic/C-G pipeline."""
    result = run_smoke_test(
        n_samples=600,
        n_features=5,
        seed=7,
        copula_name="clayton",
        tau=0.4,
        target_censoring=0.5,
        num_ibs_points=10,
        verbose=False,
    )
    assert 0.15 < result.censoring_rate < 0.85
    assert result.oracle_bs.shape == (3,)
    assert set(result.dependent_bs) == {"BG", "BG_UW", "CG_Q"}

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a synthetic oracle test of the dependent Brier score."
    )
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-features", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--copula", choices=("clayton", "frank"), default="clayton")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--target-censoring", type=float, default=0.5)
    parser.add_argument("--num-ibs-points", type=int, default=20)
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run_smoke_test(
        n_samples=args.n_samples,
        n_features=args.n_features,
        seed=args.seed,
        copula_name=args.copula,
        tau=args.tau,
        target_censoring=args.target_censoring,
        num_ibs_points=args.num_ibs_points,
        verbose=True,
    )
