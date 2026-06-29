from lifelines import CoxPHFitter
import numpy as np
from typing import Callable, Optional
import warnings
from functools import cached_property

import pandas as pd
from scipy.integrate import trapezoid
from estimators import CopulaGraphic, CopulaGraphicWrapper
from utility.metrics import estimate_concordance_index

from utility.metrics import estimate_concordance_index, predict_multi_probabilities_from_curve
from SurvivalEVAL.Evaluations.util import (check_and_convert, KaplanMeierArea, km_mean,
                                           predict_mean_survival_time, predict_median_survival_time,
                                           predict_multi_probs_from_curve)
from SurvivalEVAL.Evaluations.custom_types import NumericArrayLike

from scipy.ndimage import gaussian_filter1d

class DependentEvaluator:
    def __init__(self,
            predicted_survival_curves: NumericArrayLike,
            time_coordinates: NumericArrayLike,
            test_event_times: NumericArrayLike,
            test_event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None,
            copula_name: str = None,
            alpha: float = None,
            predict_time_method: str = "Median",
            interpolation: str = "Linear"
    ):
        self._predicted_curves = check_and_convert(predicted_survival_curves)
        self._time_coordinates = check_and_convert(time_coordinates)

        if self._time_coordinates.ndim == 1:
            if self._time_coordinates[0] != 0:
                warnings.warn("The first time coordinate is not 0. A authentic survival curve should start from 0 "
                              "with 100% survival probability. Adding 0 to the beginning of the time coordinates and"
                              " 1 to the beginning of the predicted curves.")
                # Add 0 to the beginning of the time coordinates, and add the 100% survival probability to the
                # beginning of the predicted curves.
                self._time_coordinates = np.insert(self._time_coordinates, 0, 0)
                self._predicted_curves = np.insert(self._predicted_curves, 0, 1, axis=1)

        test_event_times, test_event_indicators = check_and_convert(test_event_times, test_event_indicators)
        self.event_times = test_event_times
        self.event_indicators = test_event_indicators

        if (train_event_times is not None) and (train_event_indicators is not None):
            train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)
        self.train_event_times = train_event_times
        self.train_event_indicators = train_event_indicators

        if predict_time_method == "Median":
            self.predict_time_method = predict_median_survival_time
        elif predict_time_method == "Mean":
            self.predict_time_method = predict_mean_survival_time
        else:
            error = "Please enter one of 'Median' or 'Mean' for calculating predicted survival time."
            raise TypeError(error)

        self.interpolation = interpolation
        
        self.copula_name = copula_name
        self.alpha = alpha

    def _error_trainset(self, method_name: str):
        if (self.train_event_times is None) or (self.train_event_indicators is None):
            raise TypeError("Train set information is missing. "
                            "Evaluator cannot perform {} evaluation.".format(method_name))

    @property
    def predicted_curves(self):
        return self._predicted_curves

    @predicted_curves.setter
    def predicted_curves(self, val: NumericArrayLike):
        print("Setter called. Resetting predicted curves for this evaluator.")
        self._predicted_curves = check_and_convert(val)
        self._clear_cache()

    @property
    def time_coordinates(self):
        return self._time_coordinates

    @time_coordinates.setter
    def time_coordinates(self, val: NumericArrayLike):
        print("Setter called. Resetting time coordinates for this evaluator.")
        self._time_coordinates = check_and_convert(val)
        self._clear_cache()

    @cached_property
    def predicted_event_times(self):
        return self.predict_time_from_curve(self.predict_time_method)

    def _clear_cache(self):
        self.__dict__.pop('predicted_event_times', None)

    def predict_time_from_curve(self, predict_method: Callable) -> np.ndarray:
        if (predict_method is not predict_mean_survival_time) and (predict_method is not predict_median_survival_time):
            error = "Prediction method must be 'predict_mean_survival_time' or 'predict_median_survival_time', " \
                    "got '{}' instead".format(predict_method.__name__)
            raise TypeError(error)

        predicted_times = []
        for i in range(self.predicted_curves.shape[0]):
            predicted_time = predict_method(self.predicted_curves[i, :], self.time_coordinates, self.interpolation)
            predicted_times.append(predicted_time)
        predicted_times = np.array(predicted_times)
        return predicted_times
        
    def brier_score_multiple_points(
        self,
        method: str,
        target_times: NumericArrayLike,
    ) -> np.ndarray:
        """Compute the dependent Brier score at one or more time points.

        Parameters
        ----------
        method:
            Dependent-censoring correction. Supported values are ``"BG"``
            (copula-graphic margin-time imputation), ``"BG_UW"`` (margin-time
            imputation with uncertainty weighting), and ``"CG_Q"``
            (conditional event-status probabilities).
        target_times:
            One-dimensional array of evaluation times.

        Returns
        -------
        np.ndarray
            Brier score at each requested time, in the same order as
            ``target_times``.
        """
        self._error_trainset("dependent Brier score")

        target_times = np.asarray(check_and_convert(target_times), dtype=float)
        if target_times.ndim == 0:
            target_times = target_times.reshape(1)
        if target_times.ndim != 1:
            raise TypeError("target_times must be a one-dimensional array.")
        if target_times.size == 0:
            raise ValueError("target_times must contain at least one time point.")
        if np.any(~np.isfinite(target_times)) or np.any(target_times < 0):
            raise ValueError("target_times must contain finite, non-negative values.")

        predicted_curves = check_and_convert(self.predicted_curves)
        time_bins = check_and_convert(self.time_coordinates)
        event_times = np.asarray(self.event_times, dtype=float)
        event_indicators = np.asarray(self.event_indicators, dtype=bool)
        train_event_times = np.asarray(self.train_event_times, dtype=float)
        train_event_indicators = np.asarray(self.train_event_indicators, dtype=bool)

        predict_probs_mat = np.array([
            predict_multi_probs_from_curve(
                predicted_curves[i, :],
                time_bins,
                target_times,
            )
            for i in range(predicted_curves.shape[0])
        ])

        n_samples = len(event_times)
        n_times = len(target_times)
        target_times_mat = np.repeat(target_times.reshape(1, -1), n_samples, axis=0)

        if method == "BG":
            censored_mask = ~event_indicators
            cg_model = CopulaGraphicWrapper(
                train_event_times,
                train_event_indicators,
                copula_name=self.copula_name,
                alpha=self.alpha,
            )

            event_times_bg = event_times.copy()
            if np.any(censored_mask):
                event_times_bg[censored_mask] = cg_model.best_guess(
                    event_times[censored_mask]
                )

            event_times_mat = np.repeat(event_times_bg.reshape(-1, 1), n_times, axis=1)
            event_before_or_at = event_times_mat <= target_times_mat
            event_free = ~event_before_or_at

            square_error_mat = (
                np.square(predict_probs_mat) * event_before_or_at
                + np.square(1.0 - predict_probs_mat) * event_free
            )

        elif method == "BG_UW":
            censored_mask = ~event_indicators
            censored_times = event_times[censored_mask]
            cg_model = CopulaGraphicWrapper(
                train_event_times,
                train_event_indicators,
                copula_name=self.copula_name,
                alpha=self.alpha,
            )

            event_times_bg = event_times.copy()
            if censored_times.size > 0:
                event_times_bg[censored_mask] = cg_model.best_guess(censored_times)

            event_times_mat = np.repeat(event_times_bg.reshape(-1, 1), n_times, axis=1)
            event_before_or_at = event_times_mat <= target_times_mat
            event_free = ~event_before_or_at

            w_row = np.ones(n_samples, dtype=float)
            if censored_times.size > 0:
                cg_model_uncert = CopulaGraphic(
                    train_event_times,
                    train_event_indicators,
                    alpha=self.alpha,
                    type=self.copula_name,
                )
                event_cdf_at_censoring = 1.0 - cg_model_uncert.predict(censored_times)
                w_row[censored_mask] = np.clip(event_cdf_at_censoring, 0.0, 1.0)

            # Preserve the original normalization: the average observation
            # weight is one, so uncertainty weighting does not change scale.
            w_row /= w_row.mean() + 1e-12
            weight_mat = np.repeat(w_row.reshape(-1, 1), n_times, axis=1)

            square_error_mat = weight_mat * (
                np.square(predict_probs_mat) * event_before_or_at
                + np.square(1.0 - predict_probs_mat) * event_free
            )

        elif method == "CG_Q":
            cg_model = CopulaGraphicWrapper(
                train_event_times,
                train_event_indicators,
                copula_name=self.copula_name,
                alpha=self.alpha,
            )

            event_times_mat = np.repeat(event_times.reshape(-1, 1), n_times, axis=1)
            observed_mask = event_indicators
            censored_mask = ~observed_mask
            q_mat = np.zeros((n_samples, n_times), dtype=float)

            # For observed events, q_i(t) is the known survival status I(E_i > t).
            q_mat[observed_mask, :] = (
                event_times_mat[observed_mask, :] > target_times_mat[observed_mask, :]
            ).astype(float)

            # For censored observations, replace the unknown status with
            # P(E_i > t | E_i > C_i) under the fitted copula-graphic model.
            if np.any(censored_mask):
                q_mat[censored_mask, :] = cg_model.conditional_survival_after_censoring(
                    event_times[censored_mask],
                    target_times,
                )

            square_error_mat = (
                q_mat * np.square(1.0 - predict_probs_mat)
                + (1.0 - q_mat) * np.square(predict_probs_mat)
            )

        else:
            raise ValueError(
                f"Unknown dependent Brier-score method {method!r}. "
                "Expected one of {'BG', 'BG_UW', 'CG_Q'}."
            )

        brier_scores = np.mean(square_error_mat, axis=0)
        if np.isnan(brier_scores).any():
            warnings.warn("Time-dependent Brier Score contains nan")

        return brier_scores

    def brier_score(self, method: str, target_time: float) -> float:
        """Compute the dependent Brier score at a single time point."""
        target_time = np.asarray(target_time)
        if target_time.ndim != 0:
            raise TypeError("target_time must be a scalar.")

        return float(
            self.brier_score_multiple_points(
                method=method,
                target_times=np.array([float(target_time)]),
            )[0]
        )

    def integrated_brier_score(self, method: str, num_points: int):
        """Compute IBS by integrating the dependent Brier score over time."""
        self._error_trainset("integrated dependent Brier score")
        if not isinstance(num_points, (int, np.integer)) or num_points < 2:
            raise ValueError("num_points must be an integer greater than or equal to 2.")

        max_target_time = np.max(
            np.concatenate((self.event_times, self.train_event_times))
        )
        if max_target_time <= 0:
            raise ValueError("The maximum observed time must be positive to compute IBS.")

        time_points = np.linspace(0.0, max_target_time, num_points)
        brier_scores = self.brier_score_multiple_points(method, time_points)
        return float(trapezoid(brier_scores, time_points) / max_target_time)

class IndependentEvaluator:
    """
    Integrated Brier score under conditionally independent censoring.

    This evaluator uses a Cox proportional hazards model for the censoring
    distribution G(t | X) = P(C > t | X), rather than a marginal Kaplan-Meier
    censoring curve. It is therefore the conditional/independent-censoring
    analogue of the usual KM-IPCW IBS.

    Parameters
    ----------
    predicted_survival_curves:
        Predicted event survival curves S_hat(t | x) for the test set, with
        shape (n_test, n_time_points). A pandas DataFrame is accepted.
    time_coordinates:
        Time grid corresponding to the columns of predicted_survival_curves.
    test_event_times, test_event_indicators:
        Observed test times and event indicators, where 1 means event and
        0 means censored.
    train_event_times, train_event_indicators:
        Observed train times and event indicators, where 1 means event and
        0 means censored.
    train_features, test_features:
        Covariate matrices used for the censoring CoxPH model. These should be
        the same covariates available to the event model after preprocessing.
    """

    def __init__(
        self,
        predicted_survival_curves: NumericArrayLike,
        time_coordinates: NumericArrayLike,
        test_event_times: NumericArrayLike,
        test_event_indicators: NumericArrayLike,
        train_event_times: NumericArrayLike,
        train_event_indicators: NumericArrayLike,
        train_features,
        test_features,
        predict_time_method: str = "Median",
        interpolation: str = "Linear",
        censor_penalizer: float = 0.01,
    ):
        self._predicted_curves = check_and_convert(predicted_survival_curves)
        self._time_coordinates = check_and_convert(time_coordinates)

        if self._time_coordinates.ndim == 1 and self._time_coordinates[0] != 0:
            warnings.warn(
                "The first time coordinate is not 0. Adding 0 to the time grid "
                "and 1 to the beginning of the predicted survival curves."
            )
            self._time_coordinates = np.insert(self._time_coordinates, 0, 0)
            self._predicted_curves = np.insert(self._predicted_curves, 0, 1, axis=1)

        test_event_times, test_event_indicators = check_and_convert(
            test_event_times, test_event_indicators
        )
        train_event_times, train_event_indicators = check_and_convert(
            train_event_times, train_event_indicators
        )

        self.event_times = test_event_times.astype(float)
        self.event_indicators = test_event_indicators.astype(bool)
        self.train_event_times = train_event_times.astype(float)
        self.train_event_indicators = train_event_indicators.astype(bool)

        self.train_features = self._as_feature_frame(train_features, prefix="X")
        self.test_features = self._as_feature_frame(test_features, prefix="X")

        if self.train_features.shape[0] != self.train_event_times.shape[0]:
            raise ValueError("train_features must have one row per training observation.")
        if self.test_features.shape[0] != self.event_times.shape[0]:
            raise ValueError("test_features must have one row per test observation.")

        if predict_time_method == "Median":
            self.predict_time_method = predict_median_survival_time
        elif predict_time_method == "Mean":
            self.predict_time_method = predict_mean_survival_time
        else:
            raise TypeError("Please enter one of 'Median' or 'Mean'.")

        self.interpolation = interpolation
        self.censor_penalizer = float(censor_penalizer)
        self._censor_model = None
        self._constant_censor_survival = None
        self._km_censor_times = None
        self._km_censor_survival = None
        self._censor_fit_status = None
        self._cox_fit_error = None

        # Lifelines CoxPH is very sensitive to non-finite values, constant
        # columns and extreme scaling. Clean the train/test frames once so the
        # censoring model sees a stable design matrix with identical columns in
        # train and test.
        self._sanitize_feature_frames()
        self._fit_censoring_model()

    @staticmethod
    def _as_feature_frame(features, prefix: str) -> pd.DataFrame:
        if isinstance(features, pd.DataFrame):
            X = features.copy()
        else:
            X = pd.DataFrame(np.asarray(features))
        X = X.reset_index(drop=True)
        X.columns = [str(c) if str(c) not in ["__time", "__censor_event"] else f"{prefix}{j}"
                     for j, c in enumerate(X.columns)]
        return X.astype(float)

    def _sanitize_feature_frames(self):
        """Make train/test covariates safe for lifelines CoxPH.

        The conditional IPCW censoring model is a nuisance model. It should not
        make the whole experiment fail because of a constant column, an inf, or
        a badly scaled feature. We therefore apply deterministic preprocessing:
        replace inf with nan, impute by train medians, drop constant columns,
        and standardize using train statistics.
        """
        X_train = self.train_features.replace([np.inf, -np.inf], np.nan).copy()
        X_test = self.test_features.replace([np.inf, -np.inf], np.nan).copy()

        med = X_train.median(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_train = X_train.fillna(med)
        X_test = X_test.fillna(med)

        # Drop zero/near-zero variance columns using train statistics only.
        std = X_train.std(axis=0, ddof=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        keep_cols = std[std > 1e-12].index.tolist()

        if len(keep_cols) == 0:
            # No usable covariates; use marginal censoring KM below.
            self.train_features = X_train.iloc[:, :0].copy()
            self.test_features = X_test.iloc[:, :0].copy()
            return

        X_train = X_train[keep_cols]
        X_test = X_test[keep_cols]

        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0, ddof=0).replace(0.0, 1.0)

        self.train_features = ((X_train - mu) / sd).astype(float)
        self.test_features = ((X_test - mu) / sd).astype(float)

    @staticmethod
    def _fit_marginal_km(times: np.ndarray, event_indicators: np.ndarray):
        """Kaplan-Meier survival for the censoring distribution.

        Here event_indicators=1 means observed censoring. Returns step-function
        support times and survival values G(t)=P(C>t).
        """
        times = np.asarray(times, dtype=float)
        event_indicators = np.asarray(event_indicators, dtype=bool)
        mask = np.isfinite(times) & (times >= 0)
        times = times[mask]
        event_indicators = event_indicators[mask]

        event_times = np.sort(np.unique(times[event_indicators]))
        if event_times.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)

        surv_vals = []
        surv = 1.0
        for t in event_times:
            at_risk = np.sum(times >= t)
            n_events = np.sum((times == t) & event_indicators)
            if at_risk > 0:
                surv *= max(0.0, 1.0 - n_events / at_risk)
            surv_vals.append(surv)

        return event_times.astype(float), np.asarray(surv_vals, dtype=float)

    @staticmethod
    def _km_predict(times: np.ndarray, km_times: np.ndarray, km_survival: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        if km_times is None or km_survival is None or len(km_times) == 0:
            return np.ones_like(times, dtype=float)
        idx = np.searchsorted(km_times, times, side="right") - 1
        out = np.ones_like(times, dtype=float)
        valid = idx >= 0
        out[valid] = km_survival[idx[valid]]
        return out

    @property
    def predicted_curves(self):
        return self._predicted_curves

    @property
    def time_coordinates(self):
        return self._time_coordinates

    @cached_property
    def predicted_event_times(self):
        return self.predict_time_from_curve(self.predict_time_method)

    def predict_time_from_curve(self, predict_method: Callable) -> np.ndarray:
        if (predict_method is not predict_mean_survival_time) and (predict_method is not predict_median_survival_time):
            raise TypeError(
                "Prediction method must be 'predict_mean_survival_time' or "
                "'predict_median_survival_time'."
            )
        return np.array([
            predict_method(self.predicted_curves[i, :], self.time_coordinates, self.interpolation)
            for i in range(self.predicted_curves.shape[0])
        ])

    def predict_multi_probabilities_from_curve(self, target_times: np.ndarray) -> np.ndarray:
        predict_probs_mat = []
        for i in range(self.predicted_curves.shape[0]):
            predict_probs = predict_multi_probs_from_curve(
                self.predicted_curves[i, :],
                self.time_coordinates,
                target_times,
                self.interpolation,
            ).tolist()
            predict_probs_mat.append(predict_probs)
        return np.array(predict_probs_mat)

    def _fit_censoring_model(self):
        if CoxPHFitter is None:
            raise ImportError(
                "IndependentEvaluator requires lifelines. Install it with `pip install lifelines` "
                "or add it to your conda environment."
            )

        # For censoring model, censoring is the event of interest.
        censor_events = (~self.train_event_indicators).astype(int)

        if np.sum(censor_events) == 0:
            # No observed censoring in train: G(t | X) = 1 on the observed support.
            self._constant_censor_survival = 1.0
            self._censor_fit_status = "constant_no_censoring"
            return

        # If all covariates were dropped, use marginal KM rather than failing.
        if self.train_features.shape[1] == 0:
            self._km_censor_times, self._km_censor_survival = self._fit_marginal_km(
                self.train_event_times,
                censor_events,
            )
            self._censor_fit_status = "km_fallback_no_covariates"
            warnings.warn(
                "CoxPH censoring model has no usable covariates after preprocessing; "
                "falling back to marginal KM censoring survival for IPCW (CoxPH)."
            )
            return

        df_censor = self.train_features.copy()
        df_censor["__time"] = self.train_event_times
        df_censor["__censor_event"] = censor_events
        df_censor = df_censor.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

        if df_censor["__censor_event"].sum() == 0:
            self._constant_censor_survival = 1.0
            self._censor_fit_status = "constant_no_censoring_after_cleaning"
            return

        # Lifelines can be sensitive to separation/collinearity. Retry with
        # stronger ridge penalization and smaller Newton steps before falling
        # back to marginal KM. The fallback keeps the semisynthetic sweep from
        # crashing because CoxPH is only a nuisance model for IPCW.
        last_error = None
        penalizers = [self.censor_penalizer, 0.1, 1.0, 10.0, 100.0]
        fit_options_list = [
            None,
            {"step_size": 0.5},
            {"step_size": 0.25},
            {"step_size": 0.1},
        ]

        for penalizer in penalizers:
            for fit_options in fit_options_list:
                try:
                    model = CoxPHFitter(penalizer=float(penalizer))
                    fit_kwargs = dict(
                        duration_col="__time",
                        event_col="__censor_event",
                        show_progress=False,
                    )
                    if fit_options is not None:
                        fit_kwargs["fit_options"] = fit_options

                    model.fit(df_censor, **fit_kwargs)
                    self._censor_model = model
                    self.censor_penalizer = float(penalizer)
                    self._censor_fit_status = "coxph"
                    return
                except Exception as exc:  # lifelines raises several convergence-related errors
                    last_error = exc

        # Robust fallback: marginal KM censoring survival. This is less targeted
        # than conditional IPCW, but finite and preferable to terminating the
        # whole run. Keep the original error for diagnostics.
        self._cox_fit_error = last_error
        self._km_censor_times, self._km_censor_survival = self._fit_marginal_km(
            self.train_event_times,
            censor_events,
        )
        self._censor_fit_status = "km_fallback_coxph_failed"
        warnings.warn(
            "CoxPH censoring model failed to fit even after preprocessing, stronger "
            f"penalization and smaller Newton steps. Falling back to marginal KM "
            f"censoring survival for IPCW (CoxPH). Last error: {last_error}"
        )
        return

    def _predict_censor_survival(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        """Return G_hat(times_i | X_i) for matched rows/times."""
        times = np.asarray(times, dtype=float)
        X = X.reset_index(drop=True)

        if X.shape[0] != times.shape[0]:
            raise ValueError("X and times must have the same number of rows/elements.")

        if self._constant_censor_survival is not None:
            return np.full(X.shape[0], float(self._constant_censor_survival), dtype=float)

        if self._km_censor_times is not None:
            return self._km_predict(times, self._km_censor_times, self._km_censor_survival)

        # Use unique times to avoid relying on lifelines preserving duplicate
        # requested times. The result has shape (n_unique_times, n_rows).
        unique_times, inverse = np.unique(times, return_inverse=True)
        surv = self._censor_model.predict_survival_function(X, times=unique_times)
        values = np.asarray(surv.values, dtype=float)
        if values.shape[0] != unique_times.shape[0]:
            values = values.T

        out = values[inverse, np.arange(X.shape[0])]
        out = np.asarray(out, dtype=float)
        out[~np.isfinite(out)] = np.nan
        return out

    def _predict_censor_survival_matrix(self, target_times: np.ndarray) -> np.ndarray:
        """Return matrix G_hat(t_j | X_i), shape (n_test, n_times)."""
        target_times = np.asarray(target_times, dtype=float)

        if self._constant_censor_survival is not None:
            return np.full((self.test_features.shape[0], target_times.shape[0]),
                           float(self._constant_censor_survival), dtype=float)

        if self._km_censor_times is not None:
            g = self._km_predict(target_times, self._km_censor_times, self._km_censor_survival)
            return np.repeat(g.reshape(1, -1), repeats=self.test_features.shape[0], axis=0)

        surv = self._censor_model.predict_survival_function(self.test_features, times=target_times)
        values = np.asarray(surv.values, dtype=float)
        if values.shape[0] == target_times.shape[0]:
            return values.T
        return values

    def brier_score_multiple_points(self, target_times: np.ndarray) -> np.ndarray:
        if target_times.ndim != 1:
            raise TypeError("target_times must be a one-dimensional array.")

        predict_probs_mat = self.predict_multi_probabilities_from_curve(target_times)

        target_times_mat = np.repeat(target_times.reshape(1, -1), repeats=len(self.event_times), axis=0)
        event_times_mat = np.repeat(self.event_times.reshape(-1, 1), repeats=len(target_times), axis=1)
        event_indicators_mat = np.repeat(self.event_indicators.reshape(-1, 1), repeats=len(target_times), axis=1)

        # G_hat(T_i | X_i), used for observed events before/equal t.
        G_at_event = self._predict_censor_survival(self.test_features, self.event_times)
        G_at_event_mat = np.repeat(G_at_event.reshape(-1, 1), repeats=len(target_times), axis=1)

        # G_hat(t | X_i), used for individuals known to be event-free at t.
        G_at_target_mat = self._predict_censor_survival_matrix(target_times)

        eps = 1e-3  # or 1e-2 for stronger truncation
        G_at_event_mat = np.clip(G_at_event_mat, eps, 1.0)
        G_at_target_mat = np.clip(G_at_target_mat, eps, 1.0)
        
        weight_cat1 = ((event_times_mat <= target_times_mat) & event_indicators_mat) / G_at_event_mat
        weight_cat2 = (event_times_mat > target_times_mat) / G_at_target_mat

        weight_cat1[~np.isfinite(weight_cat1)] = 0.0
        weight_cat2[~np.isfinite(weight_cat2)] = 0.0

        square_error_mat = (
            np.square(predict_probs_mat) * weight_cat1
            + np.square(1.0 - predict_probs_mat) * weight_cat2
        )
        return np.mean(square_error_mat, axis=0)

    def integrated_brier_score(self, num_points: int = None, draw_figure: bool = False) -> float:
        max_target_time = np.max(np.concatenate((self.event_times, self.train_event_times)))

        if num_points is None:
            censored_times = self.event_times[~self.event_indicators]
            time_points = np.unique(censored_times)
            if time_points.size == 0:
                raise ValueError(
                    "No censored observations in the test set; provide num_points for calculating IBS."
                )
            time_range = np.max(time_points) - np.min(time_points)
        else:
            time_points = np.linspace(0, max_target_time, num_points)
            time_range = max_target_time

        brier_scores = self.brier_score_multiple_points(time_points)
        if np.isnan(brier_scores).any():
            warnings.warn("Time-dependent Brier Score contains nan")

        integral_value = trapezoid(brier_scores, time_points)
        ibs_score = integral_value / time_range

        if draw_figure:
            import matplotlib.pyplot as plt
            plt.plot(time_points, brier_scores, 'bo-')
            plt.xlabel('Time')
            plt.ylabel('Brier Score')
            plt.show()

        return float(ibs_score)