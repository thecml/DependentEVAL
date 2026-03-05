import numpy as np
from typing import Callable, Optional
import warnings
from functools import cached_property

from scipy.integrate import trapezoid
from estimators import CopulaGraphic, CopulaGraphicWrapper

from SurvivalEVAL.Evaluations.util import (check_and_convert, predict_mean_survival_time,
                                           predict_median_survival_time, predict_multi_probs_from_curve)
from SurvivalEVAL.Evaluations.custom_types import NumericArrayLike

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
        
    def integrated_brier_score(
        self,
        num_points: int,
        uncertainty_weighting: bool = True,
    ) -> float:
        """
        Calculate the Dependent Integrated Brier score (IBS-Dep) from the predicted survival curve.

        Parameters
        ----------
        num_points:
            Number of evaluation time points on [0, max_target_time].
        uncertainty_weighting:
            If True, apply uncertainty weights to censored observations.
        """
        if (self.train_event_times is None) or (self.train_event_indicators is None):
            raise TypeError(
                "Train set information is missing. Evaluator cannot perform integrated Brier score evaluation."
            )

        predicted_curves = check_and_convert(self.predicted_curves)
        time_bins = check_and_convert(self.time_coordinates)

        event_times = self.event_times
        event_indicators = self.event_indicators.astype(bool)

        train_event_times = self.train_event_times
        train_event_indicators = self.train_event_indicators.astype(bool)

        max_target_time = np.max(np.concatenate((event_times, train_event_times)))
        time_points = np.linspace(0, max_target_time, num_points)
        time_range = max_target_time if max_target_time > 0 else 1.0

        # Predicted event probabilities at evaluation time points
        predict_probs_mat = []
        for i in range(predicted_curves.shape[0]):
            predict_probs = predict_multi_probs_from_curve(
                predicted_curves[i, :],
                time_bins,
                time_points,
            ).tolist()
            predict_probs_mat.append(predict_probs)
        predict_probs_mat = np.asarray(predict_probs_mat)

        # Best-guess imputation for censored times
        censored_mask = ~event_indicators
        censored_times = event_times[censored_mask]

        cg_wrapper = CopulaGraphicWrapper(
            train_event_times,
            train_event_indicators,
            copula_name=self.copula_name,
            alpha=self.alpha,
        )

        event_times_bg = event_times.copy()
        if censored_times.size > 0:
            event_times_bg[censored_mask] = cg_wrapper.best_guess(censored_times)

        target_times_mat = np.repeat(time_points.reshape(1, -1), repeats=len(event_times), axis=0)
        event_times_mat = np.repeat(event_times_bg.reshape(-1, 1), repeats=len(time_points), axis=1)

        weight_cat1 = (event_times_mat <= target_times_mat)
        weight_cat2 = (event_times_mat > target_times_mat)

        if uncertainty_weighting:
            if censored_times.size > 0:
                cg_uncert = CopulaGraphic(
                    train_event_times,
                    train_event_indicators,
                    alpha=self.alpha,
                    type=self.copula_name,
                )
                S_e = cg_uncert.predict(censored_times)
                F_c = 1.0 - S_e
                w_c = np.clip(F_c, 0.0, 1.0)
            else:
                w_c = np.array([])

            w_row = np.ones(len(event_times), dtype=float)
            if w_c.size > 0:
                w_row[censored_mask] = w_c

            w_row /= (w_row.mean() + 1e-12)
            w_mat = np.repeat(w_row.reshape(-1, 1), repeats=len(time_points), axis=1)

            weight_cat1 = weight_cat1 * w_mat
            weight_cat2 = weight_cat2 * w_mat

        square_error_mat = (
            np.square(predict_probs_mat) * weight_cat1
            + np.square(1.0 - predict_probs_mat) * weight_cat2
        )
        brier_scores = np.mean(square_error_mat, axis=0)

        if np.isnan(brier_scores).any():
            warnings.warn("Time-dependent Brier Score contains nan")
            bs_dict = {tp: bs for tp, bs in zip(time_points, brier_scores)}
            print(f"Brier scores for multiple time points are {bs_dict}")

        ibs_score = trapezoid(brier_scores, time_points) / time_range
        return float(ibs_score)