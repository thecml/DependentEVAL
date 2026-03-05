import numpy as np
from dataclasses import InitVar, dataclass, field

"""
Courtesy of https://github.com/shi-ang/SurvivalEVAL/blob/main/SurvivalEVAL/NonparametricEstimator/SingleEvent.py#L281
"""
@dataclass
class CopulaGraphic:
    """
    Implementation of the Copula Graphic estimator for survival function, under the dependent censoring assumption.

    This implementation supports three types of copulas:
    - Clayton
    - Gumbel
    - Frank
    The estimator is based on the R code in the package 'compound.Cox' by Takeshi Emura.
    To see the original R code, visit:
    https://github.com/cran/compound.Cox/tree/master/R/CG.Clayton.R
    https://github.com/cran/compound.Cox/tree/master/R/CG.Gumbel.R
    https://github.com/cran/compound.Cox/tree/master/R/CG.Frank.R

    However, the original R code (and also the math derivation) cannot handle ties --
    e.g., when a censored instance and an event instance have the same time point.
    This implementation correctly handling ties.
    based on the derivation in paper: https://arxiv.org/abs/2502.19460
    """
    event_times: InitVar[np.array]
    event_indicators: InitVar[np.array]
    alpha: InitVar[float]
    type: InitVar[str] = "clayton"
    n_samples: int = field(init=False)
    survival_times: np.array = field(init=False)
    population_count: np.array = field(init=False)
    events: np.array = field(init=False)
    survival_probabilities: np.array = field(init=False)

    def __post_init__(self, event_times, event_indicators, alpha, type):
        alpha = max(alpha, 1e-9)
        self.n_samples = len(event_times)
        index = np.lexsort((event_indicators, event_times))
        unique_times = np.unique(event_times[index], return_counts=True)
        self.survival_times = unique_times[0]
        self.population_count = np.flip(np.flip(unique_times[1]).cumsum())

        event_counter = np.append(0, unique_times[1].cumsum()[:-1])
        event_ind = list()
        for i in range(np.size(event_counter[:-1])):
            event_ind.append(event_counter[i])
            event_ind.append(event_counter[i + 1])
        event_ind.append(event_counter[-1])
        event_ind.append(len(event_indicators))
        self.events = np.add.reduceat(np.append(event_indicators[index], 0), event_ind)[::2]

        event_diff = self.population_count - self.events

        with np.errstate(divide="ignore"):
            # ignore division by zero warnings,
            # such warnings are expected when the last time point has an event so the event_diff is 0.
            # but we will set the last point to 0 anyway.
            if type == "clayton":
                diff_ = (event_diff / self.n_samples) ** (- alpha) - (self.population_count / self.n_samples) ** (- alpha)
                diff_[-1] = 0
                self.survival_probabilities = (1.0 + np.cumsum(diff_)) ** ( - 1.0 / alpha)
            elif type == "gumbel":
                diff_ = ((- np.log(event_diff / self.n_samples)) ** (alpha + 1) -
                         (-np.log(self.population_count / self.n_samples)) ** (alpha + 1))
                diff_[-1] = 0
                self.survival_probabilities = np.exp(  -np.cumsum(diff_) ** (1 / (1 + alpha))  )
            elif type == "frank":
                log_diff_ = np.log(  (np.exp(-alpha * event_diff / self.n_samples) - 1) / (np.exp(-alpha * self.population_count / self.n_samples) - 1)  )
                log_diff_[-1] = 0
                self.survival_probabilities = -1 / alpha * np.log(  1 + (np.exp(-alpha) - 1) * np.exp(np.cumsum(log_diff_))  )
            else:
                raise ValueError(f"Unknown copula type: {type}. Supported types are 'clayton', 'gumbel', and 'frank'.")

        self.cumulative_dens = 1 - self.survival_probabilities
        self.probability_dens = np.diff(np.append(self.cumulative_dens, 1))

    def predict(self, prediction_times: np.array) -> np.array:
        """
        Predict the survival probabilities at the given prediction times.
        Parameters
        ----------
        prediction_times: np.array
            The times at which to predict the survival probabilities.
        Returns
        -------
        np.array
            The predicted survival probabilities at the given times.
        """
        probability_index = np.digitize(prediction_times, self.survival_times)
        probability_index = np.where(
            probability_index == self.survival_times.size + 1,
            probability_index - 1,
            probability_index,
        )
        probabilities = np.append(1, self.survival_probabilities)[probability_index]

        return probabilities

    def predict_median(self) -> float:
        """
        Predict the median survival time based on the survival probabilities.
        It is calculated as the first time point where the survival probability is less than or equal to 0.5.
        No interpolation is performed, as we are strictly following the implementation of the CG estimator in R.
        Returns
        -------
        float
            The predicted median survival time.
        """
        median_index = np.where(self.survival_probabilities <= 0.5)[0]
        if median_index.size == 0:
            return np.inf
        return self.survival_times[median_index[0]]

class CopulaGraphicWrapper():
    """
    Wrapper class of the CopulaGraphic estimator that implements the best_guess() function.
    """
    def __init__(self, event_times, event_indicators,
                 copula_name="clayton", alpha=0) -> None:
        self.cg_estimator = CopulaGraphic(
            event_times, event_indicators, alpha=alpha, type=copula_name
        )
        
        index = np.lexsort((event_indicators, event_times))
        unique_times = np.unique(event_times[index], return_counts=True)
        self.survival_times = unique_times[0]
        
        self.survival_probabilities = self.cg_estimator.predict(self.survival_times)
        self.survival_probabilities[-1] = 0
        
        area_probabilities = np.append(1, self.survival_probabilities)
        area_times = np.append(0, self.survival_times)
        self.cg_linear_zero = -1 / ((area_probabilities[-1] - 1) / area_times[-1])
        if self.survival_probabilities[-1] != 0:
            area_times = np.append(area_times, self.cg_linear_zero)
            area_probabilities = np.append(area_probabilities, 0)

        area_diff = np.diff(area_times, 1)
        average_probabilities = (area_probabilities[0:-1] + area_probabilities[1:]) / 2
        area = np.flip(np.flip(area_diff * average_probabilities).cumsum())

        self.area_times = np.append(area_times, np.inf).astype(float)
        self.area_probabilities = np.asarray(area_probabilities, dtype=float)
        self.area = np.append(area, 0).astype(float)

    def predict(self, prediction_times: np.array):
        """Predict survival probabilities at given times using CG estimator."""
        return self.cg_estimator.predict(prediction_times)
    
    def best_guess(self, censor_times: np.array):
        """Compute best guess (expected time to event) for censored subjects."""
        slope = (1 - min(self.survival_probabilities)) / (0 - max(self.survival_times))
        
        before_last_idx = censor_times <= max(self.survival_times)
        after_last_idx = censor_times > max(self.survival_times)
        surv_prob = np.empty_like(censor_times).astype(float)
        surv_prob[after_last_idx] = 1 + censor_times[after_last_idx] * slope
        surv_prob[before_last_idx] = self.predict(censor_times[before_last_idx])

        surv_prob = np.clip(surv_prob, a_min=1e-10, a_max=None)

        censor_indexes = np.digitize(censor_times, self.area_times)
        censor_indexes = np.where(
            censor_indexes == self.area_times.size + 1,
            censor_indexes - 1,
            censor_indexes,
        )

        beyond_idx = censor_indexes > len(self.area_times) - 2
        censor_area = np.zeros_like(censor_times).astype(float)

        censor_area[~beyond_idx] = ((self.area_times[censor_indexes[~beyond_idx]] - censor_times[~beyond_idx]) *
                                    (self.area_probabilities[censor_indexes[~beyond_idx]] + surv_prob[~beyond_idx])
                                    * 0.5)
        censor_area[~beyond_idx] += self.area[censor_indexes[~beyond_idx]]
        return censor_times + censor_area / surv_prob
        
        