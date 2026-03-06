# DependentEVAL

Code for "Overcoming Dependent Censoring in the Evaluation of Survival Models (2025)"

Preprint: https://arxiv.org/abs/2502.19460 **(Under review)**

## Usage

`DependentEvaluator` computes a dependent-censoring-aware Integrated Brier Score (IBS).  

```python
from src.evaluator import DependentEvaluator  # adjust import path if needed

# survival_outputs: array of predicted survival curves, shape (n_test, n_time_bins)
# time_coordinates: array of time coordinates for the survival curves, shape (n_time_bins,)
# test_event_times / test_event_indicators: test event times and indicators
# train_event_times / train_event_indicators: train event times and indicators
# copula_name: copula family name (eg, "clayton")
# alpha: dependence parameter (eg, 2.0)
dep_evaluator = DependentEvaluator(
    predicted_survival_curves=survival_outputs,
    time_coordinates=time_bins,
    test_event_times=test_times,
    test_event_indicators=test_events,
    train_event_times=train_times,
    train_event_indicators=train_events,
    copula_name=copula_name,
    alpha=alpha,
)

# Dependent IBS (BG, no uncertainty weighting)
ibs_dep_bg = dep_evaluator.integrated_brier_score(num_points=10, uncertainty_weighting=False)
print("Dependent IBS (BG):", ibs_dep_bg)

# Dependent IBS (BG with uncertainty weighting; default)
ibs_dep_bg_uw = dep_evaluator.integrated_brier_score(num_points=10, uncertainty_weighting=True)
print("Dependent IBS (BG+UW):", ibs_dep_bg_uw)
```
num_points controls the number of evaluation time points used for numerical integration. <br>
uncertainty_weighting=True applies additional down-/up-weighting of imputed censored observations.

Citation
--------
If you find this paper useful in your work, please consider citing it:
 
```
@article{lillelund_overcoming_2025,
  title={Overcoming Dependent Censoring in the Evaluation of Survival Models}, 
  author={Christian Marius Lillelund and Shi-ang Qi and Russell Greiner},
  journal={preprint, arXiv:2502.19460},
  year={2025},
}
```
