# DependentEVAL

Code for "Overcoming Dependent Censoring in the Evaluation of Survival Models (2026)"

Preprint: https://arxiv.org/abs/2502.19460 **Accepted to UAI 2026**

Standard survival evaluation methods commonly assume that the event time and censoring time are conditionally independent. When this assumption is violated, commonly used estimators and metrics can give biased estimates of predictive performance. This repository provides tools for modeling the dependence explicitly using copulas and for evaluating survival predictions with dependence-aware metrics.

## What is dependent censoring?

Let

- \(E\) denote the event time,
- \(C\) denote the censoring time, and
- \(X\) denote the observed covariates.

For each individual, we observe only

\[
T = \min(E,C),
\qquad
\Delta = \mathbb{1}[E \leq C].
\]

A common assumption in survival analysis is **conditional independent censoring**:

\[
E \perp C \mid X.
\]

Censoring is dependent when this condition does not hold:

\[
E \not\perp C \mid X.
\]

This may happen when an unobserved variable affects both the event and censoring processes. For example, an unobserved tumor grade may influence both cancer relapse and study dropout.

<table>
  <tr>
    <td align="center" width="30%">
      <img src="plots/dependence1.png" width="100%" alt="Unobserved confounding between event and censoring times">
    </td>
    <td align="center" width="36%">
      <img src="plots/dependence2.png" width="100%" alt="Effect of dependent censoring on survival estimation">
    </td>
    <td align="center" width="34%">
      <img src="plots/dependence3.png" width="100%" alt="Event and censoring times generated from a Clayton copula">
    </td>
  </tr>
  <tr>
    <td valign="top">
      <b>(a) Unobserved confounding.</b><br>
      An unobserved covariate induces residual dependence between the event and censoring times.
    </td>
    <td valign="top">
      <b>(b) Biased survival estimation.</b><br>
      A patient is censored shortly before an event for reasons related to the event risk. The true and estimated survival curves agree until censoring but diverge afterward.
    </td>
    <td valign="top">
      <b>(c) Copula dependence.</b><br>
      Event and censoring times generated using a Clayton copula. Larger Kendall's \(\tau\) corresponds to stronger dependence.
    </td>
  </tr>
</table>

Under dependent censoring, censored individuals are not necessarily representative of those who remain under observation. Kaplan--Meier and inverse-probability-of-censoring weighted estimators may therefore assign inappropriate survival probabilities or censoring weights.

As a result, an evaluation metric may estimate a distorted prediction error rather than the error that would have been observed if all event times were available.

> **Note:** GitHub does not reliably render PDF files inline. Export the three introductory figures as
> `plots/dependence1.png`, `plots/dependence2.png`, and `plots/dependence3.png`.
> The publication-quality PDF versions can remain in the same directory.

---

## Can dependent censoring be identified from the data?

Not without additional assumptions.

For each individual, only one of \(E\) and \(C\) is observed. Their joint distribution, and therefore their dependence structure, is generally not identifiable from ordinary right-censored data alone.

In practice, dependent censoring can be investigated through:

1. **Domain knowledge:** whether dropout or loss to follow-up may be related to disease severity or event risk.
2. **Observed predictors of censoring:** whether measured variables are associated with both censoring and the outcome.
3. **Sensitivity analysis:** whether conclusions change across plausible copula families and dependence strengths.
4. **Joint modeling assumptions:** whether a specified copula model provides a plausible description of the latent event and censoring times.

The estimated copula parameter should therefore be interpreted as model-dependent rather than as assumption-free evidence of dependent censoring.

---

## Fitting a copula model

The repository includes an experiment for fitting and comparing candidate copula models:

```bash
python src/experiments/find_copula.py --help
```

See [`src/experiments/find_copula.py`](src/experiments/find_copula.py).

The script can be used to:

- fit supported copula families,
- estimate their dependence parameters,
- compare candidate dependence models,
- report the implied Kendall's \(\tau\), and
- perform sensitivity analysis under alternative assumptions.

The joint survival distribution is modeled as

\[
S_{E,C}(e,c \mid X)
=
C_\theta\!\left(
S_E(e\mid X),
S_C(c\mid X)
\right),
\]

where \(C_\theta\) is a copula with dependence parameter \(\theta\).

For an Archimedean copula,

\[
C_\theta(u_1,u_2)
=
\varphi_\theta^{-1}
\left(
\varphi_\theta(u_1)+\varphi_\theta(u_2)
\right),
\]

where \(\varphi_\theta\) is the generator.

Different copula families encode different dependence structures. Copula selection should therefore be guided by both empirical fit and the dependence patterns that are scientifically plausible for the application.

---

## Methods in this repository

The repository contains implementations of:

- copula-based modeling of event and censoring times,
- copula-graphic estimation,
- margin-time estimation,
- dependence-adjusted Brier score and integrated Brier score,
- dependence-adjusted concordance evaluation,
- dependence-adjusted event-time error,
- standard Kaplan--Meier and IPCW baselines, and
- synthetic and semi-synthetic evaluation experiments.

Reusable implementations are located in [`src/`](src/).

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone TODO_REPOSITORY_URL
cd TODO_REPOSITORY_NAME

python -m venv .venv
```

Activate the environment.

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Repository structure

```text
.
├── notebooks/       # Analysis and plotting notebooks
├── plots/           # Figures used in the paper and repository documentation
├── results/         # Processed experimental results
├── scripts/         # Experiment, aggregation, and plotting entry points
├── src/             # Copulas, estimators, metrics, models, and utilities
├── requirements.txt
├── LICENSE
└── README.md
```

### `src/`

Reusable implementations of the methods introduced or evaluated in the paper.

### `scripts/`

Command-line scripts for running experiments, aggregating outputs, and generating figures.

### `notebooks/`

Notebooks for inspecting results and reproducing selected figures and tables. Core methodological implementations should remain in `src/` rather than being defined only in notebooks.

### `results/`

Processed results underlying the reported tables and figures. Large intermediate outputs are not included in the repository.

### `plots/`

Final figures used in the paper, appendix, and repository documentation.

---

## Reproducing the experiments

The paper contains two main groups of experiments:

1. controlled synthetic experiments, and
2. semi-synthetic experiments based on real covariates.

### Synthetic experiments

The synthetic experiments generate event and censoring times under controlled:

- copula families,
- dependence strengths,
- censoring levels,
- marginal time distributions, and
- model specifications.

```bash
python scripts/TODO_SYNTHETIC_SCRIPT.py
```

### Semi-synthetic experiments

The semi-synthetic experiments use covariates from real survival datasets while simulating event and censoring times. Because the latent event times are known, the bias of each evaluation metric can be measured against oracle evaluation.

```bash
python scripts/TODO_SEMISYNTHETIC_SCRIPT.py
```

### Generate tables and figures

Aggregate the experimental outputs:

```bash
python scripts/TODO_AGGREGATION_SCRIPT.py
```

Generate the paper figures:

```bash
python scripts/TODO_PLOTTING_SCRIPT.py
```

Plotting notebooks are available in [`notebooks/`](notebooks/).

---

## Data

The synthetic datasets are generated entirely from code.

The semi-synthetic experiments use covariates from public or access-controlled survival datasets. Some source datasets cannot be redistributed through this repository.

The final repository should document:

- dataset sources,
- access requirements,
- preprocessing steps,
- expected input formats, and
- train, validation, and test procedures.

No restricted patient-level data should be included in the repository.

---

## Main findings

The experiments show that:

- standard IPCW evaluation is reliable when its censoring assumptions are satisfied;
- evaluation bias increases when event and censoring times remain dependent after conditioning on the available covariates;
- modeling only the marginal censoring distribution does not generally eliminate this bias;
- copula-based corrections can reduce evaluation error when the dependence model is sufficiently well specified; and
- sensitivity to the copula family and dependence parameter should be reported when the dependence structure is uncertain.

The proposed methods do not eliminate the identifiability problem associated with dependent censoring. Instead, they make the dependence assumption explicit and provide tools for evaluation under a specified or estimated dependence structure.

---

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
