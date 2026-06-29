from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

import numpy as np
import pandas as pd
import warnings

class RobustCoxPHSurvivalAnalysis:
    """
    Small sklearn-style wrapper around scikit-survival CoxPH.
    It makes CoxPH usable in semi-synthetic sweeps where some dataset/strategy/seed
    combinations produce collinear or separating covariates. The wrapper:
      - cleans non-finite values,
      - imputes using training medians,
      - drops constant columns,
      - standardizes features,
      - retries CoxPH with stronger ridge penalties.
    """
    def __init__(self, alpha=1e-4, ties="breslow", n_iter=100, tol=1e-9):
        self.alpha = float(alpha)
        self.ties = ties
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.model_ = None
        self.alpha_ = None
        self.columns_ = None
        self.keep_columns_ = None
        self.median_ = None
        self.mean_ = None
        self.std_ = None

    def _prepare_fit_X(self, X):
        X = pd.DataFrame(X).copy()
        self.columns_ = list(X.columns)

        X = X.replace([np.inf, -np.inf], np.nan).astype(float)
        self.median_ = X.median(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X = X.fillna(self.median_)

        # Drop constant / near-constant features, which are common causes of
        # CoxPH convergence failures with feature-selection regimes.
        var = X.var(axis=0)
        self.keep_columns_ = var.index[np.asarray(var > 1e-12)].tolist()
        if len(self.keep_columns_) == 0:
            raise ValueError("No non-constant features left for CoxPH.")
        X = X[self.keep_columns_]

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0).replace(0.0, 1.0).fillna(1.0)
        X = (X - self.mean_) / self.std_
        X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return X

    def _prepare_predict_X(self, X):
        X = pd.DataFrame(X).copy()

        # Preserve the original training column order when possible.
        if self.columns_ is not None and set(self.columns_).issubset(set(X.columns)):
            X = X[self.columns_]

        X = X.replace([np.inf, -np.inf], np.nan).astype(float)
        X = X.fillna(self.median_)
        X = X[self.keep_columns_]
        X = (X - self.mean_) / self.std_
        X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return X

    def fit(self, X, y):
        X_fit = self._prepare_fit_X(X)

        # Try the requested alpha first, then progressively stronger ridge.
        alphas = []
        for a in [self.alpha, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
            if a not in alphas:
                alphas.append(float(a))

        last_error = None
        for alpha in alphas:
            try:
                model = CoxPHSurvivalAnalysis(
                    alpha=alpha,
                    ties=self.ties,
                    n_iter=self.n_iter,
                    tol=self.tol,
                )
                model.fit(X_fit, y)
                self.model_ = model
                self.alpha_ = alpha
                if alpha != self.alpha:
                    warnings.warn(
                        f"CoxPH fit required stronger ridge alpha={alpha} "
                        f"(requested alpha={self.alpha})."
                    )
                return self
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            "Robust CoxPH failed for all ridge penalties. "
            f"Last error: {last_error}"
        )

    def predict(self, X):
        return self.model_.predict(self._prepare_predict_X(X))
    
    def score(self, X, y):
        return self.model_.score(self._prepare_predict_X(X), y)

    def predict_survival_function(self, X, *args, **kwargs):
        return self.model_.predict_survival_function(
            self._prepare_predict_X(X), *args, **kwargs
        )

    def predict_cumulative_hazard_function(self, X, *args, **kwargs):
        return self.model_.predict_cumulative_hazard_function(
            self._prepare_predict_X(X), *args, **kwargs
        )

def make_cox_model(config):
    n_iter = config['n_iter']
    tol = config['tol']
    alpha = float(config.get('alpha', 1e-4))
    ties = config.get('ties', 'breslow')
    model = RobustCoxPHSurvivalAnalysis(alpha=alpha, ties=ties, n_iter=n_iter, tol=tol)
    return model

def make_gbsa_model(config):
    n_estimators = config['n_estimators']
    learning_rate = config['learning_rate']
    max_depth = config['max_depth']
    loss = config['loss']
    min_samples_split = config['min_samples_split']
    min_samples_leaf = config['min_samples_leaf']
    max_features = config['max_features']
    subsample = config['subsample']
    model = GradientBoostingSurvivalAnalysis(n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            max_depth=max_depth,
                                            loss=loss,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            max_features=max_features,
                                            subsample=subsample,
                                            random_state=0)
    return model

def make_rsf_model(config):
    n_estimators = config['n_estimators']
    max_depth = config['max_depth']
    min_samples_split = config['min_samples_split']
    min_samples_leaf =  config['min_samples_leaf']
    max_features = config['max_features']
    model = RandomSurvivalForest(random_state=0,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                max_features=max_features)
    return model
