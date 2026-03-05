from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

from sota.weibull_aft import WeibullAFTWrapper

def make_cox_model(config):
    n_iter = config['n_iter']
    tol = config['tol']
    model = CoxPHSurvivalAnalysis(alpha=0.0001)
    return model

def make_weibull_aft_model(config):
    penalizer = config['penalizer']
    l1_ratio = config['l1_ratio']
    model = WeibullAFTWrapper(penalizer=penalizer, l1_ratio=l1_ratio)
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