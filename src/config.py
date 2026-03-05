from pathlib import Path
ROOT_DIR = Path(__file__).absolute().parent.parent

RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')
PLOTS_DIR = Path.joinpath(ROOT_DIR, 'plots')
DATA_DIR = Path.joinpath(ROOT_DIR, 'data')

COXPH_PARAMS = {
    'alpha': 0.01,
    'ties': 'breslow',
    'n_iter': 100,
    'tol': 1e-9
}

GBSA_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'loss': 'coxph',
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'subsample': 0.8,
    'seed': 0,
    'test_size': 0.3,
}

RSF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 3,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    "random_state": 0
}

DEEPSURV_PARAMS = {
    'hidden_size': 100,
    'verbose': False,
    'lr': 0.001,
    'c1': 0.01,
    'num_epochs': 1000,
    'batch_size': 32,
    'dropout': 0.25,
    'early_stop': True,
    'patience': 10
}

MTLR_PARAMS = {
    'verbose': False,
    'lr': 0.001,
    'c1': 0.01,
    'num_epochs': 1000,
    'dropout': 0.25,
    'batch_size': 32,
    'early_stop': True,
    'patience': 10
}
