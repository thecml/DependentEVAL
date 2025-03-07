import h5py
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from pycop import simulation
import torch
from utility.survival import convert_to_structured, kendall_tau_to_theta
from utility.survival import make_stratified_split
from dgp import DGP_Weibull_linear, DGP_Weibull_nonlinear
import config as cfg
from pathlib import Path
from sksurv.datasets import load_gbsg2, load_aids, load_whas500, load_flchain
from collections import defaultdict

def _make_df(data):
    x = data['x']
    t = data['t']
    d = data['e']
    colnames = ['x'+str(i) for i in range(x.shape[1])]
    df = (pd.DataFrame(x, columns=colnames)
          .assign(duration=t)
          .assign(event=d))
    return df

class BaseDataLoader(ABC):
    """
    Base class for data loaders.
    """
    def __init__(self):
        """Initilizer method that takes a file path, file name,
        settings and optionally a converter"""
        self.X: pd.DataFrame = None
        self.y: np.ndarray = None
        self.num_features: List[str] = None
        self.cat_features: List[str] = None
        self.min_time = None
        self.max_time = None
        self.n_events = None
        self.params = None

    @abstractmethod
    def load_data(self, n_samples) -> None:
        """Loads the data from a data set at startup"""
        
    @abstractmethod
    def split_data(self) -> None:
        """Loads the data from a data set at startup"""

    def get_data(self) -> pd.DataFrame:
        """
        This method returns the features and targets
        :return: df
        """
        df = pd.DataFrame(self.X)
        df['time'] = self.y['time']
        df['event'] = self.y['event']
        return df

    def get_features(self) -> List[str]:
        """
        This method returns the names of numerical and categorial features
        :return: the columns of X as a list
        """
        return self.num_features, self.cat_features

    def _get_num_features(self, data) -> List[str]:
        return data.select_dtypes(include=np.number).columns.tolist()

    def _get_cat_features(self, data) -> List[str]:
        return data.select_dtypes(['object']).columns.tolist()

def get_data_loader(dataset_name: str) -> BaseDataLoader:
    if dataset_name == "synthetic":
        return SingleEventSyntheticDataLoader()
    elif dataset_name == "gbsg":
        return GbsgDataLoader()
    elif dataset_name == "metabric":
        return MetabricDataLoader()
    elif dataset_name == "mimic_all":
        return MimicAllDataLoader()
    elif dataset_name == "mimic_hospital":
        return MimicHospitalDataLoader()
    elif dataset_name == "nacd":
        return NacdDataLoader()
    elif dataset_name == "support":
        return SupportDataLoader()
    elif dataset_name == "whas":
        return WhasDataLoader()
    elif dataset_name == "aids":
        return AidsDataLoader()
    elif dataset_name == "seer_brain":
        return SeerBrainDataLoader()
    elif dataset_name == "seer_breast":
        return SeerBreastDataLoader()
    elif dataset_name == "seer_liver":
        return SeerLiverDataLoader()
    elif dataset_name == "seer_prostate":
        return SeerProstateDataLoader()
    elif dataset_name == "seer_stomach":
        return SeerStomachDataLoader()
    else:
        raise NotImplementedError()

class SingleEventSyntheticDataLoader(BaseDataLoader):
    def load_data(self, data_config, copula_name='clayton', k_tau=0,
                  linear=True, device='cpu', dtype=torch.float64):
        """
        This method generates synthetic data for single event (and censoring)
        DGP1: Data generation process for event
        DGP2: Data generation process for censoring
        """
        alpha_e1 = data_config['alpha_e1']
        alpha_e2 = data_config['alpha_e2']
        gamma_e1 = data_config['gamma_e1']
        gamma_e2 = data_config['gamma_e2']
        n_samples = data_config['n_samples']
        n_features = data_config['n_features']
        
        X = torch.rand((n_samples, n_features), device=device, dtype=dtype)

        if linear:
            dgp1 = DGP_Weibull_linear(n_features, alpha_e1, gamma_e1, device, dtype)
            dgp2 = DGP_Weibull_linear(n_features, alpha_e2, gamma_e2, device, dtype)
        else:
            dgp1 = DGP_Weibull_nonlinear(n_features, alpha=alpha_e1,
                                         gamma=gamma_e1, device=device, dtype=dtype)
            dgp2 = DGP_Weibull_nonlinear(n_features, alpha=alpha_e2,
                                         gamma=gamma_e2, device=device, dtype=dtype)
            
        if copula_name is None or k_tau == 0:
            rng = np.random.default_rng(0)
            u = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
            v = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
            uv = torch.stack([u, v], dim=1)
        else:
            theta = kendall_tau_to_theta(copula_name, k_tau)
            u, v = simulation.simu_archimedean(copula_name, 2, X.shape[0], theta=theta)
            u = torch.from_numpy(u).type(dtype).reshape(-1,1)
            v = torch.from_numpy(v).type(dtype).reshape(-1,1)
            uv = torch.cat([u, v], axis=1)
        
        t1_times = dgp1.rvs(X, uv[:,0].to(device))
        t2_times = dgp2.rvs(X, uv[:,1].to(device))
        
        observed_times = np.minimum(t1_times, t2_times)
        event_indicators = np.array((t2_times < t1_times), dtype=np.int32)
        
        self.true_censor_times = t1_times
        self.true_event_times = t2_times
    
        columns = [f'X{i}' for i in range(n_features)]
        self.X = pd.DataFrame(X.cpu(), columns=columns)
        self.y = convert_to_structured(observed_times, event_indicators)
        self.dgps = [dgp1, dgp2]
        
        return self
    
    def split_data(self, train_size: float, valid_size: float,
                   test_size: float, dtype=torch.float64, random_state=0):
        df = pd.DataFrame(self.X)
        df['event'] = self.y_e
        df['time'] = self.y_t
        df['true_time'] = self.true_event_times
    
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='time', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            random_state=random_state)
    
        dataframes = [df_train, df_valid, df_test]
        dicts = []
        n_features = df.shape[1] - 2
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = torch.tensor(dataframe.loc[:, 'X0':f'X{n_features-1}'].to_numpy(), dtype=dtype)
            data_dict['E'] = torch.tensor(dataframe['event'].to_numpy(), dtype=dtype)
            data_dict['T'] = torch.tensor(dataframe['time'].to_numpy(), dtype=dtype)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]

class MimicAllDataLoader(BaseDataLoader):
    """
    Data loader for MIMIC dataset
    """
    def load_data(self, n_samples:int = None):
        '''
        t and e order, followed by death
        '''
        path = Path.joinpath(cfg.DATA_DIR, "mimic_all_causes.csv")
        data = pd.read_csv(path)
        skip_cols = ['event', 'is_male', 'time', 'is_white', 'renal', 'cns', 'coagulation', 'cardiovascular']
        cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
        
        self.num_features = cols_standardize
        self.cat_features = ['is_male', 'is_white', 'renal', 'cns', 'coagulation', 'cardiovascular']
        
        self.X = data.drop(['event', 'time'], axis=1)
        self.y = convert_to_structured(data['time'], data['event'])
        self.columns = list(self.X.columns)
        
        return self

    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float64,
                random_state=0):
        raise NotImplementedError()
    
class MimicHospitalDataLoader(BaseDataLoader):
    """
    Data loader for MIMIC dataset
    """
    def load_data(self, n_samples:int = None):
        '''
        t and e order, followed by death
        '''
        path = Path.joinpath(cfg.DATA_DIR, "mimic_hosp_failure.csv")
        data = pd.read_csv(path)
        skip_cols = ['event', 'is_male', 'time', 'is_white', 'renal', 'cns', 'coagulation', 'cardiovascular']
        cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
        
        self.num_features = ['age']
        self.cat_features = ['male', 'is_white', 'ins_medicare', 'ins_medicaid', 'english',
                             'marital', 'had_ed', 'svrty', 'mtlty']
        
        self.X = data.drop(['event', 'time'], axis=1)
        self.y = convert_to_structured(data['time'], data['event'])
        self.columns = list(self.X.columns)
        
        return self

    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float64,
                random_state=0):
        raise NotImplementedError()    

class MetabricDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        data = pd.read_feather(Path.joinpath(cfg.DATA_DIR, 'metabric.feather')) 
        data['duration'] = data['duration'].apply(round)

        data = data.loc[data['duration'] > 0]
        
        outcomes = data.copy()
        outcomes['event'] =  data['event']
        outcomes['time'] = data['duration']
        outcomes = outcomes[['event', 'time']]

        self.num_features = ['x0', 'x1', 'x2', 'x3', 'x8']
        self.cat_features = ['x4', 'x5', 'x6', 'x7']
                    
        self.X = pd.DataFrame(data.drop(['duration', 'event'], axis=1), dtype=np.float64)
        self.y = convert_to_structured(outcomes['time'], outcomes['event'])

        return self

    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float64,
                random_state=0):
        raise NotImplementedError()

class SupportDataLoader(BaseDataLoader):
    """
    Data loader for SUPPORT dataset
    """
    def load_data(self, n_samples:int = None) -> None:
        path = Path.joinpath(cfg.DATA_DIR, 'support.feather')
        data = pd.read_feather(path)

        if n_samples:
            data = data.sample(n=n_samples, random_state=0)

        data = data.loc[data['duration'] > 0]

        self.num_features = ['x0', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
        self.cat_features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
        self.X = pd.DataFrame(data.drop(['duration', 'event'], axis=1), dtype=np.float64)
        self.y = convert_to_structured(data['duration'], data['event'])

        return self
    
    def split_data(self, train_size: float, valid_size: float,
                   test_size: float, dtype=torch.float64, random_state=0):
        raise NotImplementedError()

class AidsDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        X, y = load_aids()

        self.X = pd.DataFrame(X)
        self.y = convert_to_structured(y['time'], y['censor'])
        self.num_features = ['age', 'cd4', 'karnof', 'priorzdv']
        self.cat_features = ['hemophil', 'ivdrug', 'raceth', 'sex', 'strat2', 'tx', 'txgrp']
        return self
    
    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float64,
                random_state=0):
        raise NotImplementedError()

class GbsgDataLoader(BaseDataLoader):
    def load_data(self) -> BaseDataLoader:
        cols_to_drop = ['pid']
        path = Path.joinpath(cfg.DATA_DIR, 'gbsg.csv')
        data = pd.read_csv(path).drop(cols_to_drop, axis=1).rename(
            columns={"status": "event", "rfstime": "time"})
        
        self.X = pd.DataFrame(data.drop(['event', 'time'], axis=1))
        self.y = convert_to_structured(data['time'], data['event'])
        self.num_features = ['age', 'size', 'grade', 'nodes', 'pgr', 'er']
        self.cat_features = ['meno', 'hormon']
        
        return self

    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float64,
                random_state=0):
        raise NotImplementedError()

class WhasDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        X, y = load_whas500()

        self.X = pd.DataFrame(X)
        self.y = convert_to_structured(y['lenfol'], y['fstat'])
        self.num_features = ['age', 'bmi', 'diasbp', 'hr', 'los', 'sysbp']
        self.cat_features = ['afb', 'av3', 'chf', 'cvd', 'gender', 'miord', 'mitype', 'sho']
        
        return self

    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float64,
                random_state=0):
        raise NotImplementedError()
    
class NacdDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        cols_to_drop = ['PERFORMANCE_STATUS', 'STAGE_NUMERICAL', 'AGE65']
        path = Path.joinpath(cfg.DATA_DIR, "nacd_full.csv")
        data = pd.read_csv(path).drop(cols_to_drop, axis=1).rename(columns={"delta": "event"})

        data = data.drop(data[data["time"] <= 0].index)  # remove patients with negative or zero survival time
        data.reset_index(drop=True, inplace=True)
        
        self.X = pd.DataFrame(data.drop(['time', 'event'], axis=1))
        self.y = convert_to_structured(data['time'], data['event'])
        self.num_features = ['BOX1_SCORE', 'BOX2_SCORE', 'BOX3_SCORE', 'BMI', 'WEIGHT_CHANGEPOINT',
                             'AGE', 'GRANULOCYTES', 'LDH_SERUM', 'LYMPHOCYTES',
                             'PLATELET', 'WBC_COUNT', 'CALCIUM_SERUM', 'HGB', 'CREATININE_SERUM', 'ALBUMIN']
        feature_cols = data.drop(['time', 'event'], axis=1).columns.to_list()
        self.cat_features = list(set(feature_cols).symmetric_difference(self.num_features))

        return self

    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float64,
                random_state=0):
        raise NotImplementedError()
    
class SeerBrainDataLoader(BaseDataLoader):
    def load_data(self, n_samples=None) -> None:
        path = Path.joinpath(cfg.DATA_DIR, "seer_brain.csv")
        data = pd.read_csv(path).rename(columns={"Survival months": "time"})
        data = data.drop(data[data["time"] <= 0].index)  # remove patients with negative or zero survival time
        data.reset_index(drop=True, inplace=True)

        skip_cols = ['event', 'time', 'Sex', 'Behavior recode for analysis',
                    'SEER historic stage A (1973-2015)', 'RX Summ--Scope Reg LN Sur (2003+)']
        cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
        
        self.X = pd.DataFrame(data.drop(['time', 'event'], axis=1))
        self.y = convert_to_structured(data['time'], data['event'])
        self.num_features = cols_standardize
        feature_cols = data.drop(['time', 'event'], axis=1).columns.to_list()
        self.cat_features = list(set(feature_cols).symmetric_difference(self.num_features))

        return self

    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float64,
                random_state=0):
        raise NotImplementedError()

class SeerBreastDataLoader(BaseDataLoader):
    def load_data(self, n_samples=None) -> None:
        path = Path.joinpath(cfg.DATA_DIR, "seer_breast.csv")
        data = pd.read_csv(path).rename(columns={"Survival months": "time"})
        data = data.drop(data[data["time"] <= 0].index)  # remove patients with negative or zero survival time
        data.reset_index(drop=True, inplace=True)

        skip_cols = ['event', 'time', 'RX Summ--Scope Reg LN Sur (2003+)']
        cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))

        self.X = pd.DataFrame(data.drop(['time', 'event'], axis=1))
        self.y = convert_to_structured(data['time'], data['event'])
        self.num_features = cols_standardize
        feature_cols = data.drop(['time', 'event'], axis=1).columns.to_list()
        self.cat_features = list(set(feature_cols).symmetric_difference(self.num_features))

        return self

    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float64,
                random_state=0):
        raise NotImplementedError()

class SeerLiverDataLoader(BaseDataLoader):
    def load_data(self, n_samples=None) -> None:
        path = Path.joinpath(cfg.DATA_DIR, "seer_liver.csv")
        data = pd.read_csv(path).rename(columns={"Survival months": "time"})
        data = data.drop(data[data["time"] <= 0].index)  # remove patients with negative or zero survival time
        data.reset_index(drop=True, inplace=True)

        skip_cols = ['event', 'time', 'Sex']
        cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))

        self.X = pd.DataFrame(data.drop(['time', 'event'], axis=1))
        self.y = convert_to_structured(data['time'], data['event'])
        self.num_features = cols_standardize
        feature_cols = data.drop(['time', 'event'], axis=1).columns.to_list()
        self.cat_features = list(set(feature_cols).symmetric_difference(self.num_features))

        return self

    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float64,
                random_state=0):
        raise NotImplementedError()

class SeerProstateDataLoader(BaseDataLoader):
    def load_data(self, n_samples=None) -> None:
        path = Path.joinpath(cfg.DATA_DIR, "seer_prostate.csv")
        data = pd.read_csv(path).rename(columns={"Survival months": "time"})
        data = data.drop(data[data["time"] <= 0].index)  # remove patients with negative or zero survival time
        data.reset_index(drop=True, inplace=True)

        skip_cols = ['event', 'time']
        cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))

        self.X = pd.DataFrame(data.drop(['time', 'event'], axis=1))
        self.y = convert_to_structured(data['time'], data['event'])
        self.num_features = cols_standardize
        feature_cols = data.drop(['time', 'event'], axis=1).columns.to_list()
        self.cat_features = list(set(feature_cols).symmetric_difference(self.num_features))

        return self

    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float64,
                random_state=0):
        raise NotImplementedError()
    
class SeerStomachDataLoader(BaseDataLoader):
    def load_data(self, n_samples=None) -> None:
        path = Path.joinpath(cfg.DATA_DIR, "seer_stomach.csv")
        data = pd.read_csv(path).rename(columns={"Survival months": "time"})
        data = data.drop(data[data["time"] <= 0].index)  # remove patients with negative or zero survival time
        data.reset_index(drop=True, inplace=True)

        skip_cols = ['event', 'time', 'Sex']
        cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
        
        self.X = pd.DataFrame(data.drop(['time', 'event'], axis=1))
        self.y = convert_to_structured(data['time'], data['event'])
        self.num_features = cols_standardize
        feature_cols = data.drop(['time', 'event'], axis=1).columns.to_list()
        self.cat_features = list(set(feature_cols).symmetric_difference(self.num_features))

        return self

    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float64,
                random_state=0):
        raise NotImplementedError()
    