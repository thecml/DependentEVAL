from SurvivalEVAL import mean_error
import numpy as np
import pandas as pd
from typing import Optional
import warnings
import torch
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import config as cfg
from dataclasses import InitVar, dataclass, field

from metrics import mae_dependent

if __name__ == "__main__":
    # Test the functions
    train_t = np.array([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                        26, 27, 28, 29, 30, 31, 32, 33, 34,  60, 61, 62, 63, 64, 65, 66, 67,
                        74, 75, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
                        98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                        117, 118, 119, 120, 120, 120, 121, 121, 124, 125, 126, 127, 128, 129,
                        136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
                        155, 156, 157, 158, 159, 161, 182, 183, 186, 190, 191, 192, 192, 192,
                        193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 202, 203,
                        204, 202, 203, 204, 212, 213, 214, 215, 216, 217, 222, 223, 224])
    train_e = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                        1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                        0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                        1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                        0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    t = np.array([5, 10, 19, 31, 43, 59, 63, 75, 97, 113, 134, 151, 163, 176, 182, 195, 200, 210, 220])
    e = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
    predict_time = np.array([18, 19, 5, 12, 75, 100, 120, 85, 36, 95, 170, 41, 200, 210, 260, 86, 100, 120, 140])
    
    mae_km = mean_error(predict_time, t, e, train_t, train_e, method='Margin', weighted=False)
    mae_dep = mae_dependent(predict_time, t, e, train_t, train_e) # dependent based on margin
    
    print(mae_km)
    print(mae_dep)