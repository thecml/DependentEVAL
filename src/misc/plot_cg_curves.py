import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import config as cfg

from SurvivalEVAL.Evaluations.util import KaplanMeier

from data_loader import SingleEventSyntheticDataLoader

pandas2ri.activate()

compound_cox = importr("compound.Cox")

if __name__ == "__main__":
    # Make data
    t = np.array([1,3,5,4,7,8,10,13])
    e = np.array([1,0,0,1,1,0,1,0])
    
    # Calculate the KM estimate
    km_model = KaplanMeier(t, e)
    km_estimate = torch.from_numpy(km_model.predict(np.sort(t)))
    
    # Calculate the GC estimate alpha=0
    cg_estimate_0 = compound_cox.CG_Clayton(t, e, alpha=0, S_plot=False)
    cg_estimate_18 = compound_cox.CG_Clayton(t, e, alpha=18, S_plot=False)
    
    # Make plots
    plt.figure(figsize=(8, 6))
    plt.plot(np.sort(t), km_estimate, label="KM (indep.)", marker='o')
    plt.plot(cg_estimate_0.rx2('time'), cg_estimate_0.rx2('surv'), label="CG alpha=0 (K_tau=0)")
    plt.plot(cg_estimate_18.rx2('time'), cg_estimate_18.rx2('surv'), label="CG alpha=18 (K_tau=0.9)")
    plt.legend()
    plt.show()
