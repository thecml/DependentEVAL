import torch 
import math 
import numpy as np 

def safe_log(x, eps=1e-6):
    return torch.log(x+eps*(x<eps))

def log1mexp(x):
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    """
    mask = -math.log(2) < x  # x < 0
    return torch.where(
        mask,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )
    
class Clayton_Bivariate:
    def __init__(self, theta, eps, dtype, device):
        self.theta = torch.tensor([theta], device=device).type(dtype)
        self.eps = torch.tensor([eps], device=device).type(dtype)
        self.device = device
    
    def CDF(self, u):
        u = u.clamp(self.eps, 1.0 - self.eps)# can play with this range
        tmp = torch.exp(-self.theta * safe_log(u))
        tmp = torch.sum(tmp, dim=1) - 1.0
        return torch.exp((-1.0 / self.theta) * safe_log(tmp))
    
    def conditional_cdf(self, condition_on, u):
        u_eps = torch.empty_like(u, device=self.device)
        if condition_on == "u":
            u_eps[:,0] = u[:,0] + self.eps
            u_eps[:,1] = u[:,1]  
        elif condition_on == 'v':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] + self.eps
        return (self.CDF(u_eps)-self.CDF(u))/self.eps
    
    def enable_grad(self):
        self.theta.requires_grad = True

    def disable_grad(self):
        self.theta.requires_grad = False

    def parameters(self):
        return [self.theta]
    
    def set_params(self, theta):
        self.theta = theta
    
class Frank_Bivariate:
    def __init__(self, theta, eps, dtype, device) -> None:
        self.theta = torch.tensor([theta], device=device).type(dtype)
        self.eps = torch.tensor([eps], device=device).type(dtype)
        self.device = device
    
    def CDF(self, u):
        tmp = log1mexp(-self.theta*u[:,0]) + log1mexp(-self.theta*u[:,1]) - log1mexp(-self.theta)
        return -1.0 / self.theta * log1mexp(tmp)
    
    def conditional_cdf(self, condition_on, u):
        u_eps = torch.empty_like(u, device=self.device)
        if condition_on == "u":
            u_eps[:,0] = u[:,0] + self.eps
            u_eps[:,1] = u[:,1]  
        elif condition_on == 'v':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] + self.eps
        return (self.CDF(u_eps)-self.CDF(u))/self.eps
    
    def enable_grad(self):
        self.theta.requires_grad = True

    def disable_grad(self):
        self.theta.requires_grad = False

    def parameters(self):
        return [self.theta]
    
    def set_params(self, theta):
        self.theta = theta
    
class Clayton_Triple:
    def __init__(self, theta, eps, dtype, device):
        self.theta = torch.tensor([theta], device=device).type(dtype)
        self.eps = torch.tensor([eps], device=device).type(dtype)
        self.device = device

    def CDF(self, u):
        u = u.clamp(self.eps, 1.0-self.eps * 0.0)
        tmp = torch.exp(-self.theta * safe_log(u))
        tmp = torch.sum(tmp, dim = 1) - 2.0
        return torch.exp((-1.0 / self.theta) * safe_log(tmp))

    def conditional_cdf(self, condition_on, u):
        u_eps = torch.empty_like(u, device=self.device)
        if condition_on == "u":
            u_eps[:,0] = u[:,0] + self.eps
            u_eps[:,1] = u[:,1] 
            u_eps[:,2] = u[:,2] 
        elif condition_on == 'v':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] + self.eps
            u_eps[:,2] = u[:,2] 
        elif condition_on == 'w':
            u_eps[:,0] = u[:,0] 
            u_eps[:,1] = u[:,1] 
            u_eps[:,2] = u[:,2] + self.eps
        return (self.CDF(u_eps)-self.CDF(u))/self.eps

    def enable_grad(self):
        self.theta.requires_grad = True

    def disable_grad(self):
        self.theta.requires_grad = False
    
    def parameters(self):
        return [self.theta]

    def __str__(self) -> str:
        return "Clayton theta: " + str(np.round(self.theta.detach().clone().item(),3))
    
    def set_params(self, theta):
        self.theta = theta
    
class Frank_Triple:
    def __init__(self, theta, eps, dtype, device):
        self.theta = torch.tensor([theta], device=device).type(dtype)
        self.eps = torch.tensor([eps], device=device).type(dtype)
        self.device = device

    def CDF(self, u):
        tmp = log1mexp(-self.theta*u[:,0]) + log1mexp(-self.theta*u[:,1]) + log1mexp(-self.theta*u[:,2]) - log1mexp(-self.theta) - log1mexp(-self.theta)
        return -1.0 / self.theta * log1mexp(tmp)
    
    def conditional_cdf(self, condition_on, uv):
        uv_eps = torch.empty_like(uv, device=self.device)
        if condition_on == "u":
            uv_eps[:,0] = uv[:,0] + self.eps
            uv_eps[:,1] = uv[:,1]
            uv_eps[:,2] = uv[:,2]
        elif condition_on == 'v':
            uv_eps[:,1] = uv[:,1] + self.eps
            uv_eps[:,0] = uv[:,0]
            uv_eps[:,2] = uv[:,2]
        elif condition_on == 'w':
            uv_eps[:,2] = uv[:,2] + self.eps
            uv_eps[:,1] = uv[:,1]
            uv_eps[:,0] = uv[:,0]
        return (self.CDF(uv_eps) - self.CDF(uv))/self.eps
    
    def enable_grad(self):
        self.theta.requires_grad = True

    def disable_grad(self):
        self.theta.requires_grad = False
        
    def parameters(self):
        return [self.theta]

    def __str__(self) -> str:
        return "Frank theta: " + str(np.round(self.theta.detach().clone().item(),3))
    
    def set_params(self, theta):
        self.theta = theta