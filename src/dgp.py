import torch
import torch.nn as nn
from typing import List

def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

class DGP_LogNormal_linear:
    # Note this is the LogNormal model, not the LogNormal CoxPH model
    def __init__(self, mu: List[float], sigma: List[float], device='cpu', dtype=torch.float64) -> None:
        self.mu_coeff = torch.tensor(mu, device=device).type(dtype)
        self.sigma_coeff = torch.tensor(sigma, device=device).type(dtype)
        self.device = device

    def hazard(self, t, x):
        return self.PDF(t, x) / self.survival(t, x)
    
    def cum_hazard(self, t, x):
        # No closed form solution for the cumulative hazard function of the lognormal distribution
        # we have to approximate it using numerical integration
        n_steps = 1000
        t_grids = torch.linspace(1e-5, t, n_steps).to(self.device)  # avoid zero to prevent division by zero
        delta_u = t / n_steps
        h_values = self.hazard(t_grids, x)
        H_t = torch.sum(h_values * delta_u)
        return H_t

    def survival(self, t, x):
        return 1 - self.CDF(t, x)

    def CDF(self, t, x):
        mu, sigma = self.pred_params(x)
        return 0.5 + 0.5 * torch.erf((LOG(t) - mu) / (sigma * torch.sqrt(torch.tensor(2)).to(self.device)))

    def PDF(self, t, x):
        mu, sigma = self.pred_params(x)
        return (1 / (t * sigma * torch.sqrt(torch.tensor(2 * torch.pi)).to(self.device))) * torch.exp(
            -((LOG(t) - mu) ** 2) / (2 * sigma ** 2))

    def parameters(self):
        return [self.mu_coeff, self.sigma_coeff]

    def pred_params(self, x):
        mu = torch.matmul(x, self.mu_coeff)
        sigma = torch.matmul(x, self.sigma_coeff)
        return mu, sigma

    def rvs(self, x, u):
        mu, sigma = self.pred_params(x)
        return torch.exp(mu + sigma * torch.erfinv(2 * u - 1) * torch.sqrt(torch.tensor(2)).to(self.device))

class DGP_LogNormal_nonlinear(DGP_LogNormal_linear):  # Nonlinear LogNormal model
    def __init__(self, n_features, mu: list, sigma: list,
                 hidden_dim=32, device='cpu', dtype=torch.float64) -> None:
        super().__init__(mu, sigma, device, dtype)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device).type(dtype)

    def pred_params(self, x):
        risks = self.net(x)
        mu = torch.matmul(risks, self.mu_coeff)
        sigma = torch.matmul(risks, self.sigma_coeff)
        return mu, sigma

    def parameters(self):
        return list(self.net.parameters()) + [self.mu_coeff, self.sigma_coeff]

class DGP_LogNormalCox_linear: # This is linear lognormal CoxPH model
    def __init__(self, n_features, mu: float, sigma: float, device="cpu", dtype=torch.float64) -> None:
        self.mu = torch.tensor([mu]).type(dtype).to(device)
        self.sigma = torch.tensor([sigma]).type(dtype).to(device)
        self.coeff = torch.rand((n_features,)).to(device)

    def bl_hazard(self, t):
        # The baseline hazard function of the lognormal CoxPH model, here we use the hazard function of the lognormal
        # distribution as the baseline hazard function
        pdf = (1 / (t * self.sigma * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(
            -((LOG(t) - self.mu) ** 2) / (2 * self.sigma ** 2))
        survival = 0.5 - 0.5 * torch.erf((LOG(t) - self.mu) / (self.sigma * torch.sqrt(torch.tensor(2))))
        return pdf / survival

    def hazard(self, t, x):
        return self.bl_hazard(t) * torch.exp(torch.matmul(x, self.coeff))

    def cum_hazard(self, t, x):
        # No closed form solution for the cumulative hazard function of the lognormal distribution
        # we have to approximate it using numerical integration
        n_steps = 1000
        t_grids = torch.linspace(1e-5, t, n_steps)  # avoid zero to prevent division by zero
        delta_u = t / n_steps
        h_values = self.hazard(t_grids, x)
        H_t = torch.sum(h_values * delta_u)
        return H_t

    def survival(self, t, x):
        return torch.exp(-self.cum_hazard(t, x))

    def CDF(self, t, x):
        return 1.0 - self.survival(t, x)

    def PDF(self, t, x):
        return self.survival(t,x)*self.hazard(t,x)
    
    def parameters(self):
        return [self.mu, self.sigma, self.coeff]

    def rvs(self, x, u):
        # TODO: no closed form solution for the lognormal CoxPH model
        raise NotImplementedError

class DGP_Exp_linear: # This is linear exponential PH model
    def __init__(self, n_features, baseline_hazard: float, device="cpu", dtype=torch.float64) -> None:
        self.bh = torch.tensor([baseline_hazard]).type(dtype).to(device)
        self.coeff = torch.rand((n_features,)).to(device)
    
    def hazard(self, t, x):
        return self.bh * torch.exp(torch.matmul(x, self.coeff))
    
    def cum_hazard(self, t, x):
        return self.hazard(t, x) * t
    
    def survival(self, t, x):
        return torch.exp(-self.cum_hazard(t, x))
    
    def CDF(self, t, x):
        return 1.0 - self.survival(t, x)
    
    def PDF(self, t, x):
        return self.survival(t,x)*self.hazard(t,x)
        
    def parameters(self):
        return [self.bh, self.coeff]
    
    def rvs(self, x, u):
        return -LOG(u)/self.hazard(t=None, x=x)

class DGP_EXP_nonlinear(DGP_Exp_linear):  # This is the nonlinear exponential PH model
    def __init__(self, n_features, baseline_hazard: float, 
                 hidden_dim=32, device='cpu', dtype=torch.float64) -> None:
        super().__init__(n_features, baseline_hazard, device, dtype)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device).type(dtype)

    def hazard(self, t, x):
        risks = self.net(x)
        return self.bh * torch.exp(risks)

    def parameters(self):
        return list(self.net.parameters()) + [self.bh]

class DGP_Weibull_linear:
    def __init__(self, n_features, alpha: float, gamma: float, use_x=True,
                 device="cpu", dtype=torch.float64, coeff=None, generator=None):
        self.alpha = torch.tensor([alpha], device=device).type(dtype)
        self.gamma = torch.tensor([gamma], device=device).type(dtype)
        self.use_x = use_x
        self.device = device
        self.dtype = dtype

        if coeff is not None:
            self.coeff = coeff.to(device=device, dtype=dtype)
        else:
            if generator is None:
                self.coeff = 2 * torch.rand((n_features,), device=device).type(dtype) - 1
            else:
                self.coeff = 2 * torch.rand((n_features,), generator=generator, device=device, dtype=dtype) - 1

    def PDF(self, t, x):
        return self.hazard(t, x) * self.survival(t, x)
    
    def CDF(self, t, x):
        return 1 - self.survival(t, x)
    
    def survival(self, t, x):
        return torch.exp(-self.cum_hazard(t, x))
    
    def hazard(self, t, x):
        linear_term = torch.matmul(x, self.coeff) if self.use_x else 0.0
        return ((self.gamma / self.alpha) * ((t / self.alpha) ** (self.gamma - 1))) * torch.exp(linear_term)

    def cum_hazard(self, t, x):
        linear_term = torch.matmul(x, self.coeff) if self.use_x else 0.0
        return ((t / self.alpha) ** self.gamma) * torch.exp(linear_term)

    def parameters(self):
        return [self.alpha, self.gamma, self.coeff]
    
    def rvs(self, x, u):
        zero = torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)
        linear_term = torch.matmul(x, self.coeff) if self.use_x else zero
        survival_term = -torch.log(u) / torch.exp(linear_term)
        result = (survival_term ** (1 / self.gamma)) * self.alpha
        return result.detach().cpu().numpy()
    
class DGP_Weibull_nonlinear:
    def __init__(self, n_features, alpha: float, gamma: float, hidden_dim: int = 32,
                 use_x: bool = True, device: str = "cpu", dtype: torch.dtype = torch.float64,
                 coeff=None, generator=None):
        self.alpha = torch.tensor([alpha], device=device, dtype=dtype)
        self.gamma = torch.tensor([gamma], device=device, dtype=dtype)
        self.use_x = use_x
        self.device = device
        self.dtype = dtype

        # Build net
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device=device, dtype=dtype)

        # Optional: reuse externally-provided parameters (like coeff in linear DGP)
        # Here, `coeff` is interpreted as a state_dict for the network.
        if coeff is not None:
            # Accept either a plain state_dict or an nn.Module (from which we take state_dict)
            state = coeff.state_dict() if hasattr(coeff, "state_dict") else coeff
            self.net.load_state_dict(state)

        # Optional: reproducible initialization
        elif generator is not None:
            self._reset_parameters_with_generator(generator)

    def _reset_parameters_with_generator(self, generator: torch.Generator):
        """
        Re-initialize net weights deterministically using the provided torch.Generator.
        """
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                # Keep it simple: uniform init in [-bound, bound] like common defaults,
                # but using the passed generator for reproducibility.
                fan_in = m.weight.shape[1]
                bound = 1.0 / (fan_in ** 0.5)

                m.weight.data = (2 * torch.rand_like(m.weight, generator=generator) - 1) * bound
                if m.bias is not None:
                    m.bias.data = (2 * torch.rand_like(m.bias, generator=generator) - 1) * bound

    def PDF(self, t, x):
        return self.hazard(t, x) * self.survival(t, x)

    def CDF(self, t, x):
        return 1 - self.survival(t, x)

    def survival(self, t, x):
        return torch.exp(-self.cum_hazard(t, x))

    def _nonlinear_term(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns exp(net(x)) squeezed to shape (n,) (or scalar if n=1),
        or 1.0 if use_x=False (so you recover the baseline Weibull).
        """
        if not self.use_x:
            # baseline hazard multiplier
            return torch.ones((x.shape[0],), device=x.device, dtype=x.dtype)
        return torch.exp(self.net(x)).squeeze(-1)

    def hazard(self, t, x):
        nonlinear_term = self._nonlinear_term(x)
        base = (self.gamma / self.alpha) * ((t / self.alpha) ** (self.gamma - 1))
        return base * nonlinear_term

    def cum_hazard(self, t, x):
        nonlinear_term = self._nonlinear_term(x)
        return ((t / self.alpha) ** self.gamma) * nonlinear_term

    def parameters(self):
        return [self.alpha, self.gamma] + list(self.net.parameters())

    def rvs(self, x, u):
        nonlinear_term = self._nonlinear_term(x)
        survival_term = -torch.log(u) / nonlinear_term
        result = (survival_term ** (1 / self.gamma)) * self.alpha
        return result.detach().cpu().numpy()
