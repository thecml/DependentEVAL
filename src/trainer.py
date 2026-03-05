import torch
from torch.utils.data import DataLoader, TensorDataset

def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

def loss_function(model1, model2, data, copula=None):
    s1 = model1.survival(data['T'], data['X'])
    s2 = model2.survival(data['T'], data['X'])
    f1 = model1.PDF(data['T'], data['X'])
    f2 = model2.PDF(data['T'], data['X'])
    w = torch.mean(data['E'])
    if copula is None:
        p1 = LOG(f1) + LOG(s2)
        p2 = LOG(f2) + LOG(s1)
    else:
        S = torch.cat([s1.reshape(-1,1), s2.reshape(-1,1)], dim=1).clamp(0.001, 0.999)
        p1 = LOG(f1) + LOG(copula.conditional_cdf("u", S))
        p2 = LOG(f2) + LOG(copula.conditional_cdf("v", S))
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    reg = 0.01 * torch.sum(copula.theta ** 2) if copula is not None else 0
    return -torch.mean(p1 * data['E'] + (1 - data['E']) * p2) + reg

def train_copula_model(model1, model2, train_data, val_data,
                       n_epochs, batch_size=32, lr=1e-3,
                       copula_name=None, verbose=False, copula=None,
                       theta_tol=1e-4, theta_patience=100, val_patience=100):
    # Enable gradients for models
    model1.enable_grad()
    model2.enable_grad()
    if copula is not None:
        copula.enable_grad()
    
    best_val_loss = float('inf')
    best_model1_weights = None
    best_model2_weights = None
    best_copula_theta = None
    
    # Optimizer
    optimizer_params = [{"params": model1.parameters(), "lr": 1e-3},
                        {"params": model2.parameters(), "lr": 1e-3}]
    if copula is not None:
        optimizer_params.append({"params": copula.parameters(), "lr": 1e-2})
    
    optimizer = torch.optim.Adam(optimizer_params)

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(train_data['T'], train_data['X'], train_data['E']),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_data['T'], val_data['X'], val_data['E']),
        batch_size=batch_size, shuffle=False
    )

    # Track stagnation
    theta_stop_itr = 0
    last_theta = None
    val_stop_itr = 0
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Training loop
        for T, X, E in train_loader:
            batch_data = {'T': T, 'X': X, 'E': E}
            loss = loss_function(model1, model2, batch_data, copula)

            if torch.isnan(loss).any():
                print(f"NaN detected in training loss at epoch {epoch}. Skipping batch.")
                break

            loss.backward()

            # Scale gradients
            if copula is not None:
                copula.theta.grad = copula.theta.grad * 100
                copula.theta.grad = copula.theta.grad.clamp(-1, 1)

            optimizer.step()

            # Ensure valid theta
            if copula is not None:
                if copula.theta <= 0:
                    with torch.no_grad():
                        copula.theta[:] = torch.clamp(copula.theta, 0.001, 30)

        # Validation
        with torch.no_grad():
            val_loss = 0.0
            for T, X, E in val_loader:
                batch_data = {'T': T, 'X': X, 'E': E}
                batch_val_loss = loss_function(model1, model2, batch_data, copula).item()
                val_loss += batch_val_loss
            val_loss /= len(val_loader)
            
            if verbose and epoch % 10 == 0:
                copula_theta = copula.theta.item() if copula is not None else None
                print(f"Epoch {epoch}, Validation Loss: {val_loss:.3f} - Copula Theta: {copula_theta:.3f}")

            if val_loss < best_val_loss - 1e-6:  # tiny tolerance
                best_model1_weights = model1.state_dict()
                best_model2_weights = model2.state_dict()
                best_copula_theta = copula.theta.detach().clone() if copula is not None else None
                best_val_loss = val_loss
                val_stop_itr = 0  # reset patience if improved
            else:
                val_stop_itr += 1

        # Early stopping
        if copula is not None:
            theta_val = copula.theta.detach().cpu().item()
            
            min_free_theta = 1e-2      # ignore until theta > this
            min_epochs = 2000          # guard against premature stop
            theta_abs_tol = 1e-3
            theta_rel_tol = 1e-4
            
            if abs(theta_val) > min_free_theta:
                if last_theta is None:
                    last_theta = theta_val
                else:
                    abs_change = abs(theta_val - last_theta)
                    denom = max(abs(last_theta), 1e-12)
                    rel_change = abs_change / denom
                    
                    if abs_change < theta_abs_tol and rel_change < theta_rel_tol:
                        theta_stop_itr += 1
                    else:
                        theta_stop_itr = 0
                    last_theta = theta_val
            else:
                # reset until theta escapes small zone
                theta_stop_itr = 0
                last_theta = theta_val

        # --- Combined early stopping ---
        if epoch >= min_epochs and theta_stop_itr >= theta_patience and val_stop_itr >= val_patience:
            print(f"Stopping early at epoch {epoch}: θ converged ({theta_val:.4f}) "
                  f"and validation loss not improving.")
            break

    # Restore best
    if best_model1_weights is not None:
        model1.load_state_dict(best_model1_weights)
    if best_model2_weights is not None:
        model2.load_state_dict(best_model2_weights)
    if copula is not None and best_copula_theta is not None:
        copula.theta = best_copula_theta

    return model1, model2, copula, best_val_loss
