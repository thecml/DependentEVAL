import math
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
                       n_epochs, patience=1000, batch_size=32, lr=1e-3,
                       copula_name=None, verbose=False, copula=None):
    # Enable gradients for models
    model1.enable_grad()
    model2.enable_grad()
    if copula is not None:
        copula.enable_grad()
    
    best_val_loss = float('inf')
    best_model1_weights = None
    best_model2_weights = None
    best_copula_theta = None
    
    # Prepare optimizer
    optimizer_params = [{"params": model1.parameters(), "lr": lr},
                        {"params": model2.parameters(), "lr": lr}]
    if copula is not None:
        optimizer_params.append({"params": copula.parameters(), "lr": lr})
    
    optimizer = torch.optim.Adam(optimizer_params)

    # Create DataLoaders for mini-batching
    train_loader = DataLoader(TensorDataset(train_data['T'], train_data['X'], train_data['E']),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data['T'], val_data['X'], val_data['E']),
                            batch_size=batch_size, shuffle=False)

    stop_itr = 0
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Iterate over mini-batches
        for batch_idx, (T, X, E) in enumerate(train_loader):
            batch_data = {'T': T, 'X': X, 'E': E}
            loss = loss_function(model1, model2, batch_data, copula)
            
            # Check if loss is NaN
            if torch.isnan(loss).any():
                print(f"NaN detected in training loss at epoch {epoch}, batch {batch_idx}. Stopping training.")
                if best_model1_weights is not None:
                    model1.load_state_dict(best_model1_weights)
                if best_model2_weights is not None:
                    model2.load_state_dict(best_model2_weights)
                if copula is not None and best_copula_theta is not None:
                    copula.theta = best_copula_theta
                return model1, model2, copula, best_val_loss

            loss.backward()
                
            # Clip the gradients 
            torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0, norm_type=2)
            torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0, norm_type=2)
            if copula is not None:
                torch.nn.utils.clip_grad_norm_(copula.parameters(), max_norm=1.0, norm_type=2)

            optimizer.step()
            
            # Ensure valid values for theta
            if copula is not None:
                with torch.no_grad():
                    if copula_name == "clayton":
                        copula.theta.data.clamp_(-1, float('inf'))
                    else:
                        copula.theta.data.clamp_(float('-inf'), float('inf'))

        # Validation phase
        with torch.no_grad():
            val_loss = 0.0
            for batch_idx, (T, X, E) in enumerate(val_loader):
                batch_data = {'T': T, 'X': X, 'E': E}
                batch_val_loss = loss_function(model1, model2, batch_data, copula).item()
                
                if isinstance(batch_val_loss, float) and math.isnan(batch_val_loss):
                    print(f"NaN detected in validation loss at epoch {epoch}. Returning previous loss.")
                    if best_model1_weights is not None:
                        model1.load_state_dict(best_model1_weights)
                    if best_model2_weights is not None:
                        model2.load_state_dict(best_model2_weights)
                    if copula is not None and best_copula_theta is not None:
                        copula.theta = best_copula_theta
                    return model1, model2, copula, best_val_loss
                
                val_loss += batch_val_loss

            val_loss /= len(val_loader)
            
            #scheduler.step(val_loss)

            if verbose and epoch % 10 == 0:
                copula_theta = copula.theta if copula is not None else None
                print(f"Epoch {epoch}, Validation Loss: {val_loss} - Copula Theta: {copula_theta}")

            if not math.isnan(val_loss) and val_loss < best_val_loss:
                # Update best model weights if current val_loss is the best
                best_model1_weights = model1.state_dict()
                best_model2_weights = model2.state_dict()
                best_copula_theta = copula.theta.detach().clone() if copula is not None else None
                best_val_loss = val_loss
                stop_itr = 0
            else:
                stop_itr += 1
                if stop_itr == patience:
                    break

    if copula is not None:
        copula.theta = best_copula_theta
    model1.load_state_dict(best_model1_weights)
    model2.load_state_dict(best_model2_weights)

    return model1, model2, copula, best_val_loss

