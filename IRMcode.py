"""
This code is highly based on code from http://ntraft.com/exploring-invariant-risk-minimization-in-pytorch/
which is an expanded version of code from the paper Invariant Risk Minimization (https://arxiv.org/abs/1907.02893)
Code from the paper can be found here: https://github.com/facebookresearch/InvariantRiskMinimization
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim

def example_1(n=10000, dim=1, A=0):
    x1 = A + torch.randn(n, dim)
    y = x1 + torch.randn(n, dim)
    x2 = y + A + torch.randn(n, dim)
    return torch.cat((x1, x2), 1), y 

def example_2(n=10000, dim=1, sigma=1):
    x1 = torch.randn(n, dim)*sigma
    x2 = x1 + torch.randn(n, dim)
    y = x1 + torch.randn(n, dim)*sigma
    x3 = y + torch.randn(n, dim)
    return torch.cat((x1, x2, x3), 1), y 

def example_3(n=10000, dim=1, sigma=1):
    h = torch.randn(n, dim)
    x1 = h + torch.randn(n, dim)
    y = x1 + 2*h + torch.randn(n, dim) + torch.randn(n, dim)*sigma
    x2 = y + h + torch.randn(n, dim)
    return torch.cat((x1, x2), 1), y 

# Sample training data with one low variance and one high variance.
train_A = (0,1)
train_sigma = (1,0.1)
train_sigma3 = (0,1)
#train_environments = [example_1(A=a) for a in train_A]
train_environments = [example_3(sigma=s) for s in train_sigma3]

# Sample testing data with the same sigmas, but also more in between.
test_A = (2,3,4,5)
test_sigma = (0.01, 2, 3, 4)
#test_environments = [example_1(A=a) for a in test_A]
test_environments = [example_3(sigma=s) for s in test_sigma]

from torch import autograd
def compute_penalty(losses, dummy_w):
    g1 = autograd.grad(losses[0::2].mean(), dummy_w, create_graph=True)[0]
    g2 = autograd.grad(losses[1::2].mean(), dummy_w, create_graph=True)[0]
    return (g1 * g2).sum()

def print_epoch_info(epoch_idx, num_epochs, phi, avg_error, penalty, collate_prints=True):
    params = phi.detach().cpu().numpy()
    with np.printoptions(precision=3):
        # Preserve the first iteration's print, otherwise overwrite.
        if epoch_idx == 0:
            prefix = "Initial performance:\n"
            suffix = "\n"
        elif collate_prints:
            prefix = "\r"
            suffix = ""
        else:
            prefix = ""
            suffix = "\n"
        ndigits = int(np.log10(num_epochs)) + 1
        print(prefix + f"Epoch {epoch_idx+1:{ndigits}d}/{num_epochs} ({epoch_idx/num_epochs:3.0%});"
              f" params = {params.transpose()[0]};" # Print on one line.
              f" MSE = {avg_error:.4f};"
              f" penalty = {penalty:.9f}", end=suffix)


MSE_LOSS = torch.nn.MSELoss(reduction="none") #reduction="none": returns vector of squared error

def avg_loss_per_env(model, environments, loss=MSE_LOSS):
    return np.array([float(loss(model(x_e), y_e).mean()) for x_e, y_e in environments])

def eval_dataset(model, dataset_name, dataset_envs, loss=MSE_LOSS):
    env_errors = avg_loss_per_env(model, dataset_envs, loss)
    print(f"{dataset_name} errors:")
    for i, err in enumerate(env_errors):
        print(f"Environment {i+1} Error = {err}")
    print(f"Overall Average = {env_errors.mean():.4f}")

def eval_all(model, loss=MSE_LOSS):
    for name, envs in (("Train", train_environments), ("Test", test_environments)):
        eval_dataset(model, name, envs, loss)


def train(use_IRM=True, num_epochs=50000, epochs_per_eval=1000, collate_prints=True):

    # Model: y = x.dot(phi) * dummy_w
    phi = torch.nn.Parameter(torch.ones(train_environments[0][0].shape[1], 1) * 1.0)
    dummy_w = torch.nn.Parameter(torch.Tensor([1.0]))

    def model(input):
        return input @ phi * dummy_w

    # We will only learn phi.
    opt = torch.optim.SGD([phi], lr=1e-4) #lr adjustable, used 1e-4
    mse = torch.nn.MSELoss(reduction="none")

    for i in range(num_epochs):
        # Sum of average error in each environment.
        error = 0
        # Total IRM penalty across all environments.
        penalty = 0

        # Forward pass and loss computation.
        for x_e, y_e in train_environments:
            p = torch.randperm(len(x_e))
            error_e = mse(model(x_e[p]), y_e[p])
            penalty += compute_penalty(error_e, dummy_w)
            error += error_e.mean()
        
        # Print losses.
        if i % epochs_per_eval == 0:
            print_epoch_info(i,
                             num_epochs,
                             phi,
                             error / len(train_environments),
                             penalty,
                             collate_prints)

        # Backward pass.
        opt.zero_grad()
        if use_IRM:
            (1e-5 * error + penalty).backward()
        else:
            error.backward()
        opt.step()

        # Do a sanity check.
        if phi.isnan().any():
            print_epoch_info(i,
                             num_epochs,
                             phi,
                             error / len(train_environments),
                             penalty,
                             collate_prints)
            print("ERROR: Optimization diverged and became NaN. Halting training.")
            return model, phi
    
    params = phi.detach().cpu().numpy()
    print("\n\nFinal model:")
    print(f"params = {params.transpose()[0]}")
    eval_all(model, mse)
    return model, phi

irm_model, irm_params = train()
#standard_model, standard_params = train(use_IRM=False)
