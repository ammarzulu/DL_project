import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# 2a) Feed-Forward Regression Model (same as your first code)
def build_ff_model():
    return torch.nn.Sequential(
        torch.nn.Linear(11, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 1)
    )

# 2b) RealNVP Normalizing Flow Components
class ConditionalCouplingLayer(nn.Module):
    """
    A RealNVP coupling layer that transforms the 1D target y conditioned on x.
    It learns scale s(x) and translation t(x) functions.
    """
    def __init__(self, x_dim, hidden_dim):
        super().__init__()
        self.net_scale = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.net_trans = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, y, x):
        s = self.net_scale(x)
        t = self.net_trans(x)
        # Forward transformation: z = (y - t(x)) * exp(-s(x))
        z = (y - t) * torch.exp(-s)
        log_det = -s.squeeze(-1)  # log-determinant of Jacobian
        return z, log_det

    def inverse(self, z, x):
        s = self.net_scale(x)
        t = self.net_trans(x)
        # Inverse transformation: y = z * exp(s(x)) + t(x)
        y = z * torch.exp(s) + t
        return y

class RealNVP(nn.Module):
    def __init__(self, x_dim, hidden_dim=64, num_flows=2):
        super().__init__()
        self.prior = torch.distributions.Normal(0, 1)
        self.layers = nn.ModuleList([
            ConditionalCouplingLayer(x_dim, hidden_dim) for _ in range(num_flows)
        ])
    
    def forward(self, y, x):
        log_det_total = 0
        z = y
        for layer in self.layers:
            z, log_det = layer.forward(z, x)
            log_det_total += log_det
        return z, log_det_total
    
    def inverse(self, z, x):
        y = z
        for layer in reversed(self.layers):
            y = layer.inverse(y, x)
        return y
    
    def log_prob(self, y, x):
        z, log_det = self.forward(y, x)
        log_p_z = self.prior.log_prob(z)
        return log_p_z + log_det