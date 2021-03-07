import torch

from data.funcs import g0, m0
from models.base_mlp import BaseMLPEstimator
from models.double_mlp import DoubleMLPEstimator
from typing import Tuple
from wrap.utils import est_theta


def exp_stats(
    Y: torch.Tensor, 
    D: torch.Tensor, 
    X: torch.Tensor,
    m_hat: torch.Tensor, 
    l_hat: torch.Tensor, true_model
) -> Tuple[float, float, float, float, float]:
    
    with torch.no_grad():
        
        # Estimate theta
        theta_hat, v2 = est_theta(Y, D, m_hat, l_hat)
    
        # Computing residuals
        m = m0(X, true_model.lamb)
        l = g0(X) + true_model.theta * m

        dm = m - m_hat
        dl = l - l_hat

        # Evaluate the estimation errors
        dm_dl = torch.mean(dm * dl)
        dm_2 = torch.mean(dm ** 2)

        # Bias
        bias = torch.mean(dm_dl - true_model.theta * dm_2)

    return theta_hat.item(), dm_dl.item(), dm_2.item(), bias.item(), v2.item()


def base_dml(
    Y: torch.Tensor, 
    D: torch.Tensor, 
    X: torch.Tensor, 
    true_model
) -> Tuple[float, float, float, float, float]:

    bbox = BaseMLPEstimator(X.shape[1],
                        hidden_dims=(32, 32, 32),
                        activation_params=dict(negative_slope=0.1))

    # Split the data to two parts
    num_samples = len(Y)
    mid_sample = num_samples // 2
    indices = torch.randperm(num_samples)

    idx_1, idx_2 = indices[:mid_sample], indices[mid_sample:]

    # Estimate l_hat
    bbox.fit(X[idx_1], Y[idx_1].flatten())
    l_hat = bbox.predict(X[idx_2])

    # Estimate m_hat
    bbox.fit(X[idx_1], D[idx_1].flatten())
    m_hat = bbox.predict(X[idx_2])

    return exp_stats(Y[idx_2], D[idx_2], X[idx_2], m_hat, l_hat, true_model)


def prop_dml(
    Y: torch.Tensor, 
    D: torch.Tensor, 
    X: torch.Tensor, 
    true_model,
    reg_labmda: float = 1.
) -> Tuple[float, float, float, float, float]:
    bbox = DoubleMLPEstimator(true_model, X.shape[1]+1,
                         hidden_dims=(32, 32, 32),
                         activation_params=dict(negative_slope=0.1))

    # Split the data to two parts
    num_samples = len(Y)
    mid_sample = num_samples // 2
    indices = torch.randperm(num_samples)

    idx_1, idx_2 = indices[:mid_sample], indices[mid_sample:]

    # Estimate l_hat
    bbox, _, _ = bbox.fit(X[idx_1], D[idx_1], Y[idx_1], reg_labmda=reg_labmda)
    m_hat, l_hat = bbox.predict(X[idx_2], D[idx_2])

    return exp_stats(Y[idx_2], D[idx_2], X[idx_2], m_hat, l_hat, true_model)


def prop_dml_internal_stats(
    Y: torch.Tensor, 
    D: torch.Tensor, 
    X: torch.Tensor, 
    true_model,
    reg_labmda: float = 1.
) -> Tuple[float, float, float, float, float]:
    bbox = DoubleMLPEstimator(true_model, X.shape[1]+1,
                         hidden_dims=(32, 32, 32),
                         activation_params=dict(negative_slope=0.1))

    # Split the data to two parts
    num_samples = len(Y)
    mid_sample = num_samples // 2
    indices = torch.randperm(num_samples)

    idx_1, idx_2 = indices[:mid_sample], indices[mid_sample:]

    # Estimate l_hat
    bbox, dm, dl = bbox.fit(X[idx_1], D[idx_1], Y[idx_1], reg_labmda=reg_labmda)
    m_hat, l_hat = bbox.predict(X[idx_2], D[idx_2])
    
    theta_hat, _, _, _, _ = exp_stats(Y[idx_2], D[idx_2], X[idx_2], m_hat, l_hat, true_model)
   
    return dm, dl, theta_hat