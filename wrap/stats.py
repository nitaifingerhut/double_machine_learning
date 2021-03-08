import torch

from data.funcs import g0, m0
from typing import Tuple


def est_theta(
        Y: torch.Tensor,
        D: torch.Tensor,
        m_hat: torch.Tensor,
        l_hat: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute V_hat
    V_hat = (D - m_hat)

    # Mean squared V-hat
    v2 = torch.mean(V_hat * V_hat)

    # Estimate theta
    theta_hat = torch.mean(V_hat * (Y - l_hat)) / v2

    return theta_hat.item(), v2.item()


def exp_stats(
        Y: torch.Tensor,
        D: torch.Tensor,
        X: torch.Tensor,
        m_hat: torch.Tensor,
        l_hat: torch.Tensor, true_model
) -> Tuple[float, float, float, float, float]:

    # Computing residuals
    m = m0(X, true_model.lamb)
    l = g0(X) + true_model.theta * m

    dm = m - m_hat
    dl = l - l_hat

    # Evaluate the estimation errors
    dm_2 = torch.mean(dm ** 2)
    dl_2 = torch.mean(dl ** 2)
    dm_dl = torch.mean(dm * dl)

    # Bias
    bias = torch.mean(dm_dl - true_model.theta * dm_2)

    return dm_2.item(), dl_2.item(), dm_dl.item(), bias.item()
