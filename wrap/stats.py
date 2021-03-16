import torch

from data.funcs import g0, m0
from typing import Tuple


def est_theta(
    y: torch.Tensor, d: torch.Tensor, m_hat: torch.Tensor, l_hat: torch.Tensor
) -> Tuple[float, float]:
    # Compute V_hat
    v_hat = d - m_hat

    # Mean squared V-hat
    v2 = torch.mean(v_hat * v_hat)

    # Estimate theta
    theta_hat = torch.mean(v_hat * (y - l_hat)) / v2

    return theta_hat.item(), v2.item()


def exp_stats(
    x: torch.Tensor, m_hat: torch.Tensor, l_hat: torch.Tensor, theta: float, lamb: float
) -> Tuple[float, float, float, float]:

    # Computing residuals
    true_m = m0(x, lamb)
    true_l = g0(x) + theta * true_m

    dm = true_m - m_hat
    dl = true_l - l_hat

    # Evaluate the estimation errors
    dm_2 = torch.mean(dm ** 2)
    dl_2 = torch.mean(dl ** 2)
    dm_dl = torch.mean(dm * dl)

    # Bias
    bias = torch.mean(dm_dl - theta * dm_2)

    return dm_2.item(), dl_2.item(), dm_dl.item(), bias.item()
