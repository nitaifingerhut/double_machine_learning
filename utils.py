import numpy as np
import torch

from typing import Tuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def np_to_torch(
    X: np.ndarray
) -> torch.Tensor:
    return torch.from_numpy(X).to(DEVICE).type(DTYPE)


def torch_to_np(
    X: torch.Tensor
) -> np.ndarray:
    return X.detach().cpu().numpy()


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
        
    return theta_hat, v2