import torch


def g0(x: torch.Tensor) -> torch.Tensor:
    """
    :param x: a tensor of size (num_samples,num_features).
    :return: the actual value of g0(X).
    """
    out = -x[:, 0] ** 2 + 2 * torch.log(0.1 + x[:, 1] ** 2)
    return out


def m0(x: torch.Tensor, lamb: float) -> torch.Tensor:
    """
    :param x: a tensor of size (num_samples,num_features).
    :param lamb: scale parameter.
    :return: the actual value of m0(X).
    """
    out = lamb * torch.relu(x[:, 1])
    return out
