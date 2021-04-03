import torch


def g0(x: torch.Tensor) -> torch.Tensor:
    """
    :param x: a tensor of size (num_samples,num_features).
    :return: the actual value of g0(X).
    """
    # out = -x[:, 0] ** 2 + 2 * torch.log(0.1 + x[:, 1] ** 2)
    out = 1 - x[:, 0] + x[:, 1] ** 2 - x[:, 2] ** 3 + x[:, 3] ** 4 - x[:, 4] ** 5 + x[:, 5] ** 6 + torch.exp(x[:, 15])
    return out


def m0(x: torch.Tensor, lamb: float) -> torch.Tensor:
    """
    :param x: a tensor of size (num_samples,num_features).
    :param lamb: scale parameter.
    :return: the actual value of m0(X).
    """
    # out = lamb * torch.relu(x[:, 1])
    out = 1 - x[:, 3] + x[:, 4] ** 2 - x[:, 5] ** 3 + x[:, 6] ** 4 - x[:, 7] ** 5 + x[:, 8] ** 6 + lamb * torch.relu(torch.sin(x[:, 15]))
    return out
