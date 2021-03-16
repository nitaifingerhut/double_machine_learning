import numpy as np
import torch


from typing import Dict, Tuple


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_(x):
    return x.to(DEVICE).type(torch.float)


def np_to_torch(x: np.ndarray) -> torch.Tensor:
    return torch_(torch.from_numpy(x))


def torch_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def train_test_split(
    y: torch.Tensor, d: torch.Tensor, x: torch.Tensor
) -> Tuple[Dict, Dict]:

    num_samples = len(y)
    mid_sample = num_samples // 2
    indices = torch.randperm(num_samples)
    idx_1, idx_2 = indices[:mid_sample], indices[mid_sample:]

    train = {"y": y[idx_1], "d": d[idx_1], "x": x[idx_1]}
    test = {"y": y[idx_2], "d": d[idx_2], "x": x[idx_2]}

    return train, test
