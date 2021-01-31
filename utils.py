import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def np_to_torch(X: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(X).to(DEVICE).type(DTYPE)


def torch_to_np(X: torch.Tensor) -> np.ndarray:
    return X.detach().cpu().numpy()