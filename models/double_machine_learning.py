import torch
import torch.optim as optim

from models.mlp import MLP
from sklearn.base import BaseEstimator
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple
from wrap.utils import torch_


class DoubleMachineLearning(BaseEstimator):

    def __init__(
            self,
            num_features: int,
            **kwargs
    ):
        """
        :param num_features: Number of features in X.
        :param kwargs: params for the neural network.
        """
        self.net = torch_(MLP(num_features=num_features, out_features=1, **kwargs))

    def train(self):
        """
        Switch the model to train mode.
        """
        self.net.train()

    def eval(self):
        """
        Switch the model to eval mode.
        """
        self.net.eval()

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Predict response (y) for a given input (x).
        :param x: a tensor of shape (num_samples,num_features).
        :return: predictions as a tensor of size (num_samples,).
        """
        with torch.no_grad():
            pred = self.net(x)
        return pred

    def fit(self, x: torch.Tensor, y: torch.Tensor,
            batch_size: int = 32, max_epochs: int = 10):
        """
        Fit the model to the data.
        :param x: a tensor of shape (num_samples,num_features).
        :param y: a tensor of shape (num_samples,).
        :param batch_size: batch size to use.
        :param max_epochs: max epochs to train.
        """
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        mse_loss = torch.nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999))

        for epoch in range(max_epochs):
            for i, data in enumerate(dataloader, 0):

                optimizer.zero_grad()
                xb, yb = data
                pred = self.net(xb)
                loss = mse_loss(yb, pred.squeeze())
                loss.backward()
                optimizer.step()

        return self
