import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.base import BaseEstimator
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple
from wrap.utils import torch_


ACTIVATIONS = {
    'relu': nn.ReLU,
    'lrelu': nn.LeakyReLU,
    'sigmoid': nn.Sigmoid
}


class Net(nn.Module):
    def __init__(
            self,
            num_features: int,
            hidden_dims: Tuple,
            activation_type: str = "lrelu",
            activation_params: dict = dict(),
            dropout: float = 0.0,
            batchnorm: bool = True
    ):
        """
        :param num_features: Number of features in X.
        :param hidden_dims: List containing hidden dimensions of each Linear layer for the main path.
        :param activation_type: Type of activation function; supports either 'relu', 'lrelu', 'sigmoid'.
        :param activation_params: Parameters passed to activation function.
        :param dropout: Amount (p) of Dropout to apply between convolutions. Zero means don't apply dropout.
        """
        super(Net, self).__init__()

        if activation_type not in ACTIVATIONS:
            raise ValueError('Unsupported activation type')

        assert 0 <= dropout < 1

        dims = (num_features,) + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if dropout > 0:
                layers.append(nn.Dropout(dropout, inplace=True))
            layers.append(ACTIVATIONS[activation_type](**activation_params))
        layers.append(nn.Linear(dims[-1], 1))
        layers.append(ACTIVATIONS[activation_type](**activation_params))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Feed forward the network with x.
        :param x: An input tensor.
        :return: The network prediction.
        """
        return self.net(x)


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
        self.net = torch_(Net(num_features, **kwargs))

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
