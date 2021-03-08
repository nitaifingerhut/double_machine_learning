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
        layers.append(nn.Linear(dims[-1], 2))
        layers.append(ACTIVATIONS[activation_type](**activation_params))
        self.net = nn.Sequential(*layers)

    def forward(self, x, d):
        """
        Feed forward the network with (x,d).
        :param x: An input tensor of size (batch_size,num_features).
        :param d: An input tensor of size (batch_size,1).
        :return: The network prediction.
        """
        x = torch.cat((x,d.unsqueeze(1)), axis=1)
        pred = self.net(x)
        return pred[:, 0], pred[:, 1]


class CoupledMachineLearning(BaseEstimator):

    def __init__(
            self,
            true_model,
            num_features: int,
            **kwargs
    ):
        """
        :param true_model: true data model.
        :param num_features: Number of features in X.
        :param theta: ground truth of the policy coefficient theta.
        :param net_params: params for the neural network.
        :param reg_lambda: regularization factor to control the bias-variance ratio.

        """
        self.true_model = true_model
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

    def predict(self, x: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts (m, l) for a given input (x, d).
        :param x: a tensor of shape (num_samples,num_features).
        :param d: a tensor of shape (num_samples,1).
        :return: predictions as a tensor of size (num_samples,).
        """
        with torch.no_grad():
            m_pred, l_pred = self.net(x, d)
        return m_pred, l_pred

    def fit(self, x: torch.Tensor, d: torch.Tensor, y: torch.Tensor,
            batch_size: int = 32, max_epochs: int = 10, reg_labmda: float = 1.):
        """
        Fit the model to the data.
        :param x: a numpy 2d array of shape (num_samples,num_features).
        :param d: a numpy 2d array of shape (num_samples,).
        :param y: a numpy 1d array of shape (num_samples,).
        :param batch_size: batch size to use.
        :param max_epochs: max epochs to train.
        :param reg_labmda: regularization scale.
        """
        dataset = TensorDataset(x, d, y)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        mse_loss = torch.nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999))
        
        for epoch in range(max_epochs):
            for i, data in enumerate(dataloader, 0):

                optimizer.zero_grad()
                xb, db, yb = data
                m_pred, l_pred = self.net(xb, db)
                
                dm = db - m_pred
                dl = yb - l_pred

                dat_loss = mse_loss(m_pred, db) + mse_loss(l_pred, yb)
                mix_loss = torch.abs(torch.mean(dm * dl))
                loss = dat_loss + reg_labmda * mix_loss
   
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            xb, db, yb = next(iter(dataloader))
            m_pred, l_pred = self.net(xb, db)
            dm = db - m_pred
            dl = yb - l_pred
        
        return self, dm.mean().item(), dl.mean().item()
