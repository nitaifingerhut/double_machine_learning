import torch.nn as nn

from typing import Tuple


ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU, "sigmoid": nn.Sigmoid}


class MLP(nn.Module):
    def __init__(
        self,
        num_features: int,
        out_features: int,
        hidden_dims: Tuple = (32, 32, 32),
        activation_type: str = "lrelu",
        activation_params: dict = dict(inplace=True),
        dropout: float = 0.0,
    ):
        """
        :param num_features: #input features.
        :param num_features: #output features.
        :param hidden_dims: a tuple containing #features of each Linear layer..
        :param activation_type: type of activation function; supports either 'relu', 'lrelu', 'sigmoid'.
        :param activation_params: parameters passed to activation function.
        :param dropout: amount (p) of Dropout to apply between convolutions. zero means don't apply dropout.
        """
        super(MLP, self).__init__()

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        assert 0 <= dropout < 1

        dims = (num_features,) + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if dropout > 0:
                layers.append(nn.Dropout(dropout, inplace=True))
            layers.append(ACTIVATIONS[activation_type](**activation_params))
        layers.append(nn.Linear(dims[-1], out_features))
        layers.append(ACTIVATIONS[activation_type](**activation_params))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Feed forward the network with x.
        :param x: An input tensor.
        :return: The network prediction.
        """
        return self.net(x)
