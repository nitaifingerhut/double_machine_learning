import numpy as np
import torch

from data.models import CovarianceAR1, MajorityMinority, MultivariateGaussian
from data.funcs import g0, m0
from wrap.utils import np_to_torch


MODELS = dict(
    ar=CovarianceAR1,
    mg=MultivariateGaussian,
    mm=MajorityMinority,
)


class Data(object):

    def __init__(self, p: int, rho: float, theta: float,
                 lamb: float, model: str = 'ar'):
        """
        :param p: number of features in each example.
        :param rho: tunable parameter.
        :param theta: true value of theta.
        :param lamb: tunable parameter.
        :param model: data model to use.
                options:
                    'ar';auto-regression (default),
                    'mm';majority-minority,
                    'mg'; multivariate Gaussian.

        :raise KeyError is model is not one of options.
        """
        self.p = p
        self.rho = rho
        self.theta = theta
        self.lamb = lamb
        self.mu, self.sigma = MODELS[model](self.rho)(p)

    def generate(self, n: int):
        """
        Return n samples of data.
        :param n: number of samples to generate.
        :return: a tuple of tensors of the double machine learning model.
        """
        with torch.no_grad():
            mvn = np.random.multivariate_normal(self.mu, self.sigma, n)
            x = np_to_torch(mvn)
            d = m0(x, self.lamb) + 0.1 * np_to_torch(np.random.randn(n, ))
            y = d * self.theta + g0(x) + 0.1 * np_to_torch(np.random.randn(n, ))
        return y, d, x
