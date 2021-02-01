from double_machine_learning.base_mlp import BaseMLPEstimator
from double_machine_learning.double_mlp import DoubleMLPEstimator


def exp_stats(m_hat: torch.Tensor, l_hat: torch.Tensor, true_model):
    with torch.no_grad():
        # Compute V_hat
        V_hat = (D[idx_2] - m_hat)

        # Mean squared V-hat
        v2 = torch.mean(V_hat * V_hat)

        # Estimate theta
        theta_hat = torch.mean(V_hat * (Y[idx_2] - l_hat)) / v2

        # Computing residuals

        m = true_model.m0(X[idx_2])
        l = true_model.g0(X[idx_2]) + true_model.theta * m

        dm = m - m_hat
        dl = l - l_hat

        # Evaluate the estimation errors
        dm_dl = np.mean(dm * dl)
        dm_2 = np.mean(dm ** 2)

        # Bias
        bias = np.mean(dm_dl - true_model.theta * dm_2)

    return theta_hat, dm_dl, dm_2, bias, v2


def base_dml(Y: torch.Tensor, D: torch.Tensor, X: torch.Tensor, true_model):


    bbox = BaseMLPEstimator(X.shape[1],
                        hidden_dims=(32, 32, 32),
                        activation_params=dict(negative_slope=0.1))

    # Split the data to two parts
    num_samples = len(Y)
    mid_sample = num_samples // 2
    indices = torch.randperm(num_samples)

    idx_1, idx_2 = indices[:mid_sample], indices[mid_sample:]

    # Estimate l_hat
    bbox.fit(X[idx_1], Y[idx_1].flatten())
    l_hat = bbox.predict(X[idx_2])

    # Estimate m_hat
    bbox.fit(X[idx_1], D[idx_1].flatten())
    m_hat = bbox.predict(X[idx_2])

    return exp_stats(m_hat, l_hat, true_model)


def prop_dml(Y: torch.Tensor, D: torch.Tensor, X: torch.Tensor, true_model):
    bbox = DoubleMLPEstimator(true_model, X.shape[1]+1,
                         hidden_dims=(32, 32, 32),
                         activation_params=dict(negative_slope=0.1))

    # Split the data to two parts
    num_samples = len(Y)
    mid_sample = num_samples // 2
    indices = torch.randperm(num_samples)

    idx_1, idx_2 = indices[:mid_sample], indices[mid_sample:]

    # Estimate l_hat
    bbox.fit(X[idx_1], D[idx_1], Y[idx_1])
    m_hat, l_hat = bbox.predict(X[idx_2])

    return exp_stats(m_hat, l_hat, true_model)