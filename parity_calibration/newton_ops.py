"""
Online Platt scaling with Newton step.
"""

import tqdm
import numpy as np
import cvxpy as cp

from parity_calibration.common_utils import (
    sigmoid,
    logit,
)


def online_platt_scaling_newton(preds, y, beta=0.01, D=1):
    """Online Platt Scaling with Newton's Method

    Args:
        preds: sequence of logits of shape (T,)
        y: ground truth binary labels of shape (T,)
        beta: hyperparameter
        D: hyperparameter

    Returns:
        3-tuple of
        - online calibrated probabilities of shape (T,)
        - online_platt_a parameter used for online calibration, of shape (T,)
        - online_platt_b parameter used for online calibration, of shape (T,)
    """
    assert np.sum(y == 0) + np.sum(y == 1) == y.size
    assert y.size == preds.size

    def platt_gradient(x, a, b, y):
        if y == 1:
            gradient = (sigmoid(x * a + b) - 1) * np.array([x, 1])
        else:
            gradient = (sigmoid(x * a + b)) * np.array([x, 1])
        return gradient

    pred_probs_platt = np.zeros(preds.size)
    online_platt_a = np.zeros(preds.size)
    online_platt_b = np.zeros(preds.size)
    pred_probs_platt[0] = sigmoid(online_platt_a[0] * preds[0] + online_platt_b[0])
    norm_control = 1e2
    cum_hessian = (1 / (beta * D) ** 2) * np.eye(2)
    cum_hessian_inv = (beta * D) ** 2 * np.eye(2)
    for i in tqdm.tqdm(range(0, preds.size - 1)):
        g_i = platt_gradient(preds[i], online_platt_a[i], online_platt_b[i], y[i])
        cum_hessian_inv = cum_hessian_inv - np.outer(
            cum_hessian_inv @ g_i, g_i.T @ cum_hessian_inv
        ) / (1 + (g_i.T @ cum_hessian_inv @ g_i))
        cum_hessian = cum_hessian + np.outer(g_i, g_i)
        update = -(1 / beta) * (cum_hessian_inv @ g_i)
        online_platt_a[i + 1] = online_platt_a[i] + update[0]
        online_platt_b[i + 1] = online_platt_b[i] + update[1]
        if (
            online_platt_a[i + 1] ** 2 + online_platt_b[i + 1] ** 2
        ) > norm_control**2:
            x = cp.Variable(2)
            x.value = [online_platt_a[i + 1], online_platt_b[i + 1]]
            linear_factor = (
                -2
                * (cum_hessian / i)
                @ np.array([online_platt_a[i + 1], online_platt_b[i + 1]])
            )
            constraint = [cp.norm(x) <= norm_control]
            prob = cp.Problem(
                cp.Minimize(linear_factor.T @ x + cp.quad_form(x, (cum_hessian / i))),
                constraint,
            )
            try:
                prob.solve(warm_start=True)
                online_platt_a[i + 1] = x.value[0]
                online_platt_b[i + 1] = x.value[1]
            except:
                online_platt_a[i + 1] = online_platt_a[i]
                online_platt_b[i + 1] = online_platt_b[i]
        pred_probs_platt[i + 1] = sigmoid(
            online_platt_a[i + 1] * preds[i + 1] + online_platt_b[i + 1]
        )
    return pred_probs_platt, online_platt_a, online_platt_b
