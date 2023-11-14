"""
Platt scaling and online versions of Platt scaling.
"""

import copy

import pickle as pkl
import numpy as np
import tqdm
import cvxpy as cp
from sklearn.exceptions import ConvergenceWarning

from parity_calibration.common_utils import (
    logit,
    logistic_transform,
)


def fit_platt_scaling_parameters(scores, labels, regularization=False):
    """Fits the parameters of a Platt scaling model.
    This model is used to calibrate the scores of a binary classifier.

    Args:
        scores: The logit probabilities of the classifier.
        labels: The binary labels.
        regularization: Whether to apply regularization.

    Returns:
        3-tuple of
        - calibrated probabilities
        - parameters A of the Platt scaling model
        - parameters B of the Platt scaling model
        - function which applies the fitted Platt scaling model
    """

    if regularization:
        # compute proportions in the calibration set in case regularization is applied
        N_1 = (labels == 1).sum()
        N_0 = (labels == 0).sum()
        t = (labels == 1).astype(int) * (N_1 + 1) / (N_1 + 2) + (labels == 0).astype(
            int
        ) * 1.0 / (N_0 + 2)
    else:
        # just use raw labels
        t = np.copy(labels)
    A = cp.Variable(1)
    B = cp.Variable(1)
    # form an objective for maximization
    # objective corresponds to the log-likelihood of the score 1/(1+exp(-(Ax+b)))
    # given the label y
    s = A * scores + B
    objective = t @ (s) - np.ones_like(t) @ cp.logistic(s)  # = (y * s) - log(1 + e^s)
    # solve the problem
    problem = cp.Problem(cp.Maximize(objective))
    problem.solve(solver=cp.ECOS)

    if "optimal" in problem.status:
        calibrated_probs = logistic_transform(A.value * scores + B.value)

        def platt_scaler(in_scores):
            out_scores = logistic_transform(A.value * in_scores + B.value)
            return out_scores

        return calibrated_probs, A.value, B.value, platt_scaler
    else:
        raise ConvergenceWarning("CVXPY hasn't converged")


def increasing_window_platt_scaling(
    te_preds,
    te_ys,
    val_preds,
    val_ys,
    update_frequency,
    update_preds_online=False,
    verbose=False,
):
    """Increasing window Platt scaling.

    Args:
        te_preds: predicted probabilities on the test set
        te_ys: true binary labels on the test set
        val_preds: predicted probabilities on the validation set
        val_ys: true binary labels on the validation set
        update_frequency: how often to update the Platt scaling parameters:
            if this is N, with [: current_time] data, fit platt scaler,
            then recalibrate the next N datapoints,
            then use [: current_time + N] data to fit platt scaler,
            then recalibrate next N, etc...
        update_preds_online: whether to rewrite historical predictions with
            recalibrated ones
        verbose: whether to print out information about the calibration process

    Returns:
        calibrated probabilities on the test set
    """

    assert all(
        [len(x.shape) == 1 for x in [te_preds, te_ys, val_preds, val_ys]]
    ), "all inputs must be flat"
    assert te_preds.shape == te_ys.shape
    assert val_preds.shape == val_ys.shape
    num_te = te_preds.shape[0]
    num_val = val_preds.shape[0]

    # starting indices
    N_calib_points = np.arange(num_val, num_val + num_te, update_frequency).astype(int)

    pred_probs = np.concatenate([val_preds, te_preds])
    running_recal_probs = copy.deepcopy(pred_probs)
    ys = np.concatenate([val_ys, te_ys])

    pred_probs_adaptive_platt = []
    for n_calib_points in tqdm.tqdm(N_calib_points):
        platt_train_range = np.arange(n_calib_points)
        platt_test_range = np.arange(
            n_calib_points,
            np.min([num_val + num_te, n_calib_points + update_frequency]),
        )
        if verbose:
            print(f"train with {platt_train_range[0]}:{platt_train_range[-1]}")
            print(f"test on {platt_test_range[0]}:{platt_test_range[-1]}")
            print("=" * 80)
        try:
            np.testing.assert_equal(
                running_recal_probs[platt_test_range], pred_probs[platt_test_range]
            )
        except:
            raise RuntimeError("running recal probs not equal to pred probs")
        if update_preds_online:
            _, platt_a, platt_b, platt_scaler = fit_platt_scaling_parameters(
                logit(running_recal_probs[platt_train_range]), ys[platt_train_range]
            )
            curr_preds = running_recal_probs[platt_test_range]
        else:
            _, platt_a, platt_b, platt_scaler = fit_platt_scaling_parameters(
                logit(pred_probs[platt_train_range]), ys[platt_train_range]
            )
            curr_preds = pred_probs[platt_test_range]
        np.testing.assert_equal(
            pred_probs[platt_test_range], running_recal_probs[platt_test_range]
        )

        recal_curr_preds = platt_scaler(logit(curr_preds))

        pred_probs_adaptive_platt.append(recal_curr_preds)
        running_recal_probs[platt_test_range] = recal_curr_preds

        np.testing.assert_equal(running_recal_probs[:num_val], pred_probs[:num_val])
        np.testing.assert_almost_equal(
            running_recal_probs[platt_test_range], recal_curr_preds
        )
        np.testing.assert_equal(
            running_recal_probs[platt_test_range[-1] + 1 :],
            pred_probs[platt_test_range[-1] + 1 :],
        )

    pred_probs_adaptive_platt = np.concatenate(pred_probs_adaptive_platt)

    return pred_probs_adaptive_platt


def moving_window_platt_scaling(
    te_preds,
    te_ys,
    val_preds,
    val_ys,
    window_size,
    update_frequency,
    update_preds_online=False,
    verbose=False,
):
    """Moving window Platt scaling.

    Args:
        te_preds: predicted probabilities on the test set
        te_ys: true binary labels on the test set
        val_preds: predicted probabilities on the validation set
        val_ys: true binary labels on the validation set
        window_size: size of the recent history window to use for Platt scaling
        update_frequency: how often to update the Platt scaling parameters:
            if this is N, with [current_time - window_size: current_time] data,
            fit platt scaler, then recalibrate the next N datapoints,
            then use [current_time + N - window_size: current_time + N] data to fit
            platt scaler, then recalibrate next N, etc...
        update_preds_online: whether to rewrite historical predictions with
            recalibrated ones
        verbose: whether to print out information about the calibration process

    Returns:
        calibrated probabilities on the test set
    """
    assert all(
        [len(x.shape) == 1 for x in [te_preds, te_ys, val_preds, val_ys]]
    ), "all inputs must be flat"
    assert te_preds.shape == te_ys.shape
    assert val_preds.shape == val_ys.shape
    num_te = te_preds.shape[0]
    num_val = val_preds.shape[0]

    # starting indices
    N_calib_points = np.arange(num_val, num_val + num_te, update_frequency).astype(int)

    pred_probs = np.concatenate([val_preds, te_preds])
    running_recal_probs = copy.deepcopy(pred_probs)
    ys = np.concatenate([val_ys, te_ys])

    pred_probs_adaptive_platt = []
    # print(N_calib_points)
    for n_calib_points in tqdm.tqdm(N_calib_points):
        platt_train_range = np.arange(n_calib_points - window_size, n_calib_points)
        platt_test_range = np.arange(
            n_calib_points,
            np.min([num_val + num_te, n_calib_points + update_frequency]),
        )
        if verbose:
            print(f"train with {platt_train_range[0]}:{platt_train_range[-1]}")
            print(f"test on {platt_test_range[0]}:{platt_test_range[-1]}")
            print("=" * 80)

        np.testing.assert_equal(
            running_recal_probs[platt_test_range], pred_probs[platt_test_range]
        )
        if update_preds_online:
            _, platt_a, platt_b, platt_scaler = fit_platt_scaling_parameters(
                logit(running_recal_probs[platt_train_range]), ys[platt_train_range]
            )
            curr_preds = running_recal_probs[platt_test_range]
        else:
            _, platt_a, platt_b, platt_scaler = fit_platt_scaling_parameters(
                logit(pred_probs[platt_train_range]), ys[platt_train_range]
            )
            curr_preds = pred_probs[platt_test_range]
        recal_curr_preds = platt_scaler(logit(curr_preds))
        pred_probs_adaptive_platt.append(recal_curr_preds)
        running_recal_probs[platt_test_range] = recal_curr_preds

    pred_probs_adaptive_platt = np.concatenate(pred_probs_adaptive_platt)

    return pred_probs_adaptive_platt
