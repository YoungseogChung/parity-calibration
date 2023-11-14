"""
Evaluation metrics for parity probabilties: 
    parity calibration error, sharpness, binary accuracy, auroc.
"""

import numpy as np
from sklearn.metrics import roc_auc_score as auroc

from parity_calibration.parity_calibration_utils import (
    parity_calibration_plot,
    parity_ece,
)


""" Utility functions for computing metrics. """


def bin_points(scores, bin_edges):
    assert bin_edges is not None, "Bins have not been defined"
    scores = scores.squeeze()
    assert np.size(scores.shape) < 2, "scores should be a 1D vector or singleton"
    scores = np.reshape(scores, (scores.size, 1))
    bin_edges = np.reshape(bin_edges, (1, bin_edges.size))
    return np.sum(scores > bin_edges, axis=1)


def get_binned_probabilities_fixed_width(y, pred_prob, n_bins, pred_prob_base=None):
    assert n_bins >= 0
    bin_edges = np.linspace(1.0 / n_bins, 1.0, n_bins)
    pi_pred = np.zeros(n_bins)
    pi_base = np.zeros(n_bins)
    pi_true = np.zeros(n_bins)
    n_elem = np.zeros(n_bins)
    bin_assignment = bin_points(pred_prob, bin_edges)

    for i in range(n_bins):
        bin_idx = bin_assignment == i
        n_elem[i] = sum(bin_idx)
        if n_elem[i] == 0:
            continue
        pi_pred[i] = pred_prob[bin_idx].mean()
        if pred_prob_base is not None:
            pi_base[i] = pred_prob_base[bin_idx].mean()
        pi_true[i] = y[bin_idx].mean()

    assert sum(n_elem) == y.size

    return n_elem, pi_pred, pi_base, pi_true


""" Metrics. """


def parity_calibration_error(obs_fall, prob_fall, num_bins, display_plot=False):
    (
        bins_probs,
        obs_probs,
        obs_counts,
        mean_probs_per_bin,
        fig,
    ) = parity_calibration_plot(
        obs_fall=obs_fall, prob_fall=prob_fall, num_bins=num_bins, display=display_plot
    )

    calibration_error = parity_ece(
        bins_probs, obs_probs, obs_counts, mean_probs_per_bin
    )

    return calibration_error, fig


def bias_adjusted_sharpness(y, pred_prob, n_bins=15):
    n_elem, _, _, pi_true = get_binned_probabilities_fixed_width(y, pred_prob, n_bins)
    assert sum(n_elem) == y.size
    biased_estimate = (n_elem @ (pi_true**2)) / y.size
    correction = (
        n_elem @ np.divide(np.multiply(pi_true, 1 - pi_true), np.maximum(n_elem - 1, 1))
    ) / y.size
    return biased_estimate - correction  # np.sum(n_elem * (pi_true**2))/y.size


def get_all_binary_metrics(obs_fall, prob_fall, num_bins, discard_plot=False):
    assert len(prob_fall.shape) == 1
    assert len(obs_fall.shape) == 1
    assert prob_fall.shape == obs_fall.shape

    pce, fig = parity_calibration_error(
        obs_fall=obs_fall,
        prob_fall=prob_fall,
        num_bins=num_bins,
        display_plot=False,
    )
    if discard_plot:
        fig = None

    sharpness = bias_adjusted_sharpness(
        y=obs_fall, pred_prob=prob_fall, n_bins=num_bins
    )

    pred_class = (prob_fall >= 0.5).astype(int)
    assert pred_class.shape == obs_fall.shape
    binary_accuracy = np.mean(pred_class == obs_fall)
    auroc_score = auroc(obs_fall, prob_fall)

    metrics_dict = {
        "pce": pce,
        "parity_calibration_fig": fig,
        "sharpness": sharpness,
        "binary_acc": binary_accuracy,
        "auroc": auroc_score,
        "ys": obs_fall,
        "prob_preds": prob_fall,
        "num_bins": num_bins,
    }

    return metrics_dict
