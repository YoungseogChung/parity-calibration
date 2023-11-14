"""
Utility functions for parity calibration.
"""

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def get_parity_prob(mu_list, sigma_list, y_list, seq_ordering=None):
    """Get parity probability from Gaussian predictions.

    Args:
        mu_list: array of mean predictions, of shape (T,).
        sigma_list: array of standard deviation predictions, of shape (T,).
        y_list: array of ground truth continuous target values, of shape (T,).
        seq_ordering: array of indices to order the sequence, of shape (T,):
            if None, then the sequence is taken to be in order.

    Returns:
        2-tuple of
        - parity probability, of shape (T,).
        - parity observation, of shape (T,).
    """
    num_data = y_list.shape[0]
    assert mu_list.shape[0] == num_data
    assert sigma_list.shape[0] == num_data
    if seq_ordering is None:
        seq_ordering = np.arange(num_data)
    assert seq_ordering.shape[0] == num_data
    assert np.all(sigma_list.flatten() > 0)

    mu_ordered = mu_list[seq_ordering][1:]
    sigma_ordered = sigma_list[seq_ordering][1:]
    y_ordered = y_list[seq_ordering][1:]
    prev_y_ordered = y_list[seq_ordering][:-1]

    distr_ordered = norm(loc=mu_ordered, scale=sigma_ordered)
    prob_fall_batch = distr_ordered.cdf(prev_y_ordered)
    obs_fall_batch = y_ordered <= prev_y_ordered

    return prob_fall_batch, obs_fall_batch


def get_parity_labels(y_list, seq_ordering=None):
    """Get parity labels from ground truth continuous target values.

    Args:
        y_list: array of ground truth continuous target values, of shape (T,).
        seq_ordering: array of indices to order the sequence, of shape (T,):
            if None, then the sequence is taken to be in order.

    Returns:
        parity observation, of shape (T,).
    """
    num_data = y_list.shape[0]
    if seq_ordering is None:
        seq_ordering = np.arange(num_data)
    assert seq_ordering.shape[0] == num_data

    y_ordered = y_list[seq_ordering][1:]
    prev_y_ordered = y_list[seq_ordering][:-1]
    obs_fall_batch = y_ordered <= prev_y_ordered

    return obs_fall_batch


def parity_ece(bins_probs, obs_probs, obs_counts, mean_probs_per_bin):
    """Calculate parity ECE (parity calibration error).

    Args:
        bins_probs: the mean probability of each bin.
        obs_probs: the observed probabilities of the positive class per bin.
        obs_counts: number of predictions per bin.
        mean_probs_per_bin: mean of the predicted probabilities falling into each bin.

    Returns:
        the parity ECE metric:
            sum of L1 error between mean_predicted_probability and observed probability
            across each bin, weighted by the proportion of predictions falling
            into each bin.
    """
    num_positive_bins = np.sum(obs_counts > 0)
    props_per_bin = obs_counts / np.sum(obs_counts)
    np.testing.assert_almost_equal(np.sum(props_per_bin), 1.0)
    calibration_error = np.nansum(
        props_per_bin * np.abs(mean_probs_per_bin.flatten() - obs_probs.flatten())
    )

    return calibration_error


def parity_calibration_error_from_gaussian(
    mu_list,
    sigma_list,
    y_list,
    num_bins,
    seq_ordering=None,
    display_plot=False,
):
    """Calculate parity calibration error from Gaussian predictions.

    Args:
        mu_list: array of mean predictions, of shape (T,).
        sigma_list: array of standard deviation predictions, of shape (T,).
        y_list: array of ground truth continuous target values, of shape (T,).
        num_bins: number of bins to use for the parity calibration metrics.
        seq_ordering: array of indices to order the sequence, of shape (T,):
            if None, then the sequence is taken to be in order.
        display_plot: whether to display the parity calibration plot.
        props_weight: whether to weight the parity calibration error by the proportion

    Returns:
        2-tuple of
        - parity calibration error.
        - parity calibration plot.
    """
    pred_prob, obs_fall = get_parity_prob(
        mu_list=mu_list, sigma_list=sigma_list, y_list=y_list, seq_ordering=seq_ordering
    )
    (
        bins_probs,
        obs_probs,
        obs_counts,
        mean_probs_per_bin,
        fig,
    ) = parity_calibration_plot(
        obs_fall=obs_fall, prob_fall=pred_prob, num_bins=num_bins, display=display_plot
    )
    num_positive_bins = np.sum(obs_counts > 0)
    calibration_error_test = parity_ece(
        bins_probs, obs_probs, obs_counts, mean_probs_per_bin
    )
    props_per_bin = obs_counts / np.sum(obs_counts)
    calibration_error = np.nanmean(
        props_per_bin * np.abs(mean_probs_per_bin.flatten() - obs_probs.flatten())
    )
    np.testing.assert_equal(calibration_error_test, calibration_error)

    return calibration_error, fig


""" Plotting utils """


def parity_calibration_plot(obs_fall, prob_fall, num_bins, display=True, title=None):
    """Plot parity calibration plot.

    Args:
        obs_fall: array of observed fall events, of shape (T,).
        prob_fall: array of predicted probability of fall events, of shape (T,).
        num_bins: number of bins to use for the parity calibration metrics.
        display: whether to display the plot.
        title: title of the plot.

    Returns:
        5-tuple of
        - mean probability of each bin.
        - observed probability of the positive class per bin.
        - number of predictions per bin.
        - mean of the predicted probabilities falling into each bin.
        - parity calibration plot.
    """
    num_obs = obs_fall.shape[0]
    assert prob_fall.shape[0] == num_obs

    distr = np.linspace(0, 1, num_bins + 1)
    bins = [(distr[i], distr[i + 1]) for i in range(len(distr) - 1)]
    bins_probs = np.array([np.mean(x) for x in bins])

    obs_prob_ = []
    obs_count_ = []
    hist_obs_ = []
    mean_probs_per_bin_ = []
    for b in bins:
        l, u = b
        pt_idx = np.logical_and(l <= prob_fall, prob_fall < u)
        assert pt_idx.shape[0] == num_obs
        if np.sum(pt_idx.astype(int)) == 0:
            # empty bin
            empty_bin_obs_prob = np.nan  # TODO: best way to handle an empty bin?
            empty_bin_mean_pred_prob_in_bin = np.nan
            empty_bin_obs_count = 0
            empty_bin_hist_obs = np.array([])
            obs_prob_.append(empty_bin_obs_prob)
            obs_count_.append(empty_bin_obs_count)
            hist_obs_.append(empty_bin_hist_obs)
            mean_probs_per_bin_.append(empty_bin_mean_pred_prob_in_bin)
        else:
            assert all(l <= prob_fall[pt_idx])
            assert all(prob_fall[pt_idx] < u)
            obs_prob = np.mean(obs_fall[pt_idx])
            obs_prob_.append(obs_prob)
            obs_count_.append(np.sum(pt_idx.astype(int)))
            hist_obs_.append(prob_fall[pt_idx])
            mean_probs_per_bin_.append(np.mean(prob_fall[pt_idx]))
    obs_probs = np.array(obs_prob_)
    obs_counts = np.array(obs_count_)
    hist_obs = np.concatenate(hist_obs_)
    mean_probs_per_bin = np.array(mean_probs_per_bin_)

    # making the figure
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_probs_per_bin_, obs_probs, "-o")
    ax.plot([0, 1], [0, 1], c="k")
    ax2 = ax.twinx()

    ax2.hist(hist_obs, bins=distr, alpha=0.3)

    min_bin_count = np.min(obs_counts)

    ax2.set_ylabel("points per bin")
    ece_val = parity_ece(bins_probs, obs_probs, obs_counts, mean_probs_per_bin)
    if title:
        ax.set_title(f"{title} (ece {ece_val:.6f}, {min_bin_count} min count)")
    else:
        ax.set_title(f"(ece {ece_val:.6f}, {min_bin_count} min count)")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed probability")

    if display:
        plt.show()

    out = bins_probs, obs_probs, obs_counts, mean_probs_per_bin, fig

    return out
