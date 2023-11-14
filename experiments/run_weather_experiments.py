"""
Run all experiments for weather data.
"""

import sys
import os

import numpy as np
import pickle as pkl
import uncertainty_toolbox as uct

from parity_calibration.newton_ops import online_platt_scaling_newton
from parity_calibration.platt_scaling import (
    moving_window_platt_scaling,
    increasing_window_platt_scaling,
)
from parity_calibration.parity_calibration_utils import get_parity_prob
from parity_calibration.common_utils import logit
from parity_calibration.eval_metrics import get_all_binary_metrics


_DBG = False


def run_all_method_tests(
    run_suffix_removed,
    te_preds,
    te_ys,
    val_preds,
    val_ys,
    m_use_uf,
    m_use_ws,
    i_use_uf,
    use_ops_hp=None,
    _NUM_BINS=30,
    _METRICS_SAVE_DIR=None,
    run_prehoc=False,
    run_ops=False,
    run_increasing=False,
    run_moving=False,
):

    if _METRICS_SAVE_DIR is None:
        raise RuntimeError("Must specify where to save metrics")
    if not os.path.exists(_METRICS_SAVE_DIR):
        print(f"Making directory {_METRICS_SAVE_DIR}")
        os.makedirs(_METRICS_SAVE_DIR)
    print(f"Saving everything to {_METRICS_SAVE_DIR}")

    # 0) Prehoc
    if run_prehoc:
        print(f"Running TEST for prehoc")
        save_file_name = f"{_METRICS_SAVE_DIR}/{run_suffix_removed}-prehoc_metrics.pkl"
        if os.path.exists(save_file_name):
            print(f"Prehoc test already exists, skipping file {save_file_name}")
            prehoc_metrics = pkl.load(open(save_file_name, "rb"))
        else:
            prehoc_metrics = get_all_binary_metrics(te_ys, te_preds, num_bins=_NUM_BINS)
            print(f"DUMPING METRICS TO {save_file_name}")
            if not _DBG:
                pkl.dump(prehoc_metrics, open(save_file_name, "wb"))
    else:
        print("Chose not to run PREHOC")
    print("=" * 80)

    # 1) Moving Window Platt scaling
    if run_moving:
        print(
            f" Running TEST for moving Platt scaling with uf_{m_use_uf} and ws_{m_use_ws}"
        )
        save_file_name = (
            f"{_METRICS_SAVE_DIR}/{run_suffix_removed}-best_moving_metrics.pkl"
        )
        if os.path.exists(save_file_name):
            print(f"Moving test already exists, skipping file {save_file_name}")
            moving_recal_metrics = pkl.load(open(save_file_name, "rb"))
        else:
            moving_window_recal_out = moving_window_platt_scaling(
                te_preds=te_preds,
                te_ys=te_ys,
                val_preds=val_preds,
                val_ys=val_ys,
                window_size=m_use_ws,
                update_frequency=m_use_uf,
                update_preds_online=False,
            )
            moving_recal_metrics = get_all_binary_metrics(
                te_ys, moving_window_recal_out, num_bins=_NUM_BINS
            )
            print(f"DUMPING METRICS TO {save_file_name}")
            if not _DBG:
                pkl.dump(moving_recal_metrics, open(save_file_name, "wb"))
    else:
        print("Chose not to run MOVING")
    print("=" * 80)

    # 2) Increasing Window Platt scaling
    if run_increasing:
        save_file_name = (
            f"{_METRICS_SAVE_DIR}/{run_suffix_removed}-best_increasing_metrics.pkl"
        )
        if os.path.exists(save_file_name):
            print(f"Increasing test already exists, skipping file {save_file_name}")
            increasing_recal_metrics = pkl.load(open(save_file_name, "rb"))
        else:
            print(f" Running TEST for increasing Platt scaling with uf_{i_use_uf}")

            increasing_window_recal_out = increasing_window_platt_scaling(
                te_preds=te_preds,
                te_ys=te_ys,
                val_preds=val_preds,
                val_ys=val_ys,
                update_frequency=i_use_uf,
                update_preds_online=False,
            )

            increasing_recal_metrics = get_all_binary_metrics(
                te_ys, increasing_window_recal_out, num_bins=_NUM_BINS
            )
            print(f"DUMPING METRICS TO {save_file_name}")
            if not _DBG:
                pkl.dump(increasing_recal_metrics, open(save_file_name, "wb"))
    else:
        print("Chose not to run INCREASING")
    print("=" * 80)

    # 3) OPS
    if run_ops:
        # breakpoint()
        print(f"Running TEST for OPS with {use_ops_hp}")
        save_file_name = f"{_METRICS_SAVE_DIR}/{run_suffix_removed}-ops_metrics.pkl"
        if os.path.exists(save_file_name):
            print(f"OPS test already exists, skipping file {save_file_name}")
            ops_recal_metrics = pkl.load(open(save_file_name, "rb"))
        else:
            if use_ops_hp is not None and isinstance(use_ops_hp, tuple):
                use_beta, use_D = use_ops_hp
                assert isinstance(use_D, int)
                ops_out, ops_a, ops_b = online_platt_scaling_newton(
                    preds=logit(te_preds), y=te_ys, beta=use_beta, D=use_D
                )
            else:
                ops_out, ops_a, ops_b = online_platt_scaling_newton(
                    preds=logit(te_preds), y=te_ys
                )
            ops_recal_metrics = get_all_binary_metrics(
                te_ys, ops_out, num_bins=_NUM_BINS
            )
            print(f"DUMPING METRICS TO {save_file_name}")
            if not _DBG:
                pkl.dump(ops_recal_metrics, open(save_file_name, "wb"))
    else:
        print("Chose not to run OPS")
    print("=" * 80)

    try:
        print_dict = {
            "moving": moving_recal_metrics,
            "increasing": increasing_recal_metrics,
        }
        if run_prehoc:
            print_dict["prehoc"] = prehoc_metrics
        if run_ops:
            print_dict["ops"] = ops_recal_metrics
        print_metrics = ["pce", "sharpness", "binary_acc", "auroc"]
        for pm in print_metrics:
            print(pm)
            for k, v in print_dict.items():
                print(f"  {k:12}: {v[pm]:.6f}")
        print("=" * 80)
    except:
        print("Unable to print full set of test results because they were not run")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Process command line inputs.")
    parser.add_argument("--target", type=int, default=0,
                        help="Chosen index for target variable (default: 0)")
    parser.add_argument("--dataset", type=str, default='weather',
                        help="Dataset name (default: 'weather')")
    parent_abs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_pred_dir = os.path.join(parent_abs_dir, 'model_predictions')
    parser.add_argument("--pred_dir", type=str, default=default_pred_dir,
                        help="Directory where predictions are stored (default: 'model_predictions')")
    parser.add_argument("--pred_type", type=str, default='gaussian',
                        help="Prediction type, either gaussian or binary (default: 'gaussian')")
    parser.add_argument("--save_dir", type=str, default='save_dir',
                        help="Directory where outputs are stored (default: create 'save_dir')")

    parser.add_argument("--offset_beg", type=int, default=0,
                        help="First offset index to run tests at (default: 0)")
    parser.add_argument("--offset_end", type=int, default=50,
                        help="Last offset index to run tests at (default: 50)")

    args = parser.parse_args()

    ###################################################################################
    # Parse arguments into internal constants
    _NUM_BINS = 30
    _CHOSEN_TARGET = args.target
    _DATASET = args.dataset
    _PRED_TYPE = args.pred_type
    _PREDS_DIR = args.pred_dir
    _SAVE_DIR = os.path.join(args.save_dir, args.pred_type)
    _RUN_IDENTIFIER = f"{_PRED_TYPE}_model-test-target_{_CHOSEN_TARGET}"
    ###################################################################################

    if _PRED_TYPE == 'gaussian':
        if _CHOSEN_TARGET == 0:
            m_use_uf, m_use_ws,  = (2160, 8640)
            i_use_uf = 2160
            ops_hp = (1e-5, 50)
        elif _CHOSEN_TARGET == 1:
            m_use_uf, m_use_ws,  = (336, 8640)
            i_use_uf = 168
            ops_hp = (1e-5, 30)
        elif _CHOSEN_TARGET == 2:
            m_use_uf, m_use_ws,  = (2160, 2160)
            i_use_uf = 336
            ops_hp = (0.0001, 10)
        elif _CHOSEN_TARGET == 3:
            m_use_uf, m_use_ws,  = (1, 4320)
            i_use_uf = 1
            ops_hp = (0.001, 1)
        elif _CHOSEN_TARGET == 4:
            m_use_uf, m_use_ws,  = (1, 4320)
            i_use_uf = 168
            ops_hp = (1e-5, 30)
        elif _CHOSEN_TARGET == 5:
            m_use_uf, m_use_ws,  = (2160, 2160)
            i_use_uf = 720
            ops_hp = (5e-5, 10)
        elif _CHOSEN_TARGET == 6:
            m_use_uf, m_use_ws,  = (1, 168)
            i_use_uf = 24
            ops_hp = (0.0001, 10)
        else:
            raise RuntimeError(f'What??: chosen target {_CHOSEN_TARGET}')
    elif _PRED_TYPE == 'binary':
        if _CHOSEN_TARGET == 0:
            m_use_uf, m_use_ws,  = (2160, 8640)
            i_use_uf = 720
            ops_hp = (5e-5, 30)
        elif _CHOSEN_TARGET == 1:
            m_use_uf, m_use_ws,  = (1, 4320)
            i_use_uf = 168
            ops_hp = (1e-5, 130)
        elif _CHOSEN_TARGET == 2:
            m_use_uf, m_use_ws,  = (336, 4320)
            i_use_uf = 720
            ops_hp = (0.0001, 30)
        elif _CHOSEN_TARGET == 3:
            m_use_uf, m_use_ws,  = (1, 168)
            i_use_uf = 1
            ops_hp = (1e-5, 70)
        elif _CHOSEN_TARGET == 4:
            m_use_uf, m_use_ws, = (1, 2160)
            i_use_uf = 2160
            ops_hp = (1e-5, 50)
        elif _CHOSEN_TARGET == 5:
            m_use_uf, m_use_ws,  = (24, 4320)
            i_use_uf = 336
            ops_hp = (0.001, 10)
        elif _CHOSEN_TARGET == 6:
            m_use_uf, m_use_ws,  = (24, 2160)
            i_use_uf = 1
            ops_hp = (1e-5, 10)
        else:
            raise RuntimeError(f'What??: chosen target {_CHOSEN_TARGET}')

    # Print test settings
    print("%" * 30)
    print(f"MOVING SETTINGS: update frequency={m_use_uf}, window size={m_use_ws}")
    print(f"INCREASING SETTINGS: update frequency={i_use_uf}")
    print(f"OPS HP: {ops_hp}")
    print(f"TARGET IDX: {_CHOSEN_TARGET}")
    print(f"PRED TYPE: {_PRED_TYPE}")
    print("%" * 30)

    preds_fname = f"{_PREDS_DIR}/{_PRED_TYPE}_model-target_{_CHOSEN_TARGET}_preds.pkl"
    if not os.path.exists(preds_fname):
        raise RuntimeError(
            f"  Pickle file containing predictions does not exist: {preds_fname}"
        )

    print(f"  Getting base predictions from {preds_fname}")

    if _PRED_TYPE == "binary":
        binary_preds = pkl.load(open(preds_fname, "rb"))

        full_val_preds = binary_preds["val"]["prob_preds"].flatten()
        full_val_ys = binary_preds["val"]["ys"].flatten()
        full_te_preds = binary_preds["te"]["prob_preds"].flatten()
        full_te_ys = binary_preds["te"]["ys"].flatten()
    elif _PRED_TYPE == "gaussian":
        gaussian_preds = pkl.load(open(preds_fname, "rb"))

        full_val_preds, full_val_ys = get_parity_prob(
            mu_list=gaussian_preds["val"]["mu_preds"],
            sigma_list=np.exp(gaussian_preds["val"]["sigma_preds"]),
            y_list=gaussian_preds["val"]["ys"],
        )
        full_te_preds, full_te_ys = get_parity_prob(
            mu_list=gaussian_preds["te"]["mu_preds"],
            sigma_list=np.exp(gaussian_preds["te"]["sigma_preds"]),
            y_list=gaussian_preds["te"]["ys"],
        )
        full_val_preds = full_val_preds.flatten()
        full_val_ys = full_val_ys.flatten().astype(int)
        full_te_preds = full_te_preds.flatten()
        full_te_ys = full_te_ys.flatten().astype(int)

        full_val_mu = gaussian_preds["val"]["mu_preds"].flatten()
        full_val_sigma = np.exp(gaussian_preds["val"]["sigma_preds"]).flatten()
        full_te_mu = gaussian_preds["te"]["mu_preds"].flatten()
        full_te_sigma = np.exp(gaussian_preds["te"]["sigma_preds"]).flatten()
        full_val_target_y = gaussian_preds["val"]["ys"].flatten()
        full_te_target_y = gaussian_preds["te"]["ys"].flatten()

    max_offset_idx = 100
    qce_list = []
    print(f"Running from {args.offset_beg} to {args.offset_end}")
    for offset_idx in range(args.offset_beg, args.offset_end):
        if offset_idx == max_offset_idx:
            break

        print("OFFSET IDX", offset_idx)
        cur_run_identifier = _RUN_IDENTIFIER + f"-offset_{offset_idx}"

        try:
            weather_offset = offset_idx * 336  # offset by 2 weeks

            cur_val_preds = full_val_preds[-8640:]
            cur_val_ys = full_val_ys[-8640:]
            cur_te_preds = full_te_preds[weather_offset : weather_offset + (3 * 8640)]
            cur_te_ys = full_te_ys[weather_offset : weather_offset + (3 * 8640)]

            if _PRED_TYPE == 'gaussian':
                cur_te_mu = full_te_mu[weather_offset : weather_offset + (3 * 8640)]
                cur_te_sigma = full_te_sigma[weather_offset : weather_offset + (3 * 8640)]
                cur_te_target_y = full_te_target_y[
                    weather_offset : weather_offset + (3 * 8640)
                ]
                curr_qce = uct.metrics.mean_absolute_calibration_error(
                    cur_te_mu, cur_te_sigma, cur_te_target_y
                )
                qce_list.append(curr_qce)

            print("   FINAL CALL: test results will be saved to")
            print(f"       {_SAVE_DIR}/{_RUN_IDENTIFIER}-***")
            # breakpoint()
            run_all_method_tests(
                run_suffix_removed=cur_run_identifier,
                te_preds=cur_te_preds,
                te_ys=cur_te_ys,
                val_preds=cur_val_preds,
                val_ys=cur_val_ys,
                m_use_uf=m_use_uf,
                m_use_ws=m_use_ws,
                i_use_uf=i_use_uf,
                use_ops_hp=ops_hp,
                _NUM_BINS=_NUM_BINS,
                _METRICS_SAVE_DIR=_SAVE_DIR,
                run_prehoc=True,
                run_ops=True,
                run_increasing=True,
                run_moving=True,
            )
        except:
            max_offset_idx += 1

    if _PRED_TYPE == 'gaussian':
        # now, get mean and standard error of qce_list
        qce_list = np.array(qce_list)
        qce_mean = np.mean(qce_list)
        qce_std = np.std(qce_list)
        qce_stderr = qce_std / np.sqrt(len(qce_list))
        print(f"Prehoc QCE: {qce_mean:.6f} \pm {qce_std:.6f}")
