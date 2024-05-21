"""
Utility functions for quantile prediction
"""

# Author: Alessandro Brusaferri
# License: Apache-2.0 license


import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import QuantileRegressor
import sys


def build_alpha_quantiles_map(target_alpha: List, target_quantiles: List):
    """
    Build the map between PIs coverage levels and related quantiles
    """
    alpha_q = {'med': target_quantiles.index(0.5)}
    for alpha in target_alpha:
        alpha_q[alpha] = {
            'l': target_quantiles.index(alpha / 2),
            'u': target_quantiles.index(1 - alpha / 2),
        }
    return alpha_q

def build_target_quantiles(target_alpha):
    """
    Build target quantiles from the alpha list
    """
    target_quantiles = [0.5]
    for alpha in target_alpha:
        target_quantiles.append(alpha / 2)
        target_quantiles.append(1 - alpha / 2)
    target_quantiles.sort()
    return target_quantiles


def fix_quantile_crossing(preds: np.array):
    """
    Fix crossing in the predicted quantiles by means of post-hoc sorting
    """
    return np.sort(preds, axis=-1)


def plot_quantiles(results: pd.DataFrame, target: str):
    """
    Plot predicted quantiles
    """
    title = target
    idx = results[target].index
    fig1, ax1 = plt.subplots()
    for i in results.columns.to_list():
        ax1.plot(idx, results[i], linestyle="-", color='steelblue', linewidth=0.9)

    ax1.plot(idx, results[target], '-', color='firebrick', label='$y_{true}$')
    ax1.grid()
    ax1.legend()
    ax1.set_ylabel("Predicted quantiles")
    ax1.set_title(title)
    fig1.show()


def compute_qra(preds_cali: np.array, y_cali: np.array, preds_test: np.array, settings: Dict):
    """
    Compute PI using Quantile Regression Averaging.

    Python implementation, following: https://ideas.repec.org/c/wuu/hscode/m14003.html

    :param preds_cali: predictions of the ensembles over the calibration set
    :param y_cali: true labels over the calibration set
    :param preds_test: predictions over the test set
    :param settings: configurations
    :return: PIs
    """

    # Prepare the data in the proper format
    y_cali = y_cali.flatten()
    ensemble_calib_preds_fl = preds_cali.reshape(-1, preds_cali.shape[-1])
    ensemble_test_preds_fl = preds_test.reshape(-1, preds_test.shape[-1])

    # Compute PIs for each quantile in the target quantile list
    pis = []
    for q_i in settings['target_quantiles']:
        # Instantiate class to compute lower/upper PI components
        # In sklear QR alpha defines the weights in the l1 norm of the regression (set to 0 to obtain standard regression)
        # Here alpha refers to the miscoverage rate, as usually adopted in CP
        qr = QuantileRegressor(quantile=q_i, alpha=0, solver='highs')
        # Fit QR for the current quantile using ensemble predictions (and labels) on the calibration set
        qr.fit(ensemble_calib_preds_fl, y_cali)
        # Predict quantiles for the test sample, including reshape for concatenation
        pis.append(qr.predict(ensemble_test_preds_fl).reshape(-1, preds_test.shape[1], 1))

    # Concatenate the PIs
    test_PIs = np.concatenate(pis, axis=-1)

    # Fix quantile crossing by sorting
    # return prediction flattened in temporal dimension (sample over pred horizon)
    return fix_quantile_crossing(test_PIs.reshape(-1, test_PIs.shape[-1]))


def exec_cqr(preds_cali: np.array, y_cali: np.array, preds_test: np.array, settings: Dict):
    """
       Compute conformalized prediction intervals:
    """

    def __asym_mode__(conf_scores: np.array, alpha:float):
        """
        Function to compute asymmetric cqr
        """
        q=(1 - alpha / 2)
        Q_l = np.quantile(a=conf_scores[:, :, 0], q=q, axis=0)
        Q_h = np.quantile(a=conf_scores[:, :, 1], q=q, axis=0)
        return np.stack([Q_l, Q_h], axis=-1)

    def __conform_prediction_intervals__(pred_pi: np.array, q_cp: np.array):
        """
        Function to compute conformalized PIs
        """
        return np.concatenate([np.expand_dims(pred_pi[:, :, 0] - q_cp[:, 0], axis=2),
                               np.expand_dims(pred_pi[:, :, 1] + q_cp[:, 1], axis=2)], axis=2)

    # map the method to be employed to conformalize
    compute_score_quantiles = __asym_mode__

    # Conformalize predictions for each alpha
    for alpha in settings['target_alpha']:
        # get index of the lower/upper quantiles for the current alpha from the map
        lq_idx = settings['q_alpha_map'][alpha]['l']
        uq_idx = settings['q_alpha_map'][alpha]['u']

        # Compute conformity scores
        conf_scores = np.stack([preds_cali[:, :, lq_idx] - y_cali, y_cali - preds_cali[:, :, uq_idx]], axis=-1)
        conformalized_alpha_PIs= __conform_prediction_intervals__(pred_pi=preds_test[:,:,[lq_idx, uq_idx]],
                                                                  q_cp=compute_score_quantiles(conf_scores=conf_scores,
                                                                                               alpha=alpha))
        # replace test_pred PIs columns with conformalized PIs for each alpha
        preds_test[:, :, lq_idx] = conformalized_alpha_PIs[:,:,0]
        preds_test[:, :, uq_idx] = conformalized_alpha_PIs[:,:,1]

    # Fix quantile crossing
    # return prediction flattened in temporal dimension (sample over pred horizon)
    return fix_quantile_crossing(preds_test.reshape(-1, preds_test.shape[-1]))


def exec_cp(preds_cali: np.array, y_cali: np.array, preds_test: np.array, settings: Dict):
    preds_cali = np.squeeze(preds_cali, axis=-1)
    if preds_test.shape[0]>1:
        sys.exit('ERROR: exec_cup supports single test samples')
    # Compute conformity score (absolute residual)
    conf_score = np.abs(preds_cali - y_cali)
    n=conf_score.shape[0]
    # Stack the quantiles to the point pred for each alpha (next sorted by fixing crossing)
    preds_test_q=[preds_test]
    for alpha in settings['target_alpha']:
        q = np.ceil((n + 1) * (1 - alpha)) / n
        Q_1_alpha= np.expand_dims(np.quantile(a=conf_score, q=q, axis=0, method='higher'), axis=(0,-1))
        # Append lower/upper PIs for the current alpha
        preds_test_q.append(preds_test - Q_1_alpha)
        preds_test_q.append(preds_test + Q_1_alpha)
    preds_test_q = np.concatenate(preds_test_q, axis=2)
    # Fix quantile crossing
    # return prediction flattened in temporal dimension (sample over pred horizon)
    return fix_quantile_crossing(preds_test_q.reshape(-1, preds_test_q.shape[-1]))
