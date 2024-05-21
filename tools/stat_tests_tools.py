import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import chi2
from typing import List, Union, Dict


def get_critical_chi_square(significance_level: float = 0.05):
    # test confidence level
    conf_level = 1- significance_level
    return chi2.ppf(conf_level, 1)  # one degree of freedom


def kupiec_test(
        PI_hits: Union[List[int], np.ndarray, pd.Series, pd.DataFrame],
        alpha: float = 0.1,
        significance_level: float = 0.05,
) -> Dict:
    """
       code from vartests 0.2.2: Python library to perform some statistical tests to evaluate Value at Risk (VaR) Models
       https://github.com/rafa-rod/vartests

       Modified following naming convention in: https://www.sciencedirect.com/science/article/pii/S1364032117308808

       Perform Kupiec Test (1995).
       The main goal is to verify if the number of violations, i.e. proportion of failures, is consistent with the
       violations predicted by the model.

        Parameters:
            PI_hits (series):    series of hits of PIs (i.e., true value covered by PI)
            alpha (float):             miscoverage denoting the (1-alpha) confidence level of the PI
            significance_level (float):  sinificance level of the test (e.g., 5%, 1%)
        Returns:
            answer (dict):             statistics and decision of the test
    """
    if isinstance(PI_hits, pd.core.series.Series):
        n1 = PI_hits[PI_hits == 1].count()
    elif isinstance(PI_hits, pd.core.frame.DataFrame):
        n1 = PI_hits[PI_hits == 1].count().values[0]
    elif isinstance(PI_hits, list):
        lista_array = np.array(PI_hits)
        n1 = len(lista_array[lista_array == 1])
    elif isinstance(PI_hits, np.ndarray):
        n1 = len(PI_hits[PI_hits == 1])
    else:
        raise ValueError("Input must be list, array, series or dataframe.")

    c = 1 - alpha
    n = len(PI_hits)
    n0 = n - n1
    pi = min(n1 / n, 0.999999999999999)

    LR_UC = -2 * (
                    n0 * np.log(1 - c)
                  + n1 * np.log(c)
                  - n0 * np.log(1 - pi)
                  - n1 * np.log(pi)
    )

    critical_chi_square = get_critical_chi_square(significance_level)

    if LR_UC > critical_chi_square:
        result = "Reject H0"
        passed = False
    else:
        result = "Fail to reject H0"
        passed = True

    # p - value
    LR_UP_p = 1 - chi2.cdf(LR_UC, 1)

    return {
        "LR_UC": LR_UC,
        "chi square critical value": critical_chi_square,
        "null hypothesis": f"Probability of failure is {round(1 - alpha, 3)}",
        "result": result,
        "p-value": LR_UP_p,
        "passed": passed
    }



"""
Functions to compute and plot the univariate and multivariate versions of the Diebold-Mariano (DM) test.
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

def DM(p_real, losses_model_1, losses_model_2, version='univariate'):
    """
    Function that performs the one-sided DM test in the contex of electricity price forecasting

    code modified from https://github.com/jeslago/epftoolbox to abstract the specific loss function employed beyond

    The test compares whether there is a difference in predictive accuracy between two forecasters.
    Particularly, the one-sided DM test evaluates the null hypothesis H0
    of the forecasting loss of model 2 ``losses_model_2`` being larger (worse) than the forecasting loss of model 1
    ``losses_model_1`` vs the alternative hypothesis H1 of the scores ``losses_model_2`` being smaller (better).
    Hence, rejecting H0 means that the forecasts of model 2 is significantly more accurate
    that forecast of model 1. (Note that this is an informal definition. For a formal one we refer to
    `here <https://epftoolbox.readthedocs.io/en/latest/modules/cite.html>`_)
    Two versions of the test are possible:
        1. A univariate version with as many independent tests performed as prices per day, i.e. 24
        tests in most day-ahead electricity markets.
        2. A multivariate with the test performed jointly for all hours using the multivariate
        loss differential series (see this
        `article <https://epftoolbox.readthedocs.io/en/latest/modules/cite.html>`_ for details.

    Parameters
    ----------
    p_real : numpy.ndarray
        Array of shape :math:`(n_\\mathrm{days}, n_\\mathrm{prices/day})` representing the real market
        prices
    losses_model_1 : TYPE
        Array of shape :math:`(n_\\mathrm{days}, n_\\mathrm{prices/day})` representing the losses of the first model
    losses_model_2 : TYPE
        Array of shape :math:`(n_\\mathrm{days}, n_\\mathrm{prices/day})` representing the losses of the second model
    norm : int, optional
        Norm used to compute the loss differential series. At the moment, this value must either
        be 1 (for the norm-1) or 2 (for the norm-2).
    version : str, optional
        Version of the test as defined in
        `here <https://epftoolbox.readthedocs.io/en/latest/modules/cite.html>`_. It can have two values:
        ``'univariate`` or ``'multivariate``
    Returns
    -------
    float, numpy.ndarray
        The p-value after performing the test. It is a float in the case of the multivariate test
        and a numpy array with a p-value per hour for the univariate test

    """

    # Checking that all time series have the same shape
    if p_real.shape != losses_model_1.shape or p_real.shape != losses_model_2.shape:
        raise ValueError('The three time series must have the same shape')

    # Ensuring that time series have shape (n_days, n_prices_day)
    if len(p_real.shape) == 1 or (len(p_real.shape) == 2 and p_real.shape[1] == 1):
        raise ValueError('The time series must have shape (n_days, n_prices_day')

    # Computing the test statistic
    if version == 'univariate':
        # Computing the loss differential series for the univariate test
        d = losses_model_1 - losses_model_2

        # Computing the test statistic
        mean_d = np.mean(d, axis=0)
        var_d = np.var(d, ddof=0, axis=0)

    elif version == 'multivariate':
        # Computing the loss differential series for the multivariate test
        d = np.mean(losses_model_1, axis=1) - np.mean(losses_model_2, axis=1)
        # Computing the test statistic
        mean_d = np.mean(d)
        var_d = np.var(d, ddof=0)

    N = d.shape[0]
    DM_stat = mean_d / np.sqrt((1 / N) * var_d)

    p_value = 1 - stats.norm.cdf(DM_stat)

    return p_value


def plot_multivariate_DM_test(real_price, forecasts_losses, title='DM test', savefig=False, path=''):
    """Plotting the results of comparing forecasts using the multivariate DM test.

    The resulting plot is a heat map in a chessboard shape. It represents the p-value
    of the null hypothesis of the forecast in the y-axis being significantly more
    accurate than the forecast in the x-axis. In other words, p-values close to 0
    represent cases where the forecast in the x-axis is significantly more accurate
    than the forecast in the y-axis.

    Parameters
    ----------
    real_price : pandas.DataFrame
        Dataframe that contains the real prices
    forecasts_losses : TYPE
        Dataframe that contains the losses computed using the forecasts of different models. The column names are the
        forecast/model names. The number of datapoints should equal the number of datapoints
        in ``real_price``.
    title : str, optional
        Title of the generated plot
    savefig : bool, optional
        Boolean that selects whether the figure should be saved in the current folder
    path : str, optional
        Path to save the figure. Only necessary when `savefig=True`

    """

    # Computing the multivariate DM test for each forecast pair
    p_values = pd.DataFrame(index=forecasts_losses.columns, columns=forecasts_losses.columns)

    for model1 in forecasts_losses.columns:
        for model2 in forecasts_losses.columns:
            # For the diagonal elemnts representing comparing the same model we directly set a
            # p-value of 1
            if model1 == model2:
                p_values.loc[model1, model2] = 1
            else:
                p_values.loc[model1, model2] = DM(p_real=real_price.values.reshape(-1, 24),
                                                  losses_model_1=forecasts_losses.loc[:, model1].values.reshape(-1, 24),
                                                  losses_model_2=forecasts_losses.loc[:, model2].values.reshape(-1, 24),
                                                  version='multivariate')


    # Defining color map
    red = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], [0]])
    green = np.concatenate([np.linspace(0.5, 1, 50), np.zeros(50)])
    blue = np.zeros(100)
    rgb_color_map = np.concatenate([red.reshape(-1, 1), green.reshape(-1, 1),
                                    blue.reshape(-1, 1)], axis=1)
    rgb_color_map = mpl.colors.ListedColormap(rgb_color_map)

    # Generating figure
    fig = plt.figure(figsize=(3.5, 3.5))
    plt.imshow(p_values.astype(float).values, cmap=rgb_color_map, vmin=0, vmax=0.1)
    plt.xticks(range(len(forecasts_losses.columns)), forecasts_losses.columns, rotation=90.)
    plt.yticks(range(len(forecasts_losses.columns)), forecasts_losses.columns)
    plt.plot(range(p_values.shape[0]), range(p_values.shape[0]), 'wx')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()

    if savefig:
        plt.savefig(title + '.png', dpi=300)
        plt.savefig(title + '.eps')

    plt.show()