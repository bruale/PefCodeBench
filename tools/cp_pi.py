import numpy as np
from tqdm import tqdm


"""
Code from https://github.com/aangelopoulos/conformal-time-series.git
"""


def mytan(x):
    if x >= np.pi/2:
        return np.infty
    elif x <= -np.pi/2:
        return -np.infty
    else:
        return np.tan(x)

def saturation_fn_log(x, t, Csat, KI):
    if KI == 0:
        return 0
    tan_out = mytan(x * np.log(t+1)/(Csat * (t+1)))
    out = KI * tan_out
    return  out

def saturation_fn_sqrt(x, t, Csat, KI):
    return KI * mytan((x * np.sqrt(t+1))/((Csat * (t+1))))

def quantile_integrator_log(
    scores,
    alpha,
    lr,
    Csat,
    KI,
    ahead,
    T_burnin,
    proportional_lr=True,
    *args,
    **kwargs
):
    data = kwargs['data'] if 'data' in kwargs.keys() else None
    results = quantile_integrator(scores, alpha, lr, data, T_burnin, Csat, KI, True, ahead, proportional_lr=proportional_lr)
    results['method'] = "Quantile+Integrator (log)"
    return results


"""
    This is the master method for the quantile, integrator
"""
def quantile_integrator(
    scores,
    alpha,
    lr,
    data,
    T_burnin,
    Csat,
    KI,
    upper,
    ahead,
    integrate=True,
    proportional_lr=True,
    *args,
    **kwargs
):
    # Initialization
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    qts = np.zeros((T_test,))
    integrators = np.zeros((T_test,))
    covereds = np.zeros((T_test,))
    seasonal_period = kwargs.get('seasonal_period')
    if seasonal_period is None:
        seasonal_period = 1

    # Run the main loop
    # At time t, we observe y_t and make a prediction for y_{t+ahead}
    # We also update the quantile at the next time-step, q[t+1], based on information up to and including t_pred = t - ahead + 1.
    #lr_t = lr * (scores[:T_burnin].max() - scores[:T_burnin].min()) if proportional_lr and T_burnin > 0 else lr
    for t in tqdm(range(T_test)):
        t_lr = t
        t_lr_min = max(t_lr - T_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue # We can't make any predictions yet if our prediction time has not yet arrived
        # First, observe y_t and calculate coverage
        covereds[t] = qs[t] >= scores[t]
        # Next, calculate the quantile update and saturation function
        grad = alpha if covereds[t_pred] else -(1-alpha)
        integrator_arg = (1-covereds)[:t_pred].sum() - (t_pred)*alpha
        integrator = saturation_fn_log(integrator_arg, t_pred, Csat, KI)

        # Update the next quantile
        if t < T_test - 1:
            qts[t+1] = qts[t] - lr_t * grad
            integrators[t+1] = integrator if integrate else 0
            qs[t+1] = qts[t+1] + integrators[t+1]
    results = {"method": "Quantile+Integrator (log)", "q" : qs}
    return results


def cts_pid(data, alpha, lr, Csat, KI, T_burnin, score_function_name="cqr-asymmetric", ahead=1, minsize=0):
    fn = quantile_integrator_log
    kwargs = {'Csat': Csat, 'KI': KI, "T_burnin": T_burnin, "data": data, "seasonal_period": None,
              "ahead": ahead}
    # Initialize the score function
    if score_function_name == "cqr-symmetric":
        def score_function(y, forecasts):
            return np.maximum(forecasts[0] - y, y - forecasts[-1])

        def set_function(forecast, q):
            return np.array([forecast[0] - q, forecast[-1] + q])

        asymmetric = False
    elif score_function_name == "cqr-asymmetric":
        def score_function(y, forecasts):
            return np.array([forecasts[0] - y, y - forecasts[-1]])

        def set_function(forecast, q):
            return np.array([forecast[0] - q[0], forecast[-1] + q[1]])

        asymmetric = True
    else:
        raise ValueError("Invalid score function name")

    # Compute scores
    if 'scores' not in data.columns:
        data['scores'] = [score_function(y, forecast) for y, forecast in zip(data['y'], data['forecasts'])]

    # Compute the results
    results = {}
    if asymmetric:
        stacked_scores = np.stack(data['scores'].to_list())
        kwargs['upper'] = False
        q0 = fn(stacked_scores[:, 0], alpha / 2, lr, **kwargs)['q']
        kwargs['upper'] = True
        q1 = fn(stacked_scores[:, 1], alpha / 2, lr, **kwargs)['q']
        q = [np.array([q0[i], q1[i]]) for i in range(len(q0))]
    else:
        kwargs['upper'] = True
        q = fn(data['scores'].to_numpy(), alpha, lr, **kwargs)['q']

    sets = [set_function(data['forecasts'].interpolate().to_numpy()[i], q[i]) for i in range(len(q))]
    # Make sure the set size is at least minsize by setting sets[j][0] = min(sets[j][0], sets[j][1]-minsize) and sets[j][1] = max(sets[j][1], sets[j][1]+minsize)
    sets = [np.array([np.minimum(sets[j][0], sets[j][1] - minsize), np.maximum(sets[j][1], sets[j][0] + minsize)])
            for j in range(len(sets))]

    return {"q": q, "sets": sets}