import numpy as np
import pandas as pd
from scipy import stats


# ── VaR / CVaR methods ────────────────────────────────────────────────────────

def parametric_var(port_ret, confidence=0.95):
    """Normal parametric VaR and CVaR."""
    mu    = port_ret.mean()
    sigma = port_ret.std()
    z     = stats.norm.ppf(1 - confidence)
    var   = -(mu + z * sigma)
    cvar  = -(mu - sigma * stats.norm.pdf(z) / (1 - confidence))
    return abs(var), abs(cvar)


def historical_var(port_ret, confidence=0.95):
    """Historical simulation VaR and CVaR."""
    var  = np.percentile(port_ret, (1 - confidence) * 100)
    cvar = port_ret[port_ret <= var].mean()
    return abs(var), abs(cvar)


def monte_carlo_var(port_ret, confidence=0.95, n_sims=100_000, seed=42):
    """Monte Carlo VaR using normal distribution."""
    np.random.seed(seed)
    mu, sigma  = port_ret.mean(), port_ret.std()
    simulated  = np.random.normal(mu, sigma, n_sims)
    var        = np.percentile(simulated, (1 - confidence) * 100)
    cvar       = simulated[simulated <= var].mean()
    return abs(var), abs(cvar), simulated


def student_t_var(port_ret, confidence=0.95, n_sims=100_000, seed=42):
    """
    Monte Carlo VaR with Student-t innovations.
    Degrees of freedom estimated from excess kurtosis: df = 6/K + 4
    """
    np.random.seed(seed)
    mu, sigma = port_ret.mean(), port_ret.std()
    K         = port_ret.kurt()
    df        = max(6 / K + 4, 4.1)
    simulated = mu + sigma * stats.t.rvs(df=df, size=n_sims)
    var       = np.percentile(simulated, (1 - confidence) * 100)
    cvar      = simulated[simulated <= var].mean()
    print(f"Student-t df: {df:.2f}  (kurtosis={K:.2f})")
    return abs(var), abs(cvar)


def cornish_fisher_var(port_ret, confidence=0.95, n_sims=100_000, seed=42):
    """
    Cornish-Fisher VaR — adjusts normal quantile for skewness and kurtosis.
    Most reliable at 99%; can be unstable at 95%.
    """
    mu    = port_ret.mean()
    sigma = port_ret.std()
    S     = port_ret.skew()
    K     = port_ret.kurt()
    z     = stats.norm.ppf(1 - confidence)

    z_cf = (z
            + (z**2 - 1) * S / 6
            + (z**3 - 3*z) * K / 24
            - (2*z**3 - 5*z) * S**2 / 36)

    var  = abs(-(mu + z_cf * sigma))

    np.random.seed(seed)
    sim  = np.random.normal(mu, sigma, n_sims)
    cvar = abs(sim[sim <= -var].mean())

    return var, cvar


# ── Rolling VaR ───────────────────────────────────────────────────────────────

def rolling_historical_var(port_ret, window=252, confidence=0.95):
    """Rolling 1-year historical VaR and CVaR."""
    rolling_var  = pd.Series(index=port_ret.index[window:], dtype=float)
    rolling_cvar = pd.Series(index=port_ret.index[window:], dtype=float)

    for i in range(window, len(port_ret)):
        v, c = historical_var(port_ret.iloc[i-window:i], confidence)
        rolling_var.iloc[i-window]  = v
        rolling_cvar.iloc[i-window] = c

    return rolling_var, rolling_cvar


# ── Kupiec backtest ───────────────────────────────────────────────────────────

def kupiec_test(port_ret, var_value, confidence=0.95):
    """Kupiec test for a scalar (static) VaR."""
    violations = (port_ret < -var_value).sum()
    n          = len(port_ret)
    p_exp      = 1 - confidence
    p_act      = violations / n

    if p_act == 0 or p_act == 1:
        lr_stat = np.inf
    else:
        lr_stat = -2 * np.log(
            ((1-p_exp)**(n-violations) * p_exp**violations) /
            ((1-p_act)**(n-violations) * p_act**violations)
        )

    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    return {
        "violations":     int(violations),
        "violation_rate": f"{p_act:.2%}",
        "expected_rate":  f"{p_exp:.2%}",
        "LR stat":        round(lr_stat, 4),
        "p-value":        round(p_value, 4),
        "passes":         "YES" if p_value > 0.05 else "NO",
    }


def kupiec_test_series(port_ret, var_series, confidence=0.95):
    """Kupiec test for a dynamic VaR series (aligned on index)."""
    aligned            = pd.concat([port_ret, var_series], axis=1).dropna()
    aligned.columns    = ["ret", "var"]
    violations         = (aligned["ret"] < -aligned["var"]).sum()
    n                  = len(aligned)
    p_exp              = 1 - confidence
    p_act              = violations / n

    if p_act == 0 or p_act == 1:
        lr_stat = np.inf
    else:
        lr_stat = -2 * np.log(
            ((1-p_exp)**(n-violations) * p_exp**violations) /
            ((1-p_act)**(n-violations) * p_act**violations)
        )

    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    return {
        "violations":     int(violations),
        "violation_rate": f"{p_act:.2%}",
        "expected_rate":  f"{p_exp:.2%}",
        "LR stat":        round(lr_stat, 4),
        "p-value":        round(p_value, 4),
        "passes":         "YES" if p_value > 0.05 else "NO",
    }


def vol_forecast_mse(predicted_vol, actual_returns):
    """MSE of vol forecast vs |r_t| realized vol proxy."""
    realized = np.asarray(actual_returns).flatten()
    pred     = np.asarray(predicted_vol).flatten()
    min_len  = min(len(pred), len(realized))
    return float(np.mean((pred[:min_len] - realized[:min_len]) ** 2))


# ── Summary table ─────────────────────────────────────────────────────────────

def print_var_summary(port_ret, confidence=0.95):
    var_p,  cvar_p  = parametric_var(port_ret, confidence)
    var_h,  cvar_h  = historical_var(port_ret, confidence)
    var_mc, cvar_mc, _ = monte_carlo_var(port_ret, confidence)
    var_t,  cvar_t  = student_t_var(port_ret, confidence)

    print(f"\n{'Method':<20} {'VaR':>8} {'CVaR':>8} {'VaR $100k':>12} {'Kupiec':>8}")
    print("-" * 62)
    for name, var, cvar in [
        ("Parametric",     var_p,  cvar_p),
        ("Historical",     var_h,  cvar_h),
        ("MC Normal",      var_mc, cvar_mc),
        ("Student-t MC",   var_t,  cvar_t),
    ]:
        r = kupiec_test(port_ret, var, confidence)
        print(f"{name:<20} {var*100:>7.3f}%  {cvar*100:>7.3f}%  "
              f"${var*100_000:>9,.0f}  {r['passes']:>8}")

    return var_p, cvar_p, var_h, cvar_h, var_mc, cvar_mc, var_t, cvar_t
