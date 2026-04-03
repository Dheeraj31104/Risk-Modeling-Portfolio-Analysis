import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model


def fit_garch(port_ret, dist="normal"):
    """
    Fit GARCH(1,1) to portfolio returns.

    Parameters
    ----------
    port_ret : pd.Series   daily log returns (decimal)
    dist     : str         "normal" or "skewt"

    Returns
    -------
    result : arch ModelResult
    """
    port_ret_pct = port_ret * 100   # arch expects percentage returns

    model = arch_model(
        port_ret_pct,
        vol="Garch",
        p=1, q=1,
        dist=dist,
        mean="Constant",
    )
    result = model.fit(disp="off")

    alpha = result.params["alpha[1]"]
    beta  = result.params["beta[1]"]
    hl    = np.log(0.5) / np.log(alpha + beta)

    print(f"\nGARCH(1,1) [{dist}]")
    print(f"  omega        : {result.params['omega']:.4f}")
    print(f"  alpha[1]     : {alpha:.4f}")
    print(f"  beta[1]      : {beta:.4f}")
    print(f"  alpha+beta   : {alpha+beta:.4f}")
    print(f"  half-life    : {hl:.1f} days")
    print(f"  AIC          : {result.aic:.2f}")
    print(f"  Log-lik      : {result.loglikelihood:.2f}")

    if dist == "skewt":
        print(f"  eta (df)     : {result.params.get('eta', 'N/A'):.4f}")
        print(f"  lambda (skew): {result.params.get('lambda', 'N/A'):.4f}")

    return result


def garch_var(port_ret, result, confidence=0.95):
    """
    Dynamic daily VaR and CVaR from a fitted GARCH model.
    Uses normal quantile regardless of fitted innovation distribution
    (swap stats.norm → stats.t with fitted df for skewed-t variant).

    Returns
    -------
    var_series  : pd.Series  daily VaR  (positive = loss)
    cvar_series : pd.Series  daily CVaR (positive = loss)
    """
    mu       = result.params["mu"] / 100
    cond_vol = result.conditional_volatility / 100
    z        = stats.norm.ppf(1 - confidence)

    var_series  = -(mu + z * cond_vol)
    cvar_series = -(mu - cond_vol * stats.norm.pdf(z) / (1 - confidence))

    return var_series, cvar_series


def garch_summary(port_ret, result_normal, result_skt, confidence=0.95):
    """Print comparison table for Normal vs Skewed-t GARCH."""
    from src.risk_metrics import kupiec_test_series

    var_n,   cvar_n   = garch_var(port_ret, result_normal, confidence)
    var_skt, cvar_skt = garch_var(port_ret, result_skt,    confidence)

    kup_n   = kupiec_test_series(port_ret, var_n,   confidence)
    kup_skt = kupiec_test_series(port_ret, var_skt, confidence)

    cond_vol     = result_normal.conditional_volatility / 100
    cond_vol_skt = result_skt.conditional_volatility   / 100

    print(f"\n{'':30} {'Normal':>12} {'Skewed-t':>12}")
    print("-" * 56)
    for label, vn, vs in [
        ("Mean VaR",   var_n.mean(),   var_skt.mean()),
        ("Min  VaR",   var_n.min(),    var_skt.min()),
        ("Max  VaR",   var_n.max(),    var_skt.max()),
        ("Mean CVaR",  cvar_n.mean(),  cvar_skt.mean()),
        ("Violations", kup_n["violations"], kup_skt["violations"]),
    ]:
        if isinstance(vn, float):
            print(f"{label:30} {vn*100:>11.3f}%  {vs*100:>11.3f}%")
        else:
            print(f"{label:30} {vn:>12}  {vs:>12}")

    print(f"\n  Kupiec Normal  : {kup_n['violation_rate']}  → {kup_n['passes']}")
    print(f"  Kupiec Skewed-t: {kup_skt['violation_rate']}  → {kup_skt['passes']}")

    return var_n, cvar_n, var_skt, cvar_skt, cond_vol, cond_vol_skt
