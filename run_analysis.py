"""
Risk Modeling & Portfolio Analysis — full pipeline
===================================================
Levels 1–7: Parametric → Historical → Monte Carlo → Student-t → CF → GARCH → LSTM

Run:
    python run_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

from src.data_loader  import load_portfolio, CRISES
from src.risk_metrics import (parametric_var, historical_var, monte_carlo_var,
                               student_t_var, cornish_fisher_var,
                               rolling_historical_var,
                               kupiec_test, kupiec_test_series,
                               vol_forecast_mse, print_var_summary)
from src.garch_var    import fit_garch, garch_var, garch_summary
from src.lstm_model   import (VolatilityLSTM, build_sequences, train_lstm,
                               predict, lstm_var_series, get_device)

CONFIDENCE   = 0.95
PORTFOLIO_V  = 100_000
WINDOW       = 20        # LSTM sequence window
LSTM_EPOCHS  = 100


def main():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" 1. DATA LOADING")
    print("="*60)
    prices, log_ret, port_ret, weights = load_portfolio()

    # ── 2. VaR / CVaR ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" 2. VaR / CVaR — Levels 1–5")
    print("="*60)
    var_p, cvar_p, var_h, cvar_h, var_mc, cvar_mc, var_t, cvar_t = \
        print_var_summary(port_ret, CONFIDENCE)

    var_cf99, cvar_cf99 = cornish_fisher_var(port_ret, confidence=0.99)
    print(f"\nCornish-Fisher VaR (99%):  {var_cf99*100:.3f}%")
    print(f"Cornish-Fisher CVaR (99%): {cvar_cf99*100:.3f}%")

    # ── 3. Rolling historical VaR ─────────────────────────────────────────────
    print("\n Computing rolling 1-yr historical VaR ...")
    rolling_var, rolling_cvar = rolling_historical_var(port_ret, confidence=CONFIDENCE)

    # ── 4. GARCH ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" 3. GARCH(1,1) — Level 6")
    print("="*60)
    result_n   = fit_garch(port_ret, dist="normal")
    result_skt = fit_garch(port_ret, dist="skewt")

    var_gn, cvar_gn, var_gskt, cvar_gskt, cond_vol, cond_vol_skt = \
        garch_summary(port_ret, result_n, result_skt, CONFIDENCE)

    # ── 5. LSTM ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" 4. LSTM Volatility Forecaster — Level 7")
    print("="*60)

    device = get_device()
    print(f"Device: {device}")

    X_train, X_test, y_train, y_test, test_dates, scale, n_train = \
        build_sequences(port_ret, window=WINDOW)

    model = VolatilityLSTM().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_losses, val_losses = train_lstm(
        model, X_train, y_train, X_test, y_test, device, epochs=LSTM_EPOCHS)

    preds = predict(model, X_test, device, scale)

    realized_test = y_test.numpy().flatten() * scale
    lstm_mse      = vol_forecast_mse(preds, realized_test)
    garch_test    = cond_vol.values[WINDOW + n_train :]
    garch_mse     = vol_forecast_mse(garch_test, realized_test)

    print(f"\nVol forecast MSE")
    print(f"  GARCH(1,1) : {garch_mse:.2e}")
    print(f"  LSTM       : {lstm_mse:.2e}")
    print(f"  Ratio      : {lstm_mse/garch_mse:.3f}x "
          f"({'better' if lstm_mse < garch_mse else 'worse'} than GARCH)")

    # ── LSTM VaR + Kupiec ─────────────────────────────────────────────────────
    lstm_var, ret_test = lstm_var_series(preds, port_ret, n_train, WINDOW, CONFIDENCE)
    kup_lstm = kupiec_test_series(ret_test, lstm_var, CONFIDENCE)

    # GARCH on same test window for apples-to-apples
    garch_var_test = var_gn.reindex(test_dates[:len(preds)]).dropna()
    shared_idx     = lstm_var.index.intersection(garch_var_test.index)
    kup_garch_test = kupiec_test_series(
        port_ret.reindex(shared_idx),
        garch_var_test.reindex(shared_idx),
        CONFIDENCE)

    print(f"\n{'Method':<20} {'Mean VaR':>10} {'Max VaR':>10} "
          f"{'Violations':>12} {'Kupiec':>8}")
    print("-" * 65)
    for label, var_s, kup in [
        ("GARCH-Normal", garch_var_test.reindex(shared_idx), kup_garch_test),
        ("LSTM",         lstm_var.reindex(shared_idx),       kup_lstm),
    ]:
        print(f"{label:<20} {var_s.mean()*100:>9.3f}%  {var_s.max()*100:>9.3f}%  "
              f"{kup['violations']:>8} ({kup['violation_rate']:>6})  {kup['passes']:>6}")

    # ── 6. Final summary ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" FULL KUPIEC COMPARISON — all methods")
    print("="*60)
    import pandas as pd
    print(f"\n{'Method':<25} {'Violations':>10} {'Rate':>8} {'p-value':>9} {'Passes':>8}")
    print("-" * 65)
    for name, var in [
        ("Parametric",    pd.Series(var_p,  index=port_ret.index)),
        ("Historical",    pd.Series(var_h,  index=port_ret.index)),
        ("MC Normal",     pd.Series(var_mc, index=port_ret.index)),
        ("Student-t MC",  pd.Series(var_t,  index=port_ret.index)),
        ("GARCH Normal",  var_gn),
        ("GARCH Skewed-t",var_gskt),
        ("LSTM (test)",   lstm_var),
    ]:
        ret = port_ret if name != "LSTM (test)" else ret_test
        r = kupiec_test_series(ret, var, CONFIDENCE)
        print(f"{name:<25} {r['violations']:>10} {r['violation_rate']:>8} "
              f"{r['p-value']:>9} {r['passes']:>8}")


if __name__ == "__main__":
    main()
