import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats


# ── Architecture ──────────────────────────────────────────────────────────────

class VolatilityLSTM(nn.Module):
    """
    2-layer LSTM → FC head (64 → 32 → 1) with Softplus output.
    Softplus enforces σ > 0 while remaining smooth and differentiable.
    """
    def __init__(self, input_size=1, hidden_size=64,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, dropout=dropout,
                            batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # last timestep only


# ── Data preparation ──────────────────────────────────────────────────────────

def build_sequences(port_ret, window=20, train_frac=0.70, ewm_span=10):
    """
    Build LSTM input sequences and EWM-smoothed vol targets.

    Input  : |r_{t-window}|, ..., |r_{t-1}|   (window absolute returns)
    Target : EWM(span=ewm_span) of |r_t|       (smoothed realized vol)

    Both are normalized by training-set std to order-1 scale.

    Returns
    -------
    X_train, X_test  : torch.Tensor  (n, window, 1)
    y_train, y_test  : torch.Tensor  (n, 1)
    test_dates       : DatetimeIndex
    scale            : float  (multiply predictions back to decimal vol units)
    """
    ret_vals = port_ret.values.astype("float32")
    ewm_vol  = port_ret.abs().ewm(span=ewm_span).mean().values.astype("float32")

    X_list, y_list = [], []
    for i in range(window, len(ret_vals)):
        X_list.append(np.abs(ret_vals[i - window : i]))
        y_list.append(ewm_vol[i])

    X_arr = np.array(X_list, dtype="float32")
    y_arr = np.array(y_list, dtype="float32")

    n       = len(X_arr)
    n_train = int(n * train_frac)

    # Normalize using training set only — no leakage into test
    scale = float(y_arr[:n_train].std())
    X_arr = X_arr / scale
    y_arr = y_arr / scale

    X = torch.tensor(X_arr).unsqueeze(-1)   # (n, window, 1)
    y = torch.tensor(y_arr).unsqueeze(-1)   # (n, 1)

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    test_dates = port_ret.index[window + n_train:]

    print(f"Sequences  total  : {n}")
    print(f"Train             : {n_train}  ({n_train/n:.0%})")
    print(f"Test              : {n - n_train}  ({(n-n_train)/n:.0%})")
    print(f"scale             : {scale:.6f}  (unscale preds by × this)")
    print(f"y_arr std (scaled): {y_arr.std():.4f}  (should be ~1.0)")

    return X_train, X_test, y_train, y_test, test_dates, scale, n_train


# ── Training ──────────────────────────────────────────────────────────────────

def train_lstm(model, X_train, y_train, X_test, y_test, device,
               epochs=100, lr=1e-3, patience=15, batch_size=32):
    """
    Train VolatilityLSTM with Adam, ReduceLROnPlateau, early stopping.

    Returns
    -------
    train_losses, val_losses : list of float (per epoch)
    """
    n_train = len(X_train)
    n_test  = len(X_test)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),
                              batch_size=64, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state    = None
    patience_ctr  = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= n_train

        model.eval()
        with torch.no_grad():
            val_loss = sum(
                criterion(model(xb.to(device)), yb.to(device)).item() * len(xb)
                for xb, yb in test_loader
            ) / n_test

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d} | train: {epoch_loss:.2e} | "
                  f"val: {val_loss:.2e} | lr: {lr_now:.1e}")

        if patience_ctr >= patience:
            print(f"\nEarly stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    print(f"Best val MSE: {best_val_loss:.6e}")
    return train_losses, val_losses


# ── Evaluation ────────────────────────────────────────────────────────────────

def predict(model, X_test, device, scale):
    """Return unscaled predictions (decimal vol units)."""
    model.eval()
    with torch.no_grad():
        return model(X_test.to(device)).cpu().numpy().flatten() * scale


def lstm_var_series(preds, port_ret, n_train, window, confidence=0.95):
    """
    Build daily VaR series from LSTM vol forecasts using normal quantile.
    Same formula as GARCH-Normal VaR — isolates vol-estimation difference.
    """
    z          = stats.norm.ppf(1 - confidence)
    ret_window = port_ret.iloc[window + n_train : window + n_train + len(preds)]
    mu_test    = ret_window.mean()
    test_dates = ret_window.index

    var_series = pd.Series(-(mu_test + z * preds), index=test_dates[:len(preds)])
    return var_series, ret_window


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
