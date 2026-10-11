"""Helper functions for the Bass diffusion homework.

Continuous-time Bass model (Bass, 1969):
    F(t) = (1 - exp(-(p+q) t)) / (1 + (q/p) exp(-(p+q) t))
Adopters in the interval (t0, t1] = M * (F(t1) - F(t0)).
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ----------------------------------------------------------------------------
# Loading Google Trends exports
# ----------------------------------------------------------------------------

def load_trends(path: str) -> pd.DataFrame:
    """Read a Google Trends CSV export (Month/Week/Day, value) into a DataFrame.

    Handles the 1-2 junk header lines Google adds, and the "<1" placeholder.
    Returns columns: date (datetime), interest (float, 0-100 index).
    """
    with open(path, encoding="utf-8-sig") as fh:
        lines = fh.read().splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.split(",")[0].strip().lower() in {"month", "week", "day", "time"}),
        0,
    )
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df = df.iloc[:, :2]
    df.columns = ["date", "interest"]
    df["date"] = pd.to_datetime(df["date"])
    df["interest"] = (
        df["interest"].astype(str).str.replace("<1", "0.5", regex=False).astype(float)
    )
    return df.sort_values("date").reset_index(drop=True)


def annualise(df: pd.DataFrame, first_year: int, last_year: int) -> pd.DataFrame:
    """Mean monthly interest per calendar year (removes seasonality)."""
    d = df.copy()
    d["year"] = d["date"].dt.year
    out = (
        d[(d.year >= first_year) & (d.year <= last_year)]
        .groupby("year", as_index=False)["interest"]
        .mean()
    )
    out["t"] = np.arange(1, len(out) + 1)  # t = 1 for the first year
    return out


# ----------------------------------------------------------------------------
# Bass model
# ----------------------------------------------------------------------------

def bass_cdf(t, p, q):
    """Cumulative adoption fraction F(t); t may be an array (in years)."""
    t = np.asarray(t, dtype=float)
    e = np.exp(-(p + q) * t)
    return (1 - e) / (1 + (q / p) * e)


def bass_flow(t_edges, p, q, M):
    """Adopters in each interval defined by consecutive edges in t_edges."""
    F = bass_cdf(t_edges, p, q)
    return M * np.diff(F)


def bass_peak(p, q, M):
    """Peak time t* (years) and peak adoption rate (per year)."""
    if q <= p:
        return 0.0, M * p  # no interior peak: monotonically decreasing
    t_star = np.log(q / p) / (p + q)
    peak = M * (p + q) ** 2 / (4 * q)
    return t_star, peak


def _model(t, p, q, M):
    # t holds the END of each year (1,2,...); the interval is (t-1, t]
    t = np.asarray(t, dtype=float)
    return M * (bass_cdf(t, p, q) - bass_cdf(t - 1, p, q))


def fit_bass_nls(t, y, n_starts: int = 60, seed: int = 0):
    """Nonlinear least squares fit of (p, q, M) to per-period adoption flow y.

    Uses many random starting points and keeps the best fit (the Bass
    likelihood surface is flat in places).  Returns dict with p, q, M, sse, r2.
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    total = y.sum()
    bounds = ([1e-5, 1e-3, total * 0.8], [1.0, 5.0, total * 100])
    best = None
    for _ in range(n_starts):
        p0 = 10 ** rng.uniform(-4, -0.7)
        q0 = 10 ** rng.uniform(-2, 0.5)
        M0 = total * rng.uniform(1.0, 5.0)
        try:
            popt, _ = curve_fit(_model, t, y, p0=[p0, q0, M0], bounds=bounds, maxfev=20000)
        except Exception:
            continue
        sse = float(np.sum((y - _model(t, *popt)) ** 2))
        if best is None or sse < best["sse"]:
            best = {"p": popt[0], "q": popt[1], "M": popt[2], "sse": sse}
    if best is None:
        raise RuntimeError("Bass NLS did not converge from any starting point")
    sst = float(np.sum((y - y.mean()) ** 2))
    best["r2"] = 1 - best["sse"] / sst
    return best


def fit_bass_ols(y):
    """Classic Bass (1969) regression: S_t = a + b N_{t-1} + c N_{t-1}^2.

    Returns dict with p, q, M (or None values if the quadratic has no valid root).
    """
    y = np.asarray(y, float)
    N_prev = np.concatenate([[0.0], np.cumsum(y)[:-1]])
    X = np.column_stack([np.ones_like(N_prev), N_prev, N_prev ** 2])
    (a, b, c), *_ = np.linalg.lstsq(X, y, rcond=None)
    disc = b * b - 4 * a * c
    out = {"a": a, "b": b, "c": c, "p": None, "q": None, "M": None}
    if c < 0 and a > 0 and disc >= 0:
        M = (-b - np.sqrt(disc)) / (2 * c)
        out.update(M=M, p=a / M, q=-M * c)
    return out


def bootstrap_bass(t, y, fit, n_boot: int = 300, seed: int = 1):
    """Residual bootstrap of the NLS fit; returns DataFrame of p, q, M draws."""
    rng = np.random.default_rng(seed)
    t, y = np.asarray(t, float), np.asarray(y, float)
    fitted = _model(t, fit["p"], fit["q"], fit["M"])
    resid = y - fitted
    rows = []
    for _ in range(n_boot):
        y_b = np.clip(fitted + rng.choice(resid, size=len(resid), replace=True), 1e-6, None)
        try:
            r = fit_bass_nls(t, y_b, n_starts=8, seed=int(rng.integers(1e9)))
            rows.append((r["p"], r["q"], r["M"]))
        except RuntimeError:
            continue
    return pd.DataFrame(rows, columns=["p", "q", "M"])


def adopters_table(p, q, M, n_years: int, per_year: int = 1) -> pd.DataFrame:
    """Adopters per period measured from launch.

    per_year=1 -> yearly, per_year=4 -> quarterly.
    """
    edges = np.arange(0, n_years * per_year + 1) / per_year
    flow = bass_flow(edges, p, q, M)
    out = pd.DataFrame({"period": np.arange(1, len(flow) + 1), "adopters": flow})
    out["cumulative"] = out["adopters"].cumsum()
    out["penetration_pct"] = 100 * out["cumulative"] / M
    return out
