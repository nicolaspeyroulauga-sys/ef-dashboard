"""
Empirical Finance - Multi-Asset Dashboard
=========================================
Single-file Streamlit app.  Edit the CONFIGURATION block (section 1) to change
the asset universe, the title and the defaults.  Everything else is generic.

Data sources
  * Yahoo Finance (live, via yfinance)
  * Uploaded CSV / Excel file (e.g. a Bloomberg export: first column = dates,
    one column per asset)
  * Optional fallback snapshot: data/prices.csv in the repository
"""

import hashlib
import io
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yfinance as yf
from scipy import optimize, stats

pio.templates.default = "plotly_white"

st.set_page_config(
    page_title="Empirical Finance Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 1. CONFIGURATION  -  EDIT THIS SECTION
# =============================================================================
APP_TITLE = "Multi-Asset Risk & Return Dashboard"
APP_SUBTITLE = "Empirical Finance  |  MSc Wealth Management  |  University of Luxembourg"

# Earliest date downloaded.  Keep it early enough for the stress tests (GFC = 2007).
HISTORY_START = "2005-01-01"

# asset class -> {display name: Yahoo Finance ticker}
# >>> REPLACE / EXTEND WITH YOUR GROUP'S ASSETS <<<
UNIVERSE = {
    "Equities": {
        "S&P 500": "^GSPC",
        "Nasdaq 100": "^NDX",
        "Euro Stoxx 50": "^STOXX50E",
        "DAX": "^GDAXI",
        "CAC 40": "^FCHI",
        "FTSE 100": "^FTSE",
        "Nikkei 225": "^N225",
        "MSCI World (IWDA, EUR)": "IWDA.AS",
        "MSCI Emerging Markets (EEM)": "EEM",
    },
    "Fixed income": {
        "US Aggregate Bonds (AGG)": "AGG",
        "US Treasuries 7-10Y (IEF)": "IEF",
        "US Treasuries 20Y+ (TLT)": "TLT",
        "US Investment Grade (LQD)": "LQD",
        "US High Yield (HYG)": "HYG",
    },
    # Gold is held through an exchange-traded product (the module treats commodities,
    # cash and real estate as out of scope, so there is no direct commodity or REIT exposure).
    "Alternatives (ETP)": {
        "Gold (GLD)": "GLD",
    },
}

# Only these asset classes count towards the "equity share" used by the risk profiles
EQUITY_CLASS = "Equities"

DEFAULT_SELECTION = [
    "S&P 500",
    "Euro Stoxx 50",
    "MSCI World (IWDA, EUR)",
    "MSCI Emerging Markets (EEM)",
    "US Aggregate Bonds (AGG)",
    "US Treasuries 7-10Y (IEF)",
    "US Investment Grade (LQD)",
    "US High Yield (HYG)",
    "Gold (GLD)",
]
DEFAULT_BENCHMARK = "Euro Stoxx 50"

# Risk profiles: maximum equity share, volatility limit (annual).  Matches the Excel model.
RISK_PROFILES = {
    "None (no profile constraint)": (1.0, None),
    "Conservative (equity <= 40%, vol <= 7%)": (0.40, 0.07),
    "Balanced (equity <= 60%, vol <= 10%)": (0.60, 0.10),
    "Growth (equity <= 80%, vol <= 14%)": (0.80, 0.14),
}
DEFAULT_RF = 2.0  # annual risk-free rate in %

# Approximate peak-to-trough windows of the S&P 500 (used in the stress-test tab)
CRISES = {
    "Global Financial Crisis (Oct 2007 - Mar 2009)": ("2007-10-09", "2009-03-09"),
    "Eurozone debt crisis (Apr - Oct 2011)": ("2011-04-29", "2011-10-03"),
    "China / oil sell-off (Aug 2015 - Feb 2016)": ("2015-08-18", "2016-02-11"),
    "Q4 2018 sell-off (Sep - Dec 2018)": ("2018-09-20", "2018-12-24"),
    "COVID-19 crash (Feb - Mar 2020)": ("2020-02-19", "2020-03-23"),
    "Inflation & rate shock (Jan - Oct 2022)": ("2022-01-03", "2022-10-12"),
    "Tariff sell-off (Feb - Apr 2025)": ("2025-02-19", "2025-04-08"),
}

# Default hypothetical shock per asset class for the scenario tool (in %)
DEFAULT_SHOCKS = {
    "Equities": -25.0,
    "Fixed income": -3.0,
    "Alternatives (ETP)": 5.0,
    "Currencies": 0.0,
    "Other": -10.0,
}

# =============================================================================
# 2. CONSTANTS / STYLING
# =============================================================================
FREQ_PERIODS = {"Daily": 252, "Weekly": 52, "Monthly": 12}
DEFAULT_WINDOW = {"Daily": 63, "Weekly": 26, "Monthly": 12}
REBALANCE_OPTIONS = ["Every period", "Monthly", "Quarterly", "Annually", "Never (buy & hold)"]
PALETTE = [
    "#0B3D91", "#E4572E", "#17BEBB", "#FFC914", "#76B041", "#6A4C93",
    "#F28F3B", "#3A86FF", "#8D99AE", "#D00000", "#2E7D32", "#8338EC",
    "#00798C", "#B5838D", "#6D6875", "#4361EE",
]
BENCH_COLOR = "#1B1F2A"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

ALL_TICKERS = {name: tkr for cls in UNIVERSE.values() for name, tkr in cls.items()}
CLASS_OF = {name: cls for cls, items in UNIVERSE.items() for name in items}

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}
    div[data-testid="stMetric"] {
        background: #F3F6FB; border: 1px solid #E1E7F0;
        padding: 12px 16px; border-radius: 10px;
    }
    h1 {font-weight: 700; letter-spacing: -0.5px; margin-bottom: 0;}
    h2, h3 {font-weight: 650;}
    .subtitle {color: #5B6475; font-size: 0.95rem; margin-bottom: 0.8rem;}
    button[data-baseweb="tab"] {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 3. DATA LAYER
# =============================================================================
@st.cache_data(ttl=3600, show_spinner="Downloading market data from Yahoo Finance...")
def download_prices(tickers: tuple, start: str) -> pd.DataFrame:
    """Adjusted close prices for the given Yahoo tickers (cached for 1 hour)."""
    raw = yf.download(list(tickers), start=start, auto_adjust=True, progress=False, threads=True)
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.dropna(how="all").sort_index()


def parse_uploaded(file) -> pd.DataFrame:
    """Read an uploaded CSV/Excel file: first column = dates, other columns = assets."""
    if file.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file, index_col=0)
    else:
        df = pd.read_csv(file, sep=None, engine="python", index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def is_monthly_data(prices: pd.DataFrame) -> bool:
    """True if the observations are about one per month (e.g. Bloomberg month-end export)."""
    gap = prices.index.to_series().diff().median()
    return bool(pd.notna(gap) and gap > pd.Timedelta(days=20))


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Align calendars: forward-fill short gaps, drop weekends and empty rows (daily data only)."""
    prices = prices.sort_index()
    if is_monthly_data(prices):
        return prices.dropna(how="all")
    prices = prices.ffill(limit=5)
    prices = prices[prices.index.dayofweek < 5]
    return prices.dropna(how="all")


@st.cache_data(ttl=24 * 3600, show_spinner="Downloading Fama-French factors...")
def load_ff3_daily() -> pd.DataFrame:
    """Daily Fama-French 3 factors (US) from Kenneth French's data library, in decimals."""
    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_Factors_daily_CSV.zip"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        text = zf.read(zf.namelist()[0]).decode("latin-1").splitlines()
    header_idx = next(i for i, line in enumerate(text) if "Mkt-RF" in line)
    rows = []
    for line in text[header_idx + 1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5 or not parts[0].isdigit():
            break
        rows.append(parts[:5])
    ff = pd.DataFrame(rows, columns=["Date", "Mkt-RF", "SMB", "HML", "RF"])
    ff["Date"] = pd.to_datetime(ff["Date"], format="%Y%m%d")
    return ff.set_index("Date").astype(float) / 100.0


# =============================================================================
# 4. ANALYTICS
# =============================================================================
def resample_prices(prices: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "Daily":
        return prices
    if freq == "Weekly":
        out = prices.resample("W-FRI").last()
    else:
        try:
            out = prices.resample("ME").last()
        except ValueError:  # older pandas
            out = prices.resample("M").last()
    return out.dropna(how="all")


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    r = prices.div(prices.shift(1)) - 1
    return r.replace([np.inf, -np.inf], np.nan).iloc[1:]


def drawdown_series(r):
    wealth = (1 + r.fillna(0)).cumprod()
    return wealth / wealth.cummax() - 1


def asset_stats(r: pd.Series, b, rf: float, ann: int, conf: float) -> pd.Series:
    """Full set of performance / risk statistics for one return series."""
    r = r.dropna()
    if len(r) < 5:
        return pd.Series(dtype=float)
    rf_p = rf / ann
    years = len(r) / ann
    total = (1 + r).prod() - 1
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    vol = r.std() * np.sqrt(ann)
    ex = r - rf_p
    sharpe = ex.mean() * ann / vol if vol > 0 else np.nan
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(ann)
    sortino = ex.mean() * ann / downside if downside > 0 else np.nan
    mdd = drawdown_series(r).min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    q = r.quantile(1 - conf)
    out = {
        "Obs.": len(r),
        "Total return": total,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max drawdown": mdd,
        "Calmar": calmar,
        f"VaR {conf:.0%}": -q,
        f"CVaR {conf:.0%}": -r[r <= q].mean(),
        "Skewness": r.skew(),
        "Excess kurtosis": r.kurt(),
        "Hit ratio": (r > 0).mean(),
        "Best period": r.max(),
        "Worst period": r.min(),
        "Beta": np.nan,
        "Alpha (ann.)": np.nan,
        "Correlation": np.nan,
        "Tracking error": np.nan,
        "Information ratio": np.nan,
    }
    if b is not None:
        j = pd.concat([r, b], axis=1, join="inner").dropna()
        if len(j) >= 5:
            rr, bb = j.iloc[:, 0], j.iloc[:, 1]
            var_b = bb.var()
            beta = rr.cov(bb) / var_b if var_b > 0 else np.nan
            te = (rr - bb).std() * np.sqrt(ann)
            out["Beta"] = beta
            out["Alpha (ann.)"] = ((rr - rf_p).mean() - beta * (bb - rf_p).mean()) * ann
            out["Correlation"] = rr.corr(bb)
            out["Tracking error"] = te
            out["Information ratio"] = (rr - bb).mean() * ann / te if te > 0 else np.nan
    return pd.Series(out)


def build_stats(rets: pd.DataFrame, bench_name, rf, ann, conf, bench_series=None) -> pd.DataFrame:
    if bench_series is None and bench_name in rets.columns:
        bench_series = rets[bench_name]
    rows = {c: asset_stats(rets[c], bench_series, rf, ann, conf) for c in rets.columns}
    df = pd.DataFrame(rows).T
    df.index.name = "Asset"
    return df


PCT_COLS = {
    "Total return", "CAGR", "Volatility", "Max drawdown", "Best period", "Worst period",
    "Hit ratio", "Alpha (ann.)", "Tracking error",
}


def style_stats(df: pd.DataFrame):
    fmt = {}
    for c in df.columns:
        if c in PCT_COLS or str(c).startswith(("VaR", "CVaR")):
            fmt[c] = "{:.1%}"
        elif c == "Obs.":
            fmt[c] = "{:.0f}"
        else:
            fmt[c] = "{:.2f}"
    return df.style.format(fmt, na_rep="-")


def drawdown_episodes(r: pd.Series, top: int = 5) -> pd.DataFrame:
    r = r.dropna()
    if r.empty:
        return pd.DataFrame()
    dd = drawdown_series(r)
    idx = dd.index
    in_dd = dd < 0
    grp = (~in_dd).cumsum()
    rows = []
    for _, seg in dd[in_dd].groupby(grp[in_dd]):
        start_pos = idx.get_loc(seg.index[0])
        last_pos = idx.get_loc(seg.index[-1])
        peak = idx[start_pos - 1] if start_pos > 0 else idx[0]
        trough = seg.idxmin()
        rec = idx[last_pos + 1] if last_pos + 1 < len(idx) else pd.NaT
        rows.append({
            "Peak": peak,
            "Trough": trough,
            "Recovery": rec,
            "Depth": seg.min(),
            "Periods to trough": idx.get_loc(trough) - idx.get_loc(peak),
            "Periods to recover": (idx.get_loc(rec) - idx.get_loc(trough)) if pd.notna(rec) else np.nan,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("Depth").head(top).reset_index(drop=True)
    for c in ("Peak", "Trough", "Recovery"):
        df[c] = df[c].dt.strftime("%Y-%m-%d").fillna("ongoing")
    return df


def trailing_returns(prices: pd.DataFrame):
    """Trailing returns from full daily history (independent of the sidebar period)."""
    ret_rows, info_rows = {}, {}
    for c in prices.columns:
        s = prices[c].dropna()
        if len(s) < 3:
            continue
        t, last = s.index[-1], s.iloc[-1]

        def tot(ref_date):
            ref = s.loc[:ref_date]
            return last / ref.iloc[-1] - 1 if len(ref) else np.nan

        def ann_ret(years):
            ref = s.loc[: t - pd.DateOffset(years=years)]
            return (last / ref.iloc[-1]) ** (1 / years) - 1 if len(ref) else np.nan

        ret_rows[c] = {
            "1D": last / s.iloc[-2] - 1,
            "1W": tot(t - pd.Timedelta(days=7)),
            "1M": tot(t - pd.DateOffset(months=1)),
            "3M": tot(t - pd.DateOffset(months=3)),
            "6M": tot(t - pd.DateOffset(months=6)),
            "YTD": tot(pd.Timestamp(t.year - 1, 12, 31)),
            "1Y": tot(t - pd.DateOffset(years=1)),
            "3Y (ann.)": ann_ret(3),
            "5Y (ann.)": ann_ret(5),
        }
        info_rows[c] = {"Last price": last, "As of": str(t.date())}
    return pd.DataFrame(ret_rows).T, pd.DataFrame(info_rows).T


def window_perf(s: pd.Series, a, b, tol: int = 7):
    """Total return and max drawdown of series s between dates a and b (NaN if no data).
    `tol` = how many calendar days the nearest observation may be from a / b."""
    s = s.dropna()
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    base = s.loc[:a]
    if base.empty or (a - base.index[-1]).days > tol:
        return np.nan, np.nan
    end = s.loc[:b]
    if end.empty or (b - end.index[-1]).days > tol or end.index[-1] <= base.index[-1]:
        return np.nan, np.nan
    seg = s.loc[base.index[-1]: end.index[-1]]
    return seg.iloc[-1] / seg.iloc[0] - 1, (seg / seg.cummax() - 1).min()


# ---- regression ------------------------------------------------------------
def newey_west_cov(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    Xe = X * resid[:, None]
    S = Xe.T @ Xe
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = Xe[l:].T @ Xe[:-l]
        S = S + w * (G + G.T)
    xtx_inv = np.linalg.inv(X.T @ X)
    return xtx_inv @ S @ xtx_inv


def ols(y: np.ndarray, Xdf: pd.DataFrame, hac_lags: int = 0) -> dict:
    """OLS with intercept; classical or Newey-West standard errors."""
    X = np.column_stack([np.ones(len(Xdf)), Xdf.values])
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    if hac_lags > 0:
        cov = newey_west_cov(X, resid, hac_lags)
        se = np.sqrt(np.diag(cov))
        t = beta / se
        p = 2 * (1 - stats.norm.cdf(np.abs(t)))
    else:
        s2 = resid @ resid / (n - k)
        se = np.sqrt(np.diag(s2 * xtx_inv))
        t = beta / se
        p = 2 * (1 - stats.t.cdf(np.abs(t), n - k))
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / tss if tss > 0 else np.nan
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    table = pd.DataFrame(
        {"Coefficient": beta, "Std. error": se, "t-stat": t, "p-value": p},
        index=["Alpha (per period)"] + list(Xdf.columns),
    )
    return {"table": table, "r2": r2, "adj_r2": adj, "n": n, "resid": resid}


# ---- portfolio -------------------------------------------------------------
def portfolio_returns(rets: pd.DataFrame, w, rebalance: str) -> pd.Series:
    rets = rets.dropna()
    w = pd.Series(np.asarray(w, dtype=float), index=rets.columns)
    if rebalance == "Every period":
        return rets @ w
    if rebalance == "Never (buy & hold)":
        chunks = [rets]
    elif rebalance == "Monthly":
        chunks = [c for _, c in rets.groupby([rets.index.year, rets.index.month])]
    elif rebalance == "Quarterly":
        chunks = [c for _, c in rets.groupby([rets.index.year, rets.index.quarter])]
    else:
        chunks = [c for _, c in rets.groupby(rets.index.year)]
    pieces = []
    for chunk in chunks:
        g = (1 + chunk).cumprod()
        v = g.mul(w, axis=1).sum(axis=1)
        r = v.pct_change()
        r.iloc[0] = v.iloc[0] - 1
        pieces.append(r)
    return pd.concat(pieces)


def _solve(obj, n, cap, extra=None):
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if extra:
        cons += extra
    res = optimize.minimize(
        obj, np.full(n, 1.0 / n), method="SLSQP", bounds=[(0, cap)] * n,
        constraints=cons, options={"maxiter": 500, "ftol": 1e-10},
    )
    if not res.success:
        return None
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def _profile_constraints(eq_mask: np.ndarray, eq_cap: float, vol_cap, cov: np.ndarray):
    """Risk-profile constraints: equity share <= eq_cap and (optionally) volatility <= vol_cap."""
    cons = []
    if eq_cap < 0.999 and eq_mask.any():
        cons.append({"type": "ineq", "fun": lambda w: eq_cap - eq_mask @ w})
    if vol_cap is not None:
        cons.append({"type": "ineq", "fun": lambda w: vol_cap ** 2 - w @ cov @ w})
    return cons


@st.cache_data(show_spinner=False)
def optimise_portfolios(mu: np.ndarray, cov: np.ndarray, rf: float, cap: float,
                        eq_mask: np.ndarray, eq_cap: float, vol_cap=None):
    n = len(mu)
    cap = max(cap, 1.0 / n + 1e-6)
    prof = _profile_constraints(eq_mask, eq_cap, vol_cap, cov)
    out = {
        "Min variance": _solve(lambda w: w @ cov @ w, n, cap, prof),
        "Max Sharpe": _solve(lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w), n, cap, prof),
        "Risk parity": _solve(
            lambda w: np.sum((w * (cov @ w) / (w @ cov @ w) - 1.0 / n) ** 2) * 1e3, n, cap, prof
        ),
    }
    return {k: v for k, v in out.items() if v is not None}


@st.cache_data(show_spinner=False)
def efficient_frontier(mu: np.ndarray, cov: np.ndarray, cap: float, eq_mask: np.ndarray, eq_cap: float,
                       vol_cap=None, n_points: int = 40):
    n = len(mu)
    cap = max(cap, 1.0 / n + 1e-6)
    prof = _profile_constraints(eq_mask, eq_cap, None, cov)  # frontier shown without the vol limit
    w_lo = _solve(lambda w: w @ cov @ w, n, cap, prof)
    w_hi = _solve(lambda w: -(w @ mu), n, cap, prof)
    if w_lo is None or w_hi is None:
        return np.array([]), np.array([])
    vols, rets = [], []
    for target in np.linspace(w_lo @ mu, w_hi @ mu, n_points):
        w = _solve(lambda x: x @ cov @ x, n, cap,
                   prof + [{"type": "eq", "fun": lambda x, t=target: x @ mu - t}])
        if w is not None:
            vols.append(np.sqrt(w @ cov @ w))
            rets.append(w @ mu)
    return np.array(vols), np.array(rets)


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    var = w @ cov @ w
    return w * (cov @ w) / var if var > 0 else np.zeros_like(w)


def bs_put(S, K, T, r, q, sigma):
    """Black-Scholes price of a European put with continuous dividend yield q.
    Works on scalars or numpy arrays; at T <= 0 returns the intrinsic value."""
    S, K, T, r, q, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    T_safe = np.maximum(T, 1e-9)
    sig_safe = np.maximum(sigma, 1e-9)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig_safe ** 2) * T_safe) / (sig_safe * np.sqrt(T_safe))
    d2 = d1 - sig_safe * np.sqrt(T_safe)
    price = K * np.exp(-r * T_safe) * stats.norm.cdf(-d2) - S * np.exp(-q * T_safe) * stats.norm.cdf(-d1)
    return np.where(T <= 0, np.maximum(K - S, 0.0), price)


def bs_put_delta(S, K, T, r, q, sigma):
    T_safe, sig_safe = max(float(T), 1e-9), max(float(sigma), 1e-9)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig_safe ** 2) * T_safe) / (sig_safe * np.sqrt(T_safe))
    return float(-np.exp(-q * T_safe) * stats.norm.cdf(-d1))


def build_excel(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31])
    return buf.getvalue()


# =============================================================================
# 5. PLOT HELPERS
# =============================================================================
def line_fig(df: pd.DataFrame, title="", yfmt=None, height=440, log=False, colors=None, bench=None):
    fig = px.line(df, color_discrete_map=colors or {})
    fig.update_layout(
        title=title, height=height, legend_title_text="", hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    if log:
        fig.update_yaxes(type="log")
    if bench is not None:
        fig.update_traces(line=dict(color=BENCH_COLOR, dash="dash", width=2), selector=dict(name=bench))
    return fig


def heatmap_fig(df: pd.DataFrame, title="", fmt=".2f", scale="RdBu_r", zmin=None, zmax=None, mid=None, height=None):
    kwargs = dict(text_auto=fmt, color_continuous_scale=scale, aspect="auto")
    if zmin is not None:
        kwargs.update(zmin=zmin, zmax=zmax)
    if mid is not None:
        kwargs.update(color_continuous_midpoint=mid)
    fig = px.imshow(df, **kwargs)
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=50, b=10))
    if height:
        fig.update_layout(height=height)
    return fig


def show(fig, key=None):
    st.plotly_chart(fig, key=key)


# =============================================================================
# 6. SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### Controls")
    source = st.radio(
        "Data source",
        ["Yahoo Finance (live)", "Upload file (e.g. Bloomberg export)"],
        key="source",
    )

    uploaded = None
    extra_map = {}
    if source.startswith("Upload"):
        uploaded = st.file_uploader(
            "CSV or Excel: first column = dates, one column per asset",
            type=["csv", "xlsx", "xls"],
            key="upload",
        )
        available = {}
    else:
        classes = st.multiselect("Asset classes", list(UNIVERSE), default=list(UNIVERSE), key="classes")
        available = {n: t for c in classes for n, t in UNIVERSE[c].items()}
        extra_txt = st.text_input(
            "Extra Yahoo tickers (comma-separated)", "", key="extra",
            help="Example: NVDA, ASML.AS, SAP.DE. Use .AS / .PA / .DE / .L suffixes for European listings.",
        )
        for t in [x.strip().upper() for x in extra_txt.split(",") if x.strip()]:
            extra_map[t] = t
        available.update(extra_map)

    st.divider()
    st.markdown("### Analysis settings")
    preset = st.selectbox("Period", ["1Y", "3Y", "5Y", "10Y", "YTD", "Max", "Custom"], index=2, key="preset")
    custom_range = None
    if preset == "Custom":
        custom_range = st.date_input("Custom range", value=(date(2019, 1, 1), date.today()), key="custom")
    freq = st.radio("Return frequency", list(FREQ_PERIODS), index=2, horizontal=True, key="freq")
    rf_pct = st.number_input("Risk-free rate (annual, %)", 0.0, 15.0, DEFAULT_RF, 0.25, key="rf")
    conf = st.select_slider("VaR confidence level", options=[0.90, 0.95, 0.975, 0.99], value=0.95, key="conf",
                            format_func=lambda x: f"{x:.1%}")
    win = st.number_input("Rolling window (periods)", 5, 500, DEFAULT_WINDOW[freq], key=f"win_{freq}")
    log_scale = st.checkbox("Log scale for performance chart", value=False, key="log")

    st.divider()
    if st.button("Refresh data", key="refresh"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Market data is cached for 1 hour.")

rf = rf_pct / 100.0
ann = FREQ_PERIODS[freq]

# =============================================================================
# 7. LOAD DATA
# =============================================================================
notes = []

if source.startswith("Upload"):
    if uploaded is None:
        st.title(APP_TITLE)
        st.markdown(f"<div class='subtitle'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)
        st.info("Upload a CSV or Excel file in the sidebar (dates in the first column, one price column per asset).")
        st.stop()
    try:
        prices_full = clean_prices(parse_uploaded(uploaded))
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()
    available = {c: c for c in prices_full.columns}
    sel = st.sidebar.multiselect("Assets", list(available), default=list(available)[:8], key="assets_up")
    bench = st.sidebar.selectbox("Benchmark", list(available), index=0, key="bench_up")
    data_label = f"Uploaded file: {uploaded.name}"
else:
    sel = st.sidebar.multiselect(
        "Assets", list(available),
        default=[d for d in DEFAULT_SELECTION if d in available] or list(available)[:5],
        key="assets_yf",
    )
    bench_options = list(dict.fromkeys(list(ALL_TICKERS) + list(extra_map)))
    bench = st.sidebar.selectbox(
        "Benchmark / market proxy", bench_options,
        index=bench_options.index(DEFAULT_BENCHMARK) if DEFAULT_BENCHMARK in bench_options else 0,
        key="bench_yf",
    )
    name_to_ticker = {n: available.get(n, ALL_TICKERS.get(n)) for n in dict.fromkeys(sel + [bench])}
    name_to_ticker = {n: t for n, t in name_to_ticker.items() if t}
    if not sel:
        st.title(APP_TITLE)
        st.info("Select at least one asset in the sidebar.")
        st.stop()
    tickers = tuple(sorted(set(name_to_ticker.values())))
    raw = pd.DataFrame()
    try:
        raw = download_prices(tickers, HISTORY_START)
    except Exception as exc:
        notes.append(f"Yahoo Finance download failed: {exc}")
    if raw.empty:
        fallback = Path("data/prices.csv")
        if fallback.exists():
            prices_full = clean_prices(pd.read_csv(fallback, index_col=0, parse_dates=True))
            notes.append("Using the saved snapshot data/prices.csv because Yahoo Finance was unavailable.")
            data_label = "Saved snapshot (data/prices.csv)"
        else:
            st.error("No data could be downloaded and no fallback snapshot exists. Try again in a minute, or use "
                     "the file-upload option.")
            for n in notes:
                st.caption(n)
            st.stop()
    else:
        t2n = {t: n for n, t in name_to_ticker.items()}
        prices_full = clean_prices(raw.rename(columns=t2n))
        data_label = "Yahoo Finance (adjusted close)"
        missing = [n for n in name_to_ticker if n not in prices_full.columns]
        if missing:
            notes.append("No data returned for: " + ", ".join(missing))

sel = [s for s in sel if s in prices_full.columns]
if bench not in prices_full.columns:
    notes.append(f"Benchmark '{bench}' has no data; benchmark-relative statistics are disabled.")
    bench_ok = False
else:
    bench_ok = True
if not sel:
    st.error("None of the selected assets returned data.")
    st.stop()

names_all = sel + ([bench] if bench_ok and bench not in sel else [])
prices_full = prices_full[names_all].dropna(how="all")
COLOR = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names_all)}
if bench_ok:
    COLOR[bench] = BENCH_COLOR

# native data frequency (a Bloomberg month-end export is monthly)
native_monthly = is_monthly_data(prices_full)
if native_monthly and freq != "Monthly":
    freq, ann = "Monthly", 12
    notes.append("The data is monthly, so the dashboard uses monthly returns (annualisation factor 12).")
PPY = 12 if native_monthly else 252       # observations per year in the raw data
TOL = 35 if native_monthly else 7         # date-matching tolerance (days) for stress windows

# analysis window
first_d, last_d = prices_full.index[0], prices_full.index[-1]
if preset == "Max":
    start_d, end_d = first_d, last_d
elif preset == "YTD":
    start_d, end_d = pd.Timestamp(last_d.year, 1, 1), last_d
elif preset == "Custom" and custom_range is not None and len(custom_range) == 2:
    start_d, end_d = pd.Timestamp(custom_range[0]), pd.Timestamp(custom_range[1])
elif preset == "Custom":
    start_d, end_d = last_d - pd.DateOffset(years=5), last_d
else:
    start_d, end_d = last_d - pd.DateOffset(years=int(preset[:-1])), last_d

p_daily = prices_full.loc[start_d:end_d]
p_freq = resample_prices(p_daily, freq)
rets_daily = to_returns(p_daily)
rets_all = to_returns(p_freq)

if len(rets_all) < 20:
    st.error("Fewer than 20 return observations in the selected window. Extend the period or use a higher frequency.")
    st.stop()

rets_sel = rets_all[sel]
bench_ret = rets_all[bench] if bench_ok else None
common = rets_sel.dropna()
multi = len(sel) >= 2 and len(common) >= 30
if len(sel) >= 2 and len(common) < 30:
    short = rets_sel.apply(lambda s: s.first_valid_index()).sort_values().index[-1]
    notes.append(f"Selected assets overlap for fewer than 30 periods (shortest history: {short}). "
                 "Portfolio and optimisation tools are disabled; pick assets with longer histories.")

# =============================================================================
# 8. HEADER
# =============================================================================
st.title(APP_TITLE)
st.markdown(f"<div class='subtitle'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)
h1, h2, h3, h4 = st.columns(4)
h1.metric("Data source", data_label.split(" (")[0])
h2.metric("Last data point", str(last_d.date()))
h3.metric("Analysis window", f"{p_daily.index[0].date()} to {p_daily.index[-1].date()}")
h4.metric("Frequency / observations", f"{freq} / {len(rets_all)}")
for n in notes:
    st.warning(n)

stats_df = build_stats(rets_all, bench if bench_ok else None, rf, ann, conf)
corr_df = rets_sel.corr()

tab_names = [
    "Overview", "Risk & Return", "Correlation", "Distribution", "Regression & Factors",
    "Portfolio", "Seasonality & Rolling", "Stress Tests", "Hedging Overlays", "Data & Methodology",
]
tabs = st.tabs(tab_names)

# -----------------------------------------------------------------------------
# TAB 1 - OVERVIEW
# -----------------------------------------------------------------------------
with tabs[0]:
    sel_stats = stats_df.loc[sel]
    best = sel_stats["Total return"].idxmax()
    worst = sel_stats["Total return"].idxmin()
    avg_corr = (
        corr_df.values[np.triu_indices(len(sel), 1)].mean() if len(sel) >= 2 else np.nan
    )
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Assets analysed", len(sel))
    if bench_ok:
        k2.metric(f"Benchmark ({bench})", f"{stats_df.loc[bench, 'Total return']:.1%}",
                  f"Sharpe {stats_df.loc[bench, 'Sharpe']:.2f}", delta_color="off")
    else:
        k2.metric("Benchmark", "n/a")
    k3.metric("Best performer", best, f"{sel_stats.loc[best, 'Total return']:.1%}")
    k4.metric("Worst performer", worst, f"{sel_stats.loc[worst, 'Total return']:.1%}")
    k5.metric("Average pairwise correlation", "-" if np.isnan(avg_corr) else f"{avg_corr:.2f}")

    st.subheader("Cumulative performance")
    rebase_common = st.checkbox("Rebase all assets to the same start date", value=True, key="rebase")
    pf = p_freq[names_all]
    if rebase_common and not pf.dropna().empty:
        base_idx = pf.dropna().index[0]
        rebased = pf.loc[base_idx:].div(pf.loc[base_idx]) * 100
    else:
        rebased = pf.div(pf.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)) * 100
    show(line_fig(rebased, "Rebased to 100", colors=COLOR, bench=bench if bench_ok else None, log=log_scale),
         key="ov_perf")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Trailing returns")
        trail, info = trailing_returns(prices_full[names_all])
        if native_monthly:
            trail = trail.drop(columns=["1D", "1W"])
        st.caption("Calculated from the full downloaded history, independent of the period chosen in the sidebar.")
        show(heatmap_fig(trail * 100, fmt=".1f", scale="RdYlGn", mid=0,
                         height=max(300, 42 * len(trail) + 80)), key="ov_trail")
    with c2:
        st.subheader("Total return over the window")
        tr = stats_df["Total return"].sort_values()
        fig = go.Figure(go.Bar(x=tr.values, y=tr.index, orientation="h",
                               marker_color=[COLOR.get(i, "#888") for i in tr.index],
                               text=[f"{v:.1%}" for v in tr.values], textposition="outside"))
        fig.update_layout(height=max(300, 42 * len(tr) + 80), margin=dict(l=10, r=40, t=10, b=10))
        fig.update_xaxes(tickformat=".0%")
        show(fig, key="ov_bar")

    st.subheader("Latest prices")
    latest_cols = [c for c in ["1D", "1M", "YTD", "1Y"] if c in trail.columns]
    st.dataframe(info.join(trail[latest_cols]).style.format(
        {"Last price": "{:,.2f}", **{c: "{:.2%}" for c in latest_cols}}, na_rep="-"))

# -----------------------------------------------------------------------------
# TAB 2 - RISK & RETURN
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Performance and risk statistics")
    st.caption(f"Window: {p_daily.index[0].date()} to {p_daily.index[-1].date()} | {freq.lower()} returns | "
               f"annualisation factor {ann} | risk-free {rf_pct:.2f}% | benchmark: {bench if bench_ok else 'none'}")
    st.dataframe(style_stats(stats_df), height=min(600, 60 + 36 * len(stats_df)))

    c1, c2 = st.columns(2)
    with c1:
        sc = stats_df.reset_index()
        sc["Asset class"] = sc["Asset"].map(lambda a: CLASS_OF.get(a, "Other"))
        fig = px.scatter(sc, x="Volatility", y="CAGR", color="Asset class", text="Asset",
                         hover_data={"Sharpe": ":.2f", "Max drawdown": ":.1%"},
                         title="Risk vs return (annualised)", color_discrete_sequence=PALETTE)
        fig.update_traces(textposition="top center", marker=dict(size=13))
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=50, b=10))
        show(fig, key="rr_scatter")
    with c2:
        sh = stats_df["Sharpe"].sort_values()
        fig = go.Figure(go.Bar(x=sh.values, y=sh.index, orientation="h",
                               marker_color=[COLOR.get(i, "#888") for i in sh.index],
                               text=[f"{v:.2f}" for v in sh.values], textposition="outside"))
        fig.update_layout(title="Sharpe ratio", height=450, margin=dict(l=10, r=40, t=50, b=10))
        show(fig, key="rr_sharpe")

    st.subheader("Drawdowns")
    dd = drawdown_series(rets_all[names_all])
    show(line_fig(dd, "Drawdown from previous peak", yfmt=".0%", colors=COLOR, bench=bench if bench_ok else None),
         key="rr_dd")
    dd_asset = st.selectbox("Worst drawdown episodes for", names_all, key="dd_asset")
    eps = drawdown_episodes(rets_all[dd_asset], top=5)
    if eps.empty:
        st.info("No drawdowns in this window.")
    else:
        st.dataframe(eps.style.format({"Depth": "{:.1%}", "Periods to trough": "{:.0f}",
                                       "Periods to recover": "{:.0f}"}, na_rep="-"), hide_index=True)

    st.subheader(f"Rolling statistics ({win}-period window)")
    roll_assets = st.multiselect("Assets shown", names_all, default=names_all[: min(5, len(names_all))], key="roll_sel")
    if roll_assets:
        r1, r2 = st.columns(2)
        with r1:
            rv = rets_all[roll_assets].rolling(win).std() * np.sqrt(ann)
            show(line_fig(rv.dropna(how="all"), "Rolling annualised volatility", yfmt=".0%", colors=COLOR,
                          height=380), key="rr_rvol")
        with r2:
            ex = rets_all[roll_assets] - rf / ann
            rs = ex.rolling(win).mean() * ann / (rets_all[roll_assets].rolling(win).std() * np.sqrt(ann))
            show(line_fig(rs.dropna(how="all"), "Rolling Sharpe ratio", colors=COLOR, height=380), key="rr_rsharpe")

# -----------------------------------------------------------------------------
# TAB 3 - CORRELATION
# -----------------------------------------------------------------------------
with tabs[2]:
    if len(sel) < 2:
        st.info("Select at least two assets to see correlations.")
    else:
        method = st.radio("Correlation measure", ["pearson", "spearman"], horizontal=True, key="corr_method")
        cm = rets_sel.corr(method=method)
        c1, c2 = st.columns([3, 2])
        with c1:
            show(heatmap_fig(cm, f"{method.capitalize()} correlation of {freq.lower()} returns", zmin=-1, zmax=1,
                             height=max(420, 50 * len(sel) + 100)), key="corr_heat")
        with c2:
            pairs = cm.where(np.triu(np.ones(cm.shape), 1).astype(bool)).stack().sort_values()
            st.markdown("**Least correlated pairs (best diversifiers)**")
            st.dataframe(pairs.head(5).rename("Correlation").reset_index().rename(
                columns={"level_0": "Asset A", "level_1": "Asset B"}).style.format({"Correlation": "{:.2f}"}),
                hide_index=True)
            st.markdown("**Most correlated pairs**")
            st.dataframe(pairs.tail(5).iloc[::-1].rename("Correlation").reset_index().rename(
                columns={"level_0": "Asset A", "level_1": "Asset B"}).style.format({"Correlation": "{:.2f}"}),
                hide_index=True)
            st.metric("Average pairwise correlation", f"{pairs.mean():.2f}")

        st.subheader("Rolling correlation")
        pc1, pc2 = st.columns(2)
        a = pc1.selectbox("Asset A", sel, index=0, key="rc_a")
        b = pc2.selectbox("Asset B", sel, index=min(1, len(sel) - 1), key="rc_b")
        if a == b:
            st.info("Choose two different assets.")
        else:
            rc = rets_sel[a].rolling(win).corr(rets_sel[b]).dropna()
            fig = line_fig(rc.to_frame(f"{a} vs {b}"), f"Rolling {win}-period correlation", height=380)
            fig.add_hline(y=0, line_dash="dot", line_color="#888")
            fig.update_yaxes(range=[-1, 1])
            show(fig, key="rc_line")

        st.subheader("Correlation in stress vs calm markets")
        ref = bench_ret if bench_ok else rets_sel.mean(axis=1)
        ref_name = bench if bench_ok else "equal-weight average"
        q_lo, q_hi = ref.quantile(0.2), ref.quantile(0.8)
        down = rets_sel[ref <= q_lo].corr()
        up = rets_sel[ref >= q_hi].corr()
        s1, s2 = st.columns(2)
        s1.caption(f"Worst 20% of periods for {ref_name} ({int((ref <= q_lo).sum())} obs.)")
        s2.caption(f"Best 20% of periods for {ref_name} ({int((ref >= q_hi).sum())} obs.)")
        with s1:
            show(heatmap_fig(down, zmin=-1, zmax=1, height=max(380, 45 * len(sel) + 80)), key="corr_down")
        with s2:
            show(heatmap_fig(up, zmin=-1, zmax=1, height=max(380, 45 * len(sel) + 80)), key="corr_up")
        st.caption("Correlations that rise in the downside sample indicate diversification that fails when it is needed most.")

# -----------------------------------------------------------------------------
# TAB 4 - DISTRIBUTION
# -----------------------------------------------------------------------------
with tabs[3]:
    d_asset = st.selectbox("Asset", names_all, key="dist_asset")
    r = rets_all[d_asset].dropna()
    mu, sd = r.mean(), r.std()
    jb_stat, jb_p = stats.jarque_bera(r)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean (per period)", f"{mu:.3%}")
    m2.metric("Std. dev. (per period)", f"{sd:.3%}")
    m3.metric("Skewness", f"{r.skew():.2f}")
    m4.metric("Excess kurtosis", f"{r.kurt():.2f}")
    m5.metric("Jarque-Bera p-value", f"{jb_p:.4f}", "Rejects normality" if jb_p < 0.05 else "Cannot reject normality",
              delta_color="off")

    c1, c2 = st.columns(2)
    with c1:
        xs = np.linspace(r.min(), r.max(), 300)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=r, histnorm="probability density", nbinsx=60, name="Empirical",
                                   marker_color=PALETTE[0], opacity=0.75))
        fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, mu, sd), name="Normal fit",
                                 line=dict(color=PALETTE[1], width=2.5)))
        fig.update_layout(title=f"Return distribution - {d_asset}", height=420, bargap=0.02,
                          margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15))
        fig.update_xaxes(tickformat=".1%")
        show(fig, key="dist_hist")
    with c2:
        osm, osr = stats.probplot(r, dist="norm", fit=False)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Observed",
                                 marker=dict(color=PALETTE[0], size=5)))
        fig.add_trace(go.Scatter(x=[osm.min(), osm.max()], y=[mu + sd * osm.min(), mu + sd * osm.max()],
                                 mode="lines", name="Normal", line=dict(color=PALETTE[1], dash="dash")))
        fig.update_layout(title="Q-Q plot vs normal distribution", height=420,
                          xaxis_title="Theoretical quantiles", yaxis_title="Observed returns",
                          margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15))
        fig.update_yaxes(tickformat=".1%")
        show(fig, key="dist_qq")

    st.subheader("Value-at-Risk and Expected Shortfall (one period)")
    z = stats.norm.ppf(1 - conf)
    S, K = r.skew(), r.kurt()
    z_cf = z + (z ** 2 - 1) * S / 6 + (z ** 3 - 3 * z) * K / 24 - (2 * z ** 3 - 5 * z) * S ** 2 / 36
    q_hist = r.quantile(1 - conf)
    var_tbl = pd.DataFrame({
        "VaR": [-q_hist, -(mu + sd * z), -(mu + sd * z_cf)],
        "Expected shortfall": [-r[r <= q_hist].mean(), -(mu - sd * stats.norm.pdf(z) / (1 - conf)), np.nan],
    }, index=["Historical", "Parametric (normal)", "Cornish-Fisher (skew/kurtosis adjusted)"])
    st.dataframe(var_tbl.style.format("{:.2%}", na_rep="-"))
    st.caption(f"{conf:.1%} confidence. A normal VaR below the historical VaR signals fat tails that the normal model understates.")

    st.subheader("Return dispersion across assets")
    fig = go.Figure()
    for c in names_all:
        fig.add_trace(go.Box(y=rets_all[c].dropna(), name=c, marker_color=COLOR[c], boxmean=True))
    fig.update_layout(height=430, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
    fig.update_yaxes(tickformat=".1%")
    show(fig, key="dist_box")

# -----------------------------------------------------------------------------
# TAB 5 - REGRESSION & FACTORS
# -----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Single-factor model (CAPM)")
    if not bench_ok:
        st.info("Select a benchmark with available data to run the CAPM regression.")
    else:
        y_asset = st.selectbox("Asset (dependent variable)", [s for s in names_all if s != bench] or names_all,
                               key="capm_asset")
        hac = st.checkbox("Use Newey-West (HAC) standard errors", value=True, key="capm_hac")
        lags = st.slider("Newey-West lags", 1, 20, 5, key="capm_lags") if hac else 0
        dfm = pd.concat([rets_all[y_asset].rename("y"), rets_all[bench].rename("m")], axis=1).dropna()
        if len(dfm) < 30:
            st.warning("Fewer than 30 overlapping observations.")
        else:
            rf_p = rf / ann
            y = (dfm["y"] - rf_p).values
            X = pd.DataFrame({f"Market excess return ({bench})": (dfm["m"] - rf_p).values})
            res = ols(y, X, hac_lags=lags)
            tbl = res["table"].copy()
            c1, c2 = st.columns([2, 3])
            with c1:
                alpha_ann = tbl.loc["Alpha (per period)", "Coefficient"] * ann
                beta = tbl.iloc[1, 0]
                st.metric("Beta", f"{beta:.3f}")
                st.metric("Annualised alpha", f"{alpha_ann:.2%}")
                st.metric("R-squared", f"{res['r2']:.3f}")
                st.dataframe(tbl.style.format({"Coefficient": "{:.5f}", "Std. error": "{:.5f}",
                                               "t-stat": "{:.2f}", "p-value": "{:.4f}"}))
                st.caption(f"n = {res['n']} observations | "
                           f"{'Newey-West' if hac else 'classical OLS'} standard errors | "
                           "p-values below 0.05 indicate significance at the 5% level.")
            with c2:
                xv = X.iloc[:, 0].values
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xv, y=y, mode="markers", name="Observations",
                                         marker=dict(color=PALETTE[0], size=5, opacity=0.5)))
                xl = np.array([xv.min(), xv.max()])
                fig.add_trace(go.Scatter(x=xl, y=tbl.iloc[0, 0] + beta * xl, mode="lines", name="Fitted line",
                                         line=dict(color=PALETTE[1], width=3)))
                fig.update_layout(title=f"{y_asset} vs {bench} (excess returns)", height=430,
                                  margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15))
                fig.update_xaxes(tickformat=".1%")
                fig.update_yaxes(tickformat=".1%")
                show(fig, key="capm_scatter")

            rb = dfm["y"].rolling(win).cov(dfm["m"]) / dfm["m"].rolling(win).var()
            fig = line_fig(rb.dropna().to_frame(f"Rolling beta ({win} periods)"), "Rolling beta vs benchmark", height=360)
            fig.add_hline(y=1, line_dash="dot", line_color="#888")
            show(fig, key="capm_rbeta")

    st.subheader("Fama-French three-factor model")
    st.caption("Uses the daily US research factors from Kenneth French's data library (Mkt-RF, SMB, HML, RF). "
               "Best suited to US equity assets and funds.")
    use_ff = st.checkbox("Load Fama-French factors and run the regression", value=False, key="ff_on")
    if use_ff:
        try:
            ff_daily = load_ff3_daily()
        except Exception as exc:
            ff_daily = None
            st.error(f"Could not download the Fama-French factors ({exc}). "
                     "The library may be temporarily unavailable; try again later.")
        if ff_daily is not None:
            ff_idx = (1 + ff_daily).cumprod()
            ff = to_returns(resample_prices(ff_idx, freq)) if freq != "Daily" else ff_daily.copy()
            ff_asset = st.selectbox("Asset (dependent variable)", names_all, key="ff_asset")
            ff_hac = st.checkbox("Use Newey-West (HAC) standard errors", value=True, key="ff_hac")
            ff_lags = st.slider("Newey-West lags", 1, 20, 5, key="ff_lags") if ff_hac else 0
            dff = pd.concat([rets_all[ff_asset].rename("y"), ff], axis=1, join="inner").dropna()
            if len(dff) < 30:
                st.warning("Fewer than 30 overlapping observations between the asset and the factor data "
                           f"(factors end {ff_daily.index[-1].date()}). Widen the period.")
            else:
                yy = (dff["y"] - dff["RF"]).values
                res = ols(yy, dff[["Mkt-RF", "SMB", "HML"]], hac_lags=ff_lags)
                t = res["table"]
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("Annualised alpha", f"{t.loc['Alpha (per period)', 'Coefficient'] * ann:.2%}")
                f2.metric("Market beta", f"{t.loc['Mkt-RF', 'Coefficient']:.2f}")
                f3.metric("R-squared", f"{res['r2']:.3f}")
                f4.metric("Adj. R-squared", f"{res['adj_r2']:.3f}")
                st.dataframe(t.style.format({"Coefficient": "{:.5f}", "Std. error": "{:.5f}",
                                             "t-stat": "{:.2f}", "p-value": "{:.4f}"}))
                st.caption(f"n = {res['n']} | factor data available until {ff_daily.index[-1].date()} | "
                           "SMB = small minus big, HML = high minus low book-to-market.")
                st.markdown("**Interpretation guide:** a positive and significant SMB loading indicates a small-cap tilt, "
                            "a positive HML loading a value tilt, and a significant alpha indicates return not explained "
                            "by the three factors.")

# -----------------------------------------------------------------------------
# TAB 6 - PORTFOLIO
# -----------------------------------------------------------------------------
with tabs[5]:
    if not multi:
        st.info("Select at least two assets with overlapping history (30+ periods) to use the portfolio tools.")
    else:
        st.subheader("Portfolio construction")
        n_a = len(sel)
        wkey = "w_" + hashlib.md5("|".join(sel).encode()).hexdigest()[:8]
        left, right = st.columns([2, 3])
        with left:
            st.markdown("**Weights** (edit the table; they are normalised to 100%)")
            w_default = pd.DataFrame({"Asset": sel, "Weight (%)": [round(100.0 / n_a, 2)] * n_a})
            w_edit = st.data_editor(
                w_default, key=wkey, hide_index=True, disabled=["Asset"],
                column_config={"Weight (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0,
                                                                          format="%.1f")},
            )
            w_raw = w_edit["Weight (%)"].fillna(0).values.astype(float)
            total_w = w_raw.sum()
            if total_w <= 0:
                st.error("Weights must sum to more than zero.")
                st.stop()
            w_user = w_raw / total_w
            st.caption(f"Entered total: {total_w:.1f}%" + ("" if abs(total_w - 100) < 0.5 else " (normalised to 100%)"))
            rebal = st.selectbox("Rebalancing", REBALANCE_OPTIONS, index=1, key="rebal")
            cap = st.slider("Maximum weight per asset (optimiser)", 0.10, 1.0, 0.50, 0.05, key="cap",
                            help="Constraint used by the optimised portfolios (long-only, fully invested).")
            profile = st.selectbox("Risk profile (optimiser constraints)", list(RISK_PROFILES), index=2,
                                   key="profile")
            eq_cap, vol_cap = RISK_PROFILES[profile]
            eq_mask = np.array([1.0 if CLASS_OF.get(a) == EQUITY_CLASS else 0.0 for a in sel])
            if eq_cap < 1 and not eq_mask.any():
                st.caption("No asset in your selection is classed as equity, so the equity cap has no effect.")
            eq_share = float(eq_mask @ w_user)
            vol_user = float(np.sqrt(w_user @ (common.cov().values * ann) @ w_user))
            chk1, chk2 = st.columns(2)
            chk1.metric("Equity share", f"{eq_share:.1%}",
                        ("within limit" if eq_share <= eq_cap + 1e-9 else "above limit") if eq_cap < 1 else None,
                        delta_color="off")
            chk2.metric("Ex-ante volatility", f"{vol_user:.1%}",
                        ("within limit" if vol_user <= vol_cap else "above limit") if vol_cap else None,
                        delta_color="off")
        with right:
            fig = go.Figure(go.Pie(labels=sel, values=w_user, hole=0.55, sort=False,
                                   marker=dict(colors=[COLOR[s] for s in sel]), textinfo="label+percent"))
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
            show(fig, key="pf_pie")

        mu_a = (common.mean() * ann).values
        cov_a = (common.cov() * ann).values
        opt = optimise_portfolios(mu_a, cov_a, rf, cap, eq_mask, eq_cap, vol_cap)
        if len(opt) < 3:
            st.warning("The optimiser found no feasible portfolio for at least one objective under this risk profile "
                       "(equity cap, volatility limit and weight cap together). Loosen a constraint or add "
                       "lower-risk assets such as government bonds.")

        portfolios = {"Your portfolio": w_user, "Equal weight": np.full(n_a, 1.0 / n_a)}
        portfolios.update(opt)
        pr = pd.DataFrame({name: portfolio_returns(common, w, rebal if name == "Your portfolio" else "Every period")
                           for name, w in portfolios.items()})
        if bench_ok:
            pr[f"Benchmark: {bench}"] = rets_all[bench].reindex(pr.index)
        pr = pr.dropna()

        st.subheader("Portfolio comparison")
        st.caption("Optimised portfolios (min variance, max Sharpe, risk parity) are estimated in-sample on the same "
                   "history they are evaluated on, so they flatter reality. Equal weight and optimised portfolios are "
                   "rebalanced every period.")
        pcolors = {"Your portfolio": PALETTE[0], "Equal weight": PALETTE[8], "Min variance": PALETTE[4],
                   "Max Sharpe": PALETTE[1], "Risk parity": PALETTE[2], f"Benchmark: {bench}": BENCH_COLOR}
        wealth = (1 + pr).cumprod() * 100
        fig = line_fig(wealth, "Growth of 100", colors=pcolors, bench=f"Benchmark: {bench}" if bench_ok else None,
                       log=log_scale)
        show(fig, key="pf_wealth")

        pstats = build_stats(pr, f"Benchmark: {bench}" if bench_ok else None, rf, ann, conf)
        st.dataframe(style_stats(pstats), height=60 + 36 * len(pstats))

        d1, d2 = st.columns(2)
        with d1:
            show(line_fig(drawdown_series(pr), "Portfolio drawdowns", yfmt=".0%", colors=pcolors, height=380,
                          bench=f"Benchmark: {bench}" if bench_ok else None), key="pf_dd")
        with d2:
            rv = pr.rolling(win).std() * np.sqrt(ann)
            show(line_fig(rv.dropna(how="all"), "Rolling volatility", yfmt=".0%", colors=pcolors, height=380,
                          bench=f"Benchmark: {bench}" if bench_ok else None), key="pf_rvol")

        st.subheader("Weights and risk contributions")
        wt = pd.DataFrame({k: v for k, v in portfolios.items()}, index=sel)
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Weights by portfolio**")
            st.dataframe(wt.style.format("{:.1%}"))
        with e2:
            sel_pf = st.selectbox("Risk contribution for", list(portfolios), key="rc_pf")
            rc = pd.Series(risk_contributions(portfolios[sel_pf], cov_a), index=sel).sort_values()
            fig = go.Figure(go.Bar(x=rc.values, y=rc.index, orientation="h",
                                   marker_color=[COLOR[i] for i in rc.index],
                                   text=[f"{v:.1%}" for v in rc.values], textposition="outside"))
            fig.update_layout(title="Share of portfolio variance", height=360, margin=dict(l=10, r=40, t=50, b=10))
            fig.update_xaxes(tickformat=".0%")
            show(fig, key="pf_rc")

        st.subheader("Efficient frontier")
        rng = np.random.default_rng(42)
        sims = rng.dirichlet(np.ones(n_a), size=6000)
        sims = sims[sims.max(axis=1) <= max(cap, 1.0 / n_a + 1e-6)]
        if eq_cap < 1:
            sims = sims[sims @ eq_mask <= eq_cap]
        sim_ret = sims @ mu_a
        sim_vol = np.sqrt(np.einsum("ij,jk,ik->i", sims, cov_a, sims))
        sim_sharpe = (sim_ret - rf) / sim_vol
        f_vol, f_ret = efficient_frontier(mu_a, cov_a, cap, eq_mask, eq_cap)
        fig = go.Figure()
        if len(sims):
            fig.add_trace(go.Scatter(x=sim_vol, y=sim_ret, mode="markers", name="Random portfolios",
                                     marker=dict(size=4, color=sim_sharpe, colorscale="Viridis", opacity=0.55,
                                                 colorbar=dict(title="Sharpe"))))
        if len(f_vol):
            fig.add_trace(go.Scatter(x=f_vol, y=f_ret, mode="lines", name="Efficient frontier",
                                     line=dict(color=BENCH_COLOR, width=3)))
        for i, a_name in enumerate(sel):
            fig.add_trace(go.Scatter(x=[np.sqrt(cov_a[i, i])], y=[mu_a[i]], mode="markers+text", text=[a_name],
                                     textposition="top center", name=a_name, showlegend=False,
                                     marker=dict(size=10, color=COLOR[a_name], symbol="diamond")))
        for name, w in portfolios.items():
            fig.add_trace(go.Scatter(x=[np.sqrt(w @ cov_a @ w)], y=[w @ mu_a], mode="markers", name=name,
                                     marker=dict(size=15, color=pcolors.get(name, "#000"), symbol="star",
                                                 line=dict(width=1, color="white"))))
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title="Annualised volatility", yaxis_title="Annualised expected return (historical mean)",
                          legend=dict(orientation="h", y=-0.15))
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        show(fig, key="pf_frontier")
        st.caption("Expected returns are historical sample means and are very noisy; treat the frontier as an "
                   "illustration of diversification, not a forecast.")

# -----------------------------------------------------------------------------
# TAB 7 - SEASONALITY & ROLLING
# -----------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Calendar returns")
    s_asset = st.selectbox("Asset", names_all, key="seas_asset")
    sd_ = rets_daily[s_asset].dropna()
    monthly = (1 + sd_).groupby([sd_.index.year, sd_.index.month]).prod() - 1
    mt = monthly.unstack()
    mt.columns = [MONTHS[m - 1] for m in mt.columns]
    mt = mt.reindex(columns=MONTHS)
    annual = (1 + sd_).groupby(sd_.index.year).prod() - 1
    mt["Year"] = annual
    show(heatmap_fig(mt * 100, f"Monthly returns (%) - {s_asset}", fmt=".1f", scale="RdYlGn", mid=0,
                     height=max(320, 34 * len(mt) + 100)), key="seas_heat")
    st.caption("The first and last calendar years may be partial; 'Year' compounds the months available.")

    c1, c2 = st.columns(2)
    with c1:
        avg_m = mt[MONTHS].mean()
        fig = go.Figure(go.Bar(x=MONTHS, y=avg_m.values,
                               marker_color=[PALETTE[4] if v >= 0 else PALETTE[9] for v in avg_m.fillna(0).values]))
        fig.update_layout(title="Average return by calendar month", height=360, margin=dict(l=10, r=10, t=50, b=10))
        fig.update_yaxes(ticksuffix="%")
        show(fig, key="seas_avg")
    with c2:
        pos = (mt[MONTHS] > 0).sum() / mt[MONTHS].notna().sum()
        fig = go.Figure(go.Bar(x=MONTHS, y=pos.values * 100, marker_color=PALETTE[0]))
        fig.add_hline(y=50, line_dash="dot", line_color="#888")
        fig.update_layout(title="Share of years with a positive month", height=360, margin=dict(l=10, r=10, t=50, b=10))
        fig.update_yaxes(ticksuffix="%", range=[0, 100])
        show(fig, key="seas_pos")

    st.subheader("Rolling multi-year returns")
    ry = st.radio("Horizon", [1, 3, 5], horizontal=True, format_func=lambda y: f"{y}Y", key="roll_years")
    rr_assets = st.multiselect("Assets shown", names_all, default=names_all[: min(5, len(names_all))], key="roll_ret_sel")
    if rr_assets:
        n_per = PPY * ry
        out = {}
        for c in rr_assets:
            s = prices_full[c].dropna()
            out[c] = (s / s.shift(n_per) - 1) if ry == 1 else ((s / s.shift(n_per)) ** (1 / ry) - 1)
        roll_df = pd.DataFrame(out).dropna(how="all").loc[start_d:end_d]
        if roll_df.dropna(how="all").empty:
            st.info("Not enough history for this horizon in the selected window.")
        else:
            show(line_fig(roll_df, f"Rolling {ry}-year return" + (" (annualised)" if ry > 1 else ""), yfmt=".0%",
                          colors=COLOR, bench=bench if bench_ok else None), key="roll_ret")

# -----------------------------------------------------------------------------
# TAB 8 - STRESS TESTS
# -----------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Historical crisis episodes")
    st.caption("Total return over approximate S&P 500 peak-to-trough windows. Blank = asset has no history for the "
               "whole episode. Independent of the sidebar period (use a ticker with long history, e.g. indices).")
    ret_tbl, dd_tbl = {}, {}
    for crisis, (a, b_) in CRISES.items():
        ret_tbl[crisis], dd_tbl[crisis] = {}, {}
        for c in names_all:
            tr_, md_ = window_perf(prices_full[c], a, b_, TOL)
            ret_tbl[crisis][c], dd_tbl[crisis][c] = tr_, md_
    ret_df = pd.DataFrame(ret_tbl).T
    dd_df = pd.DataFrame(dd_tbl).T
    if ret_df.dropna(how="all").empty:
        st.info("No crisis window is covered by the downloaded history for these assets.")
    else:
        show(heatmap_fig(ret_df * 100, "Total return during each episode (%)", fmt=".1f", scale="RdYlGn", mid=0,
                         height=max(340, 52 * len(ret_df) + 120)), key="st_ret")
        with st.expander("Maximum drawdown inside each episode"):
            st.dataframe(dd_df.style.format("{:.1%}", na_rep="-"))

    st.subheader("Custom window")
    cw1, cw2 = st.columns(2)
    cw_a = cw1.date_input("From", value=date(2020, 2, 19), key="cw_a")
    cw_b = cw2.date_input("To", value=date(2020, 3, 23), key="cw_b")
    if cw_a < cw_b:
        rows = {c: dict(zip(["Total return", "Max drawdown"], window_perf(prices_full[c], cw_a, cw_b, TOL)))
                for c in names_all}
        st.dataframe(pd.DataFrame(rows).T.style.format("{:.1%}", na_rep="-"))
    else:
        st.info("'From' must be before 'To'.")

    st.subheader("Hypothetical scenario on your portfolio")
    if not multi:
        st.info("Select at least two assets with overlapping history to run a scenario.")
    else:
        st.caption("Edit the shock per asset (in %). Weights come from the Portfolio tab. "
                   "Portfolio impact = sum of weight x shock.")
        skey = "s_" + hashlib.md5("|".join(sel).encode()).hexdigest()[:8]
        sc_default = pd.DataFrame({
            "Asset": sel,
            "Asset class": [CLASS_OF.get(a, "Other") for a in sel],
            "Weight (%)": np.round(w_user * 100, 2),
            "Shock (%)": [DEFAULT_SHOCKS.get(CLASS_OF.get(a, "Other"), -10.0) for a in sel],
        })
        sc_edit = st.data_editor(sc_default, key=skey, hide_index=True,
                                 disabled=["Asset", "Asset class", "Weight (%)"],
                                 column_config={"Shock (%)": st.column_config.NumberColumn(min_value=-100.0,
                                                                                         max_value=200.0, step=1.0,
                                                                                         format="%.1f")})
        contrib = sc_edit["Weight (%)"].values / 100 * sc_edit["Shock (%)"].values
        g1, g2 = st.columns([1, 2])
        g1.metric("Portfolio impact", f"{contrib.sum():.2f}%")
        fig = go.Figure(go.Bar(x=sc_edit["Asset"], y=contrib,
                               marker_color=[PALETTE[4] if v >= 0 else PALETTE[9] for v in contrib]))
        fig.update_layout(title="Contribution to portfolio return (percentage points)", height=360,
                          margin=dict(l=10, r=10, t=50, b=10))
        with g2:
            show(fig, key="st_contrib")

# -----------------------------------------------------------------------------
# TAB 9 - HEDGING OVERLAYS (protective put + Bund duration overlay)
# -----------------------------------------------------------------------------
with tabs[8]:
    st.subheader("Derivative overlays")
    st.caption("Simplified, input-driven models: Black-Scholes for the put, a duration approximation for the "
               "futures overlay. Confirm every market input on the Bloomberg terminal (option data: OMON; futures: "
               "price and duration of the Euro-Bund contract) before presenting.")

    if multi:
        eq_default = float(np.clip(round(float(eq_share) * 100, 1), 0.0, 100.0))
        bond_default = float(np.clip(round(float(np.array(
            [1.0 if CLASS_OF.get(a) == "Fixed income" else 0.0 for a in sel]) @ w_user) * 100, 1), 0.0, 100.0))
    else:
        eq_default, bond_default = 50.0, 50.0

    st.markdown("#### 1. Protective put on an equity index")
    hc1, hc2, hc3 = st.columns(3)
    und_default = names_all.index("Euro Stoxx 50") if "Euro Stoxx 50" in names_all else 0
    und = hc1.selectbox("Underlying index", names_all, index=und_default, key="h_und")
    und_px = prices_full[und].dropna()
    spot0 = float(und_px.iloc[-1])
    hist_vol = float(rets_all[und].dropna().std() * np.sqrt(ann))
    if not np.isfinite(hist_vol) or hist_vol <= 0:
        hist_vol = 0.20
    spot = hc1.number_input("Spot level", min_value=0.01, value=spot0, key=f"h_spot_{und}")
    sigma = hc2.number_input("Implied volatility (%)", 1.0, 150.0, float(np.clip(round(hist_vol * 100, 1), 1.0, 150.0)), 0.5,
                             key=f"h_vol_{und}", help="Default = historical volatility in the window. "
                                                      "Replace with the implied vol from OMON.") / 100
    q = hc2.number_input("Dividend yield (%)", 0.0, 15.0, 3.0, 0.1, key="h_q") / 100
    strike_pct = hc3.slider("Strike (% of spot)", 70, 110, 90, 1, key="h_k")
    T = hc3.slider("Maturity (years)", 0.25, 3.0, 1.0, 0.25, key="h_T")
    hc4, hc5, _ = st.columns(3)
    hedge_ratio = hc4.slider("Hedge ratio (% of equity sleeve)", 0, 100, 50, 5, key="h_hr") / 100
    eq_sleeve = hc5.number_input("Equity share of portfolio (%)", 0.0, 100.0, eq_default, 1.0,
                                 key=f"h_eq_{eq_default}") / 100

    K = spot * strike_pct / 100
    put0 = float(bs_put(spot, K, T, rf, q, sigma))
    cost_pct = put0 / spot
    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("Put premium (index points)", f"{put0:,.2f}")
    pm2.metric("Premium (% of spot)", f"{cost_pct:.2%}")
    pm3.metric("Annualised cost on whole portfolio", f"{hedge_ratio * eq_sleeve * cost_pct / T:.2%}")
    pm4.metric("Put delta", f"{bs_put_delta(spot, K, T, rf, q, sigma):.2f}")

    pc1, pc2 = st.columns(2)
    with pc1:
        mv = np.linspace(-0.5, 0.5, 101)
        unh = mv
        hed = mv + hedge_ratio * (np.maximum(K / spot - (1 + mv), 0) - cost_pct)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mv, y=unh, name="Unhedged", line=dict(color=PALETTE[8], width=2.5)))
        fig.add_trace(go.Scatter(x=mv, y=hed, name="With protective put", line=dict(color=PALETTE[0], width=3)))
        fig.add_hline(y=0, line_dash="dot", line_color="#888")
        fig.update_layout(title="Equity sleeve return at expiry vs index move (premium included, not financed)",
                          height=400, margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h", y=-0.15))
        fig.update_xaxes(tickformat=".0%", title="Index move")
        fig.update_yaxes(tickformat=".0%", title="Sleeve return")
        show(fig, key="h_payoff")
    with pc2:
        strikes = np.arange(80, 100.1, 2.5)
        vols = np.arange(10, 40.1, 5.0)
        grid = np.array([[float(bs_put(100, k, T, rf, q, v / 100)) / T for v in vols] for k in strikes])
        gdf = pd.DataFrame(grid, index=[f"{k:.1f}%" for k in strikes], columns=[f"{v:.0f}%" for v in vols])
        fig = px.imshow(gdf, text_auto=".1%", color_continuous_scale="Blues", aspect="auto",
                        labels=dict(x="Implied vol", y="Strike (% of spot)", color="Cost p.a."))
        fig.update_layout(title="Sensitivity: put cost per year (% of notional)", height=400,
                          margin=dict(l=10, r=10, t=60, b=10))
        show(fig, key="h_sens")

    st.markdown("**Scenario repricing** (put revalued after the horizon with the shocked spot and volatility)")
    scen_default = pd.DataFrame({
        "Scenario": ["Mild correction", "Bear market", "Crash", "Rally"],
        "Index move (%)": [-10.0, -20.0, -35.0, 15.0],
        "Vol shock (pts)": [5.0, 10.0, 20.0, -5.0],
        "Horizon (months)": [1.0, 3.0, 3.0, 3.0],
    })
    scen = st.data_editor(scen_default, key="h_scen", hide_index=True, disabled=["Scenario"])
    rows = []
    for _, rw in scen.iterrows():
        S1 = spot * (1 + rw["Index move (%)"] / 100)
        T1 = max(T - rw["Horizon (months)"] / 12, 0.0)
        v1 = max(sigma + rw["Vol shock (pts)"] / 100, 0.01)
        p1 = float(bs_put(S1, K, T1, rf, q, v1))
        move = rw["Index move (%)"] / 100
        put_pnl = hedge_ratio * (p1 - put0) / spot
        rows.append({
            "Scenario": rw["Scenario"],
            "Put value after": p1,
            "Equity sleeve, unhedged": move,
            "Put P&L (on sleeve)": put_pnl,
            "Equity sleeve, hedged": move + put_pnl,
            "Portfolio contribution, unhedged": eq_sleeve * move,
            "Portfolio contribution, hedged": eq_sleeve * (move + put_pnl),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Scenario").style.format(
        {"Put value after": "{:,.2f}", **{c: "{:.1%}" for c in list(rows[0])[2:]}}), height=215)
    st.caption("The put is sized on the equity sleeve; premium is paid up-front and its financing cost is ignored. "
               "A real overlay would use listed contracts (multiplier, expiry grid) and would carry basis risk.")

    st.markdown("#### 2. Bund futures duration overlay")
    bc1, bc2, bc3, bc4 = st.columns(4)
    bond_sh = bc1.number_input("Bond share of portfolio (%)", 0.0, 100.0, bond_default, 1.0,
                               key=f"b_sh_{bond_default}") / 100
    dur_bonds = bc2.number_input("Average duration of bond holdings (yrs)", 0.0, 30.0, 6.0, 0.5, key="b_dur")
    dur_fut = bc3.number_input("Futures modified duration (yrs)", 1.0, 15.0, 8.0, 0.5, key="b_dfut",
                               help="Take from the Bund futures contract on Bloomberg (CTD duration).")
    dur_tgt = bc4.number_input("Target portfolio duration (yrs)", 0.0, 15.0, 1.5, 0.25, key="b_tgt")
    d_port = bond_sh * dur_bonds
    hedge_notional = (d_port - dur_tgt) / dur_fut        # as a fraction of NAV; positive = short futures
    bm1, bm2, bm3 = st.columns(3)
    bm1.metric("Current portfolio duration", f"{d_port:.2f} yrs")
    bm2.metric("Futures position (% of NAV)", f"{abs(hedge_notional):.1%}",
               "short" if hedge_notional > 0 else ("long" if hedge_notional < 0 else "flat"), delta_color="off")
    bm3.metric("Duration after overlay", f"{dur_tgt:.2f} yrs")

    shifts = np.array([-200, -100, -50, 0, 50, 100, 200, 300])
    dy = shifts / 10000.0
    unh_r = -d_port * dy
    fut_pnl = dur_fut * hedge_notional * dy
    net = unh_r + fut_pnl
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=shifts, y=unh_r, name="Without overlay", line=dict(color=PALETTE[8], width=2.5)))
    fig.add_trace(go.Scatter(x=shifts, y=net, name="With futures overlay", line=dict(color=PALETTE[0], width=3)))
    fig.add_hline(y=0, line_dash="dot", line_color="#888")
    fig.update_layout(title="Portfolio P&L from a parallel yield shift (first-order duration approximation)",
                      height=380, margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(title="Yield shift (bp)")
    fig.update_yaxes(tickformat=".1%", title="% of portfolio")
    show(fig, key="b_pnl")
    st.caption("Ignores convexity, the basis between the future and the bonds held, carry and margin financing.")

    st.markdown("#### 3. Combined stress: equity sell-off plus rate shock")
    cs1, cs2, cs3 = st.columns(3)
    c_eq = cs1.number_input("Index move (%)", -80.0, 80.0, -20.0, 1.0, key="c_eq") / 100
    c_vol = cs2.number_input("Vol shock (pts)", -20.0, 60.0, 10.0, 1.0, key="c_vol")
    c_dy = cs3.number_input("Yield shift (bp)", -300.0, 500.0, 100.0, 25.0, key="c_dy") / 10000
    c_h = st.slider("Horizon (months)", 0.0, 12.0, 3.0, 0.5, key="c_h")
    p_after = float(bs_put(spot * (1 + c_eq), K, max(T - c_h / 12, 0.0), rf, q, max(sigma + c_vol / 100, 0.01)))
    eq_un = eq_sleeve * c_eq
    bond_un = -d_port * c_dy
    put_h = eq_sleeve * hedge_ratio * (p_after - put0) / spot
    fut_h = dur_fut * hedge_notional * c_dy
    comp = pd.DataFrame({
        "Without hedges": [eq_un, bond_un, 0.0, 0.0, eq_un + bond_un],
        "With hedges": [eq_un, bond_un, put_h, fut_h, eq_un + bond_un + put_h + fut_h],
    }, index=["Equity sleeve", "Bond sleeve", "Put P&L", "Futures P&L", "Total (these sleeves)"])
    cc1, cc2 = st.columns([2, 3])
    with cc1:
        st.dataframe(comp.style.format("{:.2%}"))
        st.caption("Only the equity and bond sleeves are modelled here (cash, gold and other assets are ignored).")
    with cc2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Without hedges", "With hedges"],
                             y=[comp.loc["Total (these sleeves)", "Without hedges"],
                                comp.loc["Total (these sleeves)", "With hedges"]],
                             marker_color=[PALETTE[9], PALETTE[0]],
                             text=[f"{comp.iloc[-1, 0]:.1%}", f"{comp.iloc[-1, 1]:.1%}"], textposition="outside"))
        fig.update_layout(title="Combined stress: portfolio impact", height=320, margin=dict(l=10, r=10, t=50, b=10))
        fig.update_yaxes(tickformat=".0%")
        show(fig, key="c_bar")

# -----------------------------------------------------------------------------
# TAB 10 - DATA & METHODOLOGY
# -----------------------------------------------------------------------------
with tabs[9]:
    st.subheader("Data coverage")
    cov_tbl = pd.DataFrame({
        "Ticker / column": [name_to_ticker.get(c, c) if source.startswith("Yahoo") else c for c in names_all],
        "First date": [prices_full[c].first_valid_index().date() for c in names_all],
        "Last date": [prices_full[c].last_valid_index().date() for c in names_all],
        "Observations": [int(prices_full[c].notna().sum()) for c in names_all],
        "Missing in window (%)": [p_daily[c].isna().mean() * 100 for c in names_all],
    }, index=names_all)
    st.dataframe(cov_tbl.style.format({"Missing in window (%)": "{:.1f}"}))

    st.subheader("Download")
    sheets = {"Prices": p_daily, "Returns": rets_all, "Statistics": stats_df, "Correlation": corr_df}
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Prices (CSV)", p_daily.to_csv().encode(), "prices.csv", "text/csv", key="dl_p")
    d2.download_button("Returns (CSV)", rets_all.to_csv().encode(), "returns.csv", "text/csv", key="dl_r")
    d3.download_button("Statistics (CSV)", stats_df.to_csv().encode(), "statistics.csv", "text/csv", key="dl_s")
    d4.download_button(
        "Full workbook (Excel)", build_excel(sheets), "dashboard_export.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_x",
    )
    with st.expander("Preview price data"):
        st.dataframe(p_daily.tail(250).style.format("{:,.2f}", na_rep="-"))

    st.subheader("Methodology")
    st.markdown(
        """
**Prices and returns.** Yahoo Finance adjusted closes (dividends and splits included). Short gaps (up to 5 days) are
forward-filled, weekends removed. Returns are simple returns, `P_t / P_(t-1) - 1`, computed at the chosen frequency
(weekly = Friday close, monthly = month-end close). Assets quoted in different currencies are *not* converted; compare
like with like.

**Annualisation.** Factor 252 (daily), 52 (weekly), 12 (monthly). Volatility = std. dev. x sqrt(factor).
CAGR = (1 + total return)^(1/years) - 1.

**Sharpe** = mean excess return x factor / volatility. **Sortino** uses downside deviation of excess returns
(target = risk-free). **Calmar** = CAGR / |max drawdown|. **Information ratio** = mean active return x factor /
tracking error.

**VaR / CVaR.** Historical: empirical quantile and tail mean of period returns. Parametric: normal distribution.
Cornish-Fisher adjusts the normal quantile for sample skewness and kurtosis.

**Beta, alpha.** Beta = Cov(r, r_b) / Var(r_b). Alpha (Jensen) = (mean(r - rf) - beta x mean(r_b - rf)) x factor.

**Regressions.** OLS with an intercept; optional Newey-West heteroskedasticity-and-autocorrelation-consistent standard
errors (Bartlett kernel, normal-based p-values). Fama-French factors are US factors from Kenneth French's library.

**Portfolios.** Long-only, fully invested. Rebalancing rule applies to "Your portfolio". Optimisers (SLSQP): minimum
variance, maximum Sharpe, risk parity (equal risk contributions), each with a maximum weight cap and the chosen risk
profile (cap on the equity share and an ex-ante volatility limit). Inputs are the sample mean vector and covariance
matrix of the common history; results are in-sample.

**Protective put.** Black-Scholes price of a European put with continuous dividend yield:
`P = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)`. The scenario table revalues the put after the chosen horizon with the shocked
spot and volatility. The hedge is sized as a percentage of the equity sleeve; financing of the premium is ignored.

**Futures duration overlay.** Portfolio duration = bond share x average bond duration. Futures position (% of NAV) =
(current duration - target duration) / futures duration (positive = short). P&L uses the first-order approximation
`-D x change in yield`; convexity, basis and margin are ignored.

**Stress tests.** Total return and maximum drawdown between fixed dates; the scenario tool applies user-defined
instantaneous shocks to current weights.
        """
    )
    st.caption("For educational purposes only. Past performance is not indicative of future results. "
               "Not investment advice. Yahoo Finance data is unofficial and may contain errors.")
