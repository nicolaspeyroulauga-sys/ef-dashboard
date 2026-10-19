import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Empirical Finance Dashboard", layout="wide")
st.title("Empirical Finance Dashboard")

# ---------- Sidebar controls ----------
st.sidebar.header("Settings")
tickers_text = st.sidebar.text_input(
    "Tickers (comma separated)", "AAPL, MSFT, ^GSPC, ASML.AS"
)
tickers = tuple(t.strip().upper() for t in tickers_text.split(",") if t.strip())
start = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("End date", pd.Timestamp.today())
rf = st.sidebar.number_input("Risk-free rate (annual, %)", 0.0, 10.0, 2.0, 0.25) / 100


# ---------- Data ----------
@st.cache_data(ttl=3600)  # refresh from Yahoo at most once per hour
def load_data(tickers, start, end):
    raw = yf.download(list(tickers), start=start, end=end,
                      auto_adjust=True, progress=False)
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    return prices


if not tickers:
    st.warning("Enter at least one ticker.")
    st.stop()

try:
    prices = load_data(tickers, start, end)
except Exception as e:
    st.error(f"Could not download data: {e}")
    st.stop()

prices = prices.ffill().dropna()
if prices.empty or len(prices) < 5:
    st.error("No data returned. Check the tickers and dates.")
    st.stop()

# ---------- Calculations ----------
returns = prices.pct_change().dropna()
years = (prices.index[-1] - prices.index[0]).days / 365.25
total = prices.iloc[-1] / prices.iloc[0] - 1
ann_ret = (1 + total) ** (1 / years) - 1
ann_vol = returns.std() * (252 ** 0.5)
sharpe = (ann_ret - rf) / ann_vol
drawdown = prices / prices.cummax() - 1

stats = pd.DataFrame({
    "Total return": total,
    "Ann. return": ann_ret,
    "Ann. volatility": ann_vol,
    "Sharpe": sharpe,
    "Max drawdown": drawdown.min(),
})

# ---------- Layout ----------
st.subheader("Performance summary")
st.dataframe(
    stats.style.format({
        "Total return": "{:.1%}", "Ann. return": "{:.1%}",
        "Ann. volatility": "{:.1%}", "Sharpe": "{:.2f}",
        "Max drawdown": "{:.1%}",
    }),
    use_container_width=True,
)

col1, col2 = st.columns(2)
with col1:
    rebased = prices / prices.iloc[0] * 100
    st.plotly_chart(px.line(rebased, title="Performance (rebased to 100)"),
                    use_container_width=True)
with col2:
    st.plotly_chart(px.line(drawdown, title="Drawdown"),
                    use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(px.imshow(returns.corr(), text_auto=".2f",
                              color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Correlation of daily returns"),
                    use_container_width=True)
with col4:
    st.plotly_chart(px.scatter(stats, x="Ann. volatility", y="Ann. return",
                               text=stats.index, title="Risk vs return"),
                    use_container_width=True)

st.caption("Data: Yahoo Finance via yfinance. Refreshes hourly.")
