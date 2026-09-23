"""
BrewBox Coffee — Sales Analysis & Forecasting
A Streamlit app answering three questions for Kavya Reddy (BrewBox Coffee, Whitefield, Bengaluru):
  1. Is the café actually growing?
  2. When is she busy, and when is she wasting money on staff and stock?
  3. How much will she take over the next fortnight?

Deploy: push this file + requirements.txt + brewbox_daily_sales.csv to a GitHub repo,
then deploy on share.streamlit.io pointing at this file.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

DEFAULT_DATA_PATH = "brewbox_daily_sales.csv"
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.set_page_config(page_title="BrewBox Sales Analysis", page_icon="☕", layout="wide")


# ----------------------------------------------------------------------------
# Data loading & cleaning
# ----------------------------------------------------------------------------
@st.cache_data
def load_and_clean(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.set_index("Date")

    # Fill missing days by interpolating within each weekday's own series,
    # so a missing Tuesday is filled from nearby Tuesdays, not neighbouring days.
    for col in ["Cups_Sold", "Sales_Rs"]:
        if col in df.columns:
            df[col] = (
                df.groupby("Day_Name")[col]
                .apply(lambda s: s.interpolate(method="linear"))
                .reset_index(level=0, drop=True)
            )
    df = df.reset_index()
    df["Month"] = df["Date"].dt.to_period("M")
    df["Year"] = df["Date"].dt.year
    df["MonthNum"] = df["Date"].dt.month
    df["is_weekend"] = df["Day_Name"].isin(["Saturday", "Sunday"])
    return df


def count_missing(raw: pd.DataFrame) -> int:
    return int(raw["Sales_Rs"].isna().sum())


# ----------------------------------------------------------------------------
# Forecasting helpers
# ----------------------------------------------------------------------------
def score(actual: np.ndarray, pred: np.ndarray):
    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    return mae, rmse, mape


def run_baselines(sales: pd.Series, test_days: int = 28) -> pd.DataFrame:
    train, test = sales[:-test_days], sales[-test_days:]
    n_tiles = int(np.ceil(test_days / 7))

    preds = {
        "Naive": np.repeat(train.iloc[-1], test_days),
        "Average": np.repeat(train.mean(), test_days),
        "Moving Average (7d)": np.repeat(train.tail(7).mean(), test_days),
        "Seasonal Naive": np.tile(train.tail(7).values, n_tiles)[:test_days],
    }
    rows = []
    for name, pred in preds.items():
        mae, rmse, mape = score(test.values, pred)
        rows.append([name, mae, rmse, mape])
    return pd.DataFrame(rows, columns=["Method", "MAE", "RMSE", "MAPE (%)"]).sort_values("MAE").reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def fit_prophet(data: pd.DataFrame, holidays: pd.DataFrame):
    from prophet import Prophet

    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, holidays=holidays)
    model.fit(data)
    return model


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("☕ BrewBox Coffee")
st.sidebar.caption("Sales analysis & forecasting — Whitefield, Bengaluru")

uploaded = st.sidebar.file_uploader("Upload daily sales CSV", type="csv")
horizon = st.sidebar.slider("Forecast horizon (days)", min_value=7, max_value=30, value=14, step=1)
test_days = st.sidebar.slider("Back-test window (days)", min_value=14, max_value=56, value=28, step=7)

data_source = uploaded if uploaded is not None else DEFAULT_DATA_PATH
try:
    df = load_and_clean(data_source)
except FileNotFoundError:
    st.error(
        f"Couldn't find `{DEFAULT_DATA_PATH}` next to this app, and no file was uploaded. "
        "Upload a CSV with columns Date, Day_Name, Cups_Sold, Sales_Rs, Public_Holiday."
    )
    st.stop()

raw = pd.read_csv(data_source) if isinstance(data_source, str) else pd.read_csv(uploaded)
missing_days = count_missing(raw)

st.sidebar.metric("Days in dataset", len(df))
st.sidebar.metric("Missing days (fixed)", missing_days)


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("BrewBox Coffee — Sales Analysis & Forecasting")
st.caption(
    f"{df['Date'].min().date()} → {df['Date'].max().date()} · "
    f"{len(df)} days · average daily sales ₹{df['Sales_Rs'].mean():,.0f}"
)

tab_growth, tab_pattern, tab_backtest, tab_forecast = st.tabs(
    ["📈 Is it growing?", "🗓️ Busy vs. quiet", "🧪 Back-test", "🔮 Forecast"]
)


# ----------------------------------------------------------------------------
# Tab 1 — Growth
# ----------------------------------------------------------------------------
with tab_growth:
    st.subheader("Is BrewBox actually growing?")

    sales_by_date = df.set_index("Date")["Sales_Rs"].asfreq("D")

    fig, ax = plt.subplots(figsize=(10, 4))
    sales_by_date.rolling(7).mean().plot(ax=ax, color="#6f4e37")
    ax.set_title("7-Day Rolling Average of Sales")
    ax.set_ylabel("Sales (₹)")
    st.pyplot(fig)

    last_month = df["Date"].max().month
    ytd_now = df[(df["Year"] == df["Date"].max().year) & (df["MonthNum"] <= last_month)]
    prev_year = df["Date"].max().year - 1
    ytd_prev = df[(df["Year"] == prev_year) & (df["MonthNum"] <= last_month)]

    col1, col2, col3 = st.columns(3)
    if len(ytd_prev) > 0:
        growth = (ytd_now["Sales_Rs"].sum() - ytd_prev["Sales_Rs"].sum()) / ytd_prev["Sales_Rs"].sum() * 100
        col1.metric("Year-on-year growth", f"{growth:.1f}%")
    col2.metric("Avg. price per cup", f"₹{(df['Sales_Rs'].sum() / df['Cups_Sold'].sum()):.0f}"
                if "Cups_Sold" in df.columns else "n/a")
    col3.metric("Total sales in dataset", f"₹{df['Sales_Rs'].sum():,.0f}")

    monthly = df.groupby("Month")["Sales_Rs"].sum()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    monthly.plot(kind="bar", ax=ax2, color="#6f4e37")
    ax2.set_title("Total Sales by Month")
    ax2.set_ylabel("Sales (₹)")
    plt.xticks(rotation=60)
    st.pyplot(fig2)


# ----------------------------------------------------------------------------
# Tab 2 — Weekly / monthly / holiday pattern
# ----------------------------------------------------------------------------
with tab_pattern:
    st.subheader("When is she busy, and when is she wasting money?")

    by_day = df.groupby("Day_Name")["Sales_Rs"].mean().reindex(DAY_ORDER)
    busiest, quietest = by_day.idxmax(), by_day.idxmin()

    c1, c2 = st.columns(2)
    with c1:
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        colors = ["#2e7d32" if d == busiest else ("#c62828" if d == quietest else "#a86f43") for d in DAY_ORDER]
        ax3.bar(by_day.index, by_day.values, color=colors)
        ax3.set_title("Average Sales by Day of Week")
        ax3.set_ylabel("Sales (₹)")
        plt.xticks(rotation=45)
        st.pyplot(fig3)
        st.caption(
            f"Busiest: **{busiest}** (₹{by_day[busiest]:,.0f}) · "
            f"Quietest: **{quietest}** (₹{by_day[quietest]:,.0f}) · "
            f"{by_day[busiest] / by_day[quietest]:.1f}× gap"
        )

    with c2:
        if "Public_Holiday" in df.columns:
            hol = df.groupby("Public_Holiday")["Sales_Rs"].mean()
            fig4, ax4 = plt.subplots(figsize=(8, 4))
            ax4.bar(["Ordinary day", "Public holiday"], [hol.get("No", 0), hol.get("Yes", 0)],
                    color=["#a86f43", "#c62828"])
            ax4.set_title("Ordinary Day vs. Public Holiday")
            ax4.set_ylabel("Average Sales (₹)")
            st.pyplot(fig4)
            if "No" in hol and "Yes" in hol:
                drop = (hol["No"] - hol["Yes"]) / hol["No"] * 100
                st.caption(f"A public holiday takes **{drop:.0f}% less** than an ordinary day.")

    st.markdown("**Monthly pattern**")
    by_month = df.groupby("MonthNum")["Sales_Rs"].mean()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    by_month.index = [month_names[i - 1] for i in by_month.index]
    fig5, ax5 = plt.subplots(figsize=(10, 3.5))
    ax5.bar(by_month.index, by_month.values, color="#a86f43")
    ax5.set_ylabel("Avg. daily sales (₹)")
    st.pyplot(fig5)


# ----------------------------------------------------------------------------
# Tab 3 — Back-test the baseline methods
# ----------------------------------------------------------------------------
with tab_backtest:
    st.subheader(f"Scoring four simple methods on the last {test_days} days")
    sales_by_date = df.set_index("Date")["Sales_Rs"].asfreq("D")

    if len(sales_by_date) <= test_days:
        st.warning("Not enough history for this back-test window — reduce it in the sidebar.")
    else:
        scoreboard = run_baselines(sales_by_date, test_days=test_days)
        st.dataframe(
            scoreboard.style.format({"MAE": "₹{:.0f}", "RMSE": "₹{:.0f}", "MAPE (%)": "{:.1f}%"})
            .highlight_min(subset=["MAE"], color="#dff0d8"),
            width="stretch",
        )
        winner = scoreboard.iloc[0]
        st.caption(
            f"**{winner['Method']}** wins with a typical (MAE) miss of about ₹{winner['MAE']:,.0f} a day."
        )


# ----------------------------------------------------------------------------
# Tab 4 — Prophet forecast
# ----------------------------------------------------------------------------
with tab_forecast:
    st.subheader(f"Forecast for the next {horizon} days")

    try:
        prophet_data = df[["Date", "Sales_Rs"]].rename(columns={"Date": "ds", "Sales_Rs": "y"})
        if "Public_Holiday" in df.columns:
            holidays = df.loc[df["Public_Holiday"] == "Yes", ["Date"]].rename(columns={"Date": "ds"})
            holidays["holiday"] = "public_holiday"
        else:
            holidays = pd.DataFrame(columns=["ds", "holiday"])

        with st.spinner("Training Prophet model…"):
            model = fit_prophet(prophet_data, holidays)
            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)

        fig6 = model.plot(forecast)
        plt.title(f"History & {horizon}-Day Forecast")
        plt.ylabel("Sales (₹)")
        st.pyplot(fig6)

        next_n = forecast[forecast["ds"] > prophet_data["ds"].max()][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        next_n["Day"] = next_n["ds"].dt.day_name()
        next_n = next_n.rename(
            columns={"ds": "Date", "yhat": "Forecast (₹)", "yhat_lower": "Low (₹)", "yhat_upper": "High (₹)"}
        )[["Date", "Day", "Forecast (₹)", "Low (₹)", "High (₹)"]]
        for c in ["Forecast (₹)", "Low (₹)", "High (₹)"]:
            next_n[c] = next_n[c].round(0)

        total = next_n["Forecast (₹)"].sum()
        c1, c2 = st.columns([1, 2])
        c1.metric(f"Total forecast, next {horizon} days", f"₹{total:,.0f}")
        c1.metric("Average forecast day", f"₹{total / horizon:,.0f}")
        c2.dataframe(next_n, width="stretch", hide_index=True)

        with st.expander("Trend, weekly, and holiday components"):
            fig7 = model.plot_components(forecast)
            st.pyplot(fig7)

        st.caption(
            "This forecast assumes no unplanned public holidays or one-off events in the window. "
            "Trust it fully for the next week or so — treat anything further out as a rough guide."
        )
    except ImportError:
        st.error(
            "Prophet isn't installed. Add `prophet` to requirements.txt to enable this tab."
        )
