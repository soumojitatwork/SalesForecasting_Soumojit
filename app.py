"""
Intelligent Sales Forecasting Dashboard
Streamlit app for Superstore sales forecasting project (Task 7).

Run locally:   streamlit run app.py
Deploy free:   push this folder to a public GitHub repo, then go to
               https://share.streamlit.io -> New app -> pick the repo -> app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sns.set_style("whitegrid")
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")


# ----------------------------------------------------------------------------
# Data loading (cached so it only runs once per session)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter

    def season(m):
        return ("Winter" if m in [12, 1, 2] else
                "Spring" if m in [3, 4, 5] else
                "Summer" if m in [6, 7, 8] else "Fall")

    df["Season"] = df["Month"].apply(season)
    return df


@st.cache_data
def get_daily_weekly_monthly(df):
    daily = df.groupby("Order Date")["Sales"].sum().asfreq("D").fillna(0)
    weekly = daily.resample("W").sum()
    monthly = daily.resample("MS").sum()
    return daily, weekly, monthly


def mae(a, f):
    return float(np.mean(np.abs(np.array(a) - np.array(f))))


def rmse(a, f):
    return float(np.sqrt(np.mean((np.array(a) - np.array(f)) ** 2)))


@st.cache_data
def fit_sarima_and_forecast(monthly_series, horizon):
    train = monthly_series.iloc[:-3] if len(monthly_series) > 15 else monthly_series
    test = monthly_series.iloc[-3:] if len(monthly_series) > 15 else monthly_series.iloc[-min(3, len(monthly_series)):]
    try:
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                         enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        test_fc = fit.get_forecast(steps=len(test)).predicted_mean
        m = mae(test, test_fc)
        r = rmse(test, test_fc)

        full_model = SARIMAX(monthly_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                              enforce_stationarity=False, enforce_invertibility=False)
        full_fit = full_model.fit(disp=False)
        forecast_res = full_fit.get_forecast(steps=horizon)
        forecast = forecast_res.predicted_mean
        ci = forecast_res.conf_int()
        return forecast, ci, m, r
    except Exception:
        naive = pd.Series([monthly_series.iloc[-3:].mean()] * horizon,
                           index=pd.date_range(monthly_series.index[-1] + pd.DateOffset(months=1),
                                                periods=horizon, freq="MS"))
        return naive, None, np.nan, np.nan


df = load_data()
daily_sales, weekly_sales, monthly_sales = get_daily_weekly_monthly(df)

st.sidebar.title("📦 Sales Forecasting System")
page = st.sidebar.radio(
    "Navigate",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Superstore Sales — Time Series Forecasting, Anomaly Detection & "
    "Product Segmentation dashboard."
)

# ----------------------------------------------------------------------------
# PAGE 1 — Sales Overview
# ----------------------------------------------------------------------------
if page == "Sales Overview":
    st.title("📊 Sales Overview Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    col2.metric("Total Orders", f"{df['Order ID'].nunique():,}")
    col3.metric("Avg Order Value", f"${df.groupby('Order ID')['Sales'].sum().mean():,.2f}")

    st.subheader("Total Sales by Year")
    yearly = df.groupby("Year")["Sales"].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    yearly.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylabel("Sales ($)")
    st.pyplot(fig)

    st.subheader("Monthly Sales Trend")
    fig, ax = plt.subplots(figsize=(12, 4))
    monthly_sales.plot(ax=ax, color="#DD8452", marker="o")
    ax.set_ylabel("Sales ($)")
    st.pyplot(fig)

    st.subheader("Sales by Region & Category")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        region_filter = st.multiselect("Region", options=sorted(df["Region"].unique()),
                                        default=sorted(df["Region"].unique()))
    with fcol2:
        category_filter = st.multiselect("Category", options=sorted(df["Category"].unique()),
                                          default=sorted(df["Category"].unique()))

    filtered = df[df["Region"].isin(region_filter) & df["Category"].isin(category_filter)]
    pivot = filtered.groupby(["Region", "Category"])["Sales"].sum().unstack("Category").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Sales ($)")
    st.pyplot(fig)
    st.dataframe(pivot.style.format("${:,.0f}"))

# ----------------------------------------------------------------------------
# PAGE 2 — Forecast Explorer
# ----------------------------------------------------------------------------
elif page == "Forecast Explorer":
    st.title("🔮 Forecast Explorer")

    dim = st.selectbox("Select dimension", ["Category", "Region"])
    options = sorted(df[dim].unique())
    selection = st.selectbox(f"Select {dim}", options)
    horizon = st.select_slider("Forecast horizon (months ahead)", options=[1, 2, 3], value=3)

    seg_df = df[df[dim] == selection]
    seg_daily = seg_df.groupby("Order Date")["Sales"].sum().asfreq("D").fillna(0)
    seg_monthly = seg_daily.resample("MS").sum()

    with st.spinner("Fitting model..."):
        forecast, ci, m, r = fit_sarima_and_forecast(seg_monthly, horizon)

    fig, ax = plt.subplots(figsize=(12, 5))
    seg_monthly.iloc[-18:].plot(ax=ax, label="Actual", color="#4C72B0", marker="o")
    forecast.plot(ax=ax, label="Forecast", color="#DD8452", marker="o")
    if ci is not None:
        ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="#DD8452", alpha=0.2)
    ax.set_title(f"{selection} — {horizon}-Month Sales Forecast")
    ax.set_ylabel("Sales ($)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Forecast values")
    st.dataframe(forecast.rename("Forecasted Sales ($)").to_frame().style.format("${:,.0f}"))

    st.subheader("Model performance (on held-out test months)")
    mcol1, mcol2 = st.columns(2)
    mcol1.metric("MAE", f"${m:,.0f}" if not np.isnan(m) else "N/A")
    mcol2.metric("RMSE", f"${r:,.0f}" if not np.isnan(r) else "N/A")

# ----------------------------------------------------------------------------
# PAGE 3 — Anomaly Report
# ----------------------------------------------------------------------------
elif page == "Anomaly Report":
    st.title("🚨 Anomaly Report")

    anomaly_df = weekly_sales.to_frame("sales").copy()
    iso = IsolationForest(contamination=0.05, random_state=42)
    anomaly_df["iso_anomaly"] = iso.fit_predict(anomaly_df[["sales"]])

    window = 8
    anomaly_df["rolling_mean"] = anomaly_df["sales"].rolling(window, center=True, min_periods=3).mean()
    anomaly_df["rolling_std"] = anomaly_df["sales"].rolling(window, center=True, min_periods=3).std()
    anomaly_df["z_score"] = (anomaly_df["sales"] - anomaly_df["rolling_mean"]) / anomaly_df["rolling_std"]
    anomaly_df["zscore_anomaly"] = anomaly_df["z_score"].abs() > 2

    iso_anom = anomaly_df[anomaly_df["iso_anomaly"] == -1]
    z_anom = anomaly_df[anomaly_df["zscore_anomaly"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(anomaly_df.index, anomaly_df["sales"], color="#4C72B0", label="Weekly Sales")
    ax.scatter(iso_anom.index, iso_anom["sales"], color="red", marker="X", s=100,
               label="Isolation Forest Anomaly", zorder=5)
    ax.scatter(z_anom.index, z_anom["sales"], facecolors="none", edgecolors="orange",
               marker="o", s=180, linewidths=2, label="Z-Score Anomaly", zorder=4)
    ax.set_ylabel("Sales ($)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Detected anomaly weeks")
    tab1, tab2 = st.tabs(["Isolation Forest", "Z-Score"])
    with tab1:
        st.dataframe(iso_anom[["sales"]].rename(columns={"sales": "Sales ($)"}).style.format("${:,.0f}"))
    with tab2:
        st.dataframe(z_anom[["sales", "z_score"]].rename(
            columns={"sales": "Sales ($)", "z_score": "Z-Score"}).style.format({"Sales ($)": "${:,.0f}", "Z-Score": "{:.2f}"}))

    st.info(
        "Interpretation: sharp spikes typically align with Nov/Dec promotional "
        "periods; sharp drops often follow immediately after (post-holiday lull). "
        "Weeks flagged by both methods are the highest-confidence anomalies."
    )

# ----------------------------------------------------------------------------
# PAGE 4 — Product Demand Segments
# ----------------------------------------------------------------------------
elif page == "Product Demand Segments":
    st.title("🧩 Product Demand Segments")

    subcat = df.groupby("Sub-Category").apply(
        lambda g: pd.Series({
            "total_sales": g["Sales"].sum(),
            "avg_order_value": g["Sales"].mean(),
            "sales_volatility": g.groupby(g["Order Date"].dt.to_period("M"))["Sales"].sum().std(),
        })
    ).reset_index()

    yoy = df.groupby(["Sub-Category", "Year"])["Sales"].sum().unstack("Year")
    yoy_growth = yoy.pct_change(axis=1).mean(axis=1) * 100
    subcat = subcat.merge(yoy_growth.rename("yoy_growth_pct"), on="Sub-Category").fillna(0)

    cluster_cols = ["total_sales", "yoy_growth_pct", "sales_volatility", "avg_order_value"]
    X_scaled = StandardScaler().fit_transform(subcat[cluster_cols])

    k = st.slider("Number of clusters (K)", 2, 6, 4)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    subcat["cluster"] = kmeans.fit_predict(X_scaled)

    cluster_summary = subcat.groupby("cluster")[cluster_cols].mean()

    def assign_cluster_labels(cluster_summary):
        cs = cluster_summary.copy()
        cs["sales_rank"] = cs["total_sales"].rank(ascending=False)
        cs["volatility_rank"] = cs["sales_volatility"].rank(ascending=True)
        cs["stability_score"] = cs["sales_rank"] + cs["volatility_rank"]

        labels = {}
        remaining = list(cs.index)

        if remaining:
            c = cs.loc[remaining, "stability_score"].idxmin()
            labels[c] = "High Volume, Stable Demand"
            remaining.remove(c)
        if remaining:
            c = cs.loc[remaining, "yoy_growth_pct"].idxmax()
            labels[c] = "Growing Demand"
            remaining.remove(c)
        if remaining:
            c = cs.loc[remaining, "yoy_growth_pct"].idxmin()
            labels[c] = "Declining Demand" if cs.loc[c, "yoy_growth_pct"] < 0 else "Slower, Steadier Demand"
            remaining.remove(c)
        for c in remaining:
            labels[c] = "Low Volume, High Volatility"
        return labels

    labels = assign_cluster_labels(cluster_summary)
    subcat["cluster_label"] = subcat["cluster"].map(labels)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    subcat["pca1"], subcat["pca2"] = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=(9, 6))
    for label in subcat["cluster_label"].unique():
        subset = subcat[subcat["cluster_label"] == label]
        ax.scatter(subset["pca1"], subset["pca2"], label=label, s=120)
        for _, row in subset.iterrows():
            ax.annotate(row["Sub-Category"], (row["pca1"], row["pca2"]), fontsize=8)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Sub-categories by demand cluster")
    st.dataframe(
        subcat[["Sub-Category", "cluster_label", "total_sales", "yoy_growth_pct", "sales_volatility"]]
        .sort_values("cluster_label")
        .style.format({"total_sales": "${:,.0f}", "yoy_growth_pct": "{:.1f}%", "sales_volatility": "${:,.0f}"})
    )

    st.subheader("Recommended stocking strategy")
    st.markdown(
        "- **High Volume, Stable Demand** → steady safety stock, fixed reorder schedule\n"
        "- **Growing Demand** → increase reorder quantities ahead of trend\n"
        "- **Declining Demand** → reduce purchase orders, consider clearance\n"
        "- **Low Volume, High Volatility** → lean stock, faster replenishment cycles"
    )
