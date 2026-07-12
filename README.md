# Sales Forecasting & Demand Intelligence System

An end-to-end data science project built on the Superstore Sales dataset - covering time-series analysis, multi-model forecasting, anomaly detection, product segmentation, and a live interactive dashboard. Built as a capstone project to practice the full pipeline a real data science team would own: from messy CSVs to a business-ready tool.

> Superstore sales forecasting project: time-series decomposition, SARIMA/Prophet/XGBoost comparison, anomaly detection (Isolation Forest + Z-score), K-Means product segmentation, and a Streamlit dashboard for inventory planning.

---

## Live Demo
🔗 **http://localhost:8501/**

## Problem Statement
Retailers lose money in both directions when demand forecasting goes wrong — overstock ties up capital and storage, understock loses sales and customers. This project builds a system that predicts future product demand, flags unusual sales weeks, groups products by demand behavior, and puts all of it in front of a business user through an interactive dashboard.

---

## Features / Tasks Covered

| # | Task | What it does |
|---|---|---|
| 1 | **EDA & Feature Engineering** | Parses dates, extracts time features, aggregates to weekly/monthly granularity, answers key business questions with data |
| 2 | **Time Series Decomposition** | Trend/Seasonal/Residual breakdown + Augmented Dickey-Fuller stationarity testing |
| 3 | **3-Model Forecasting** | SARIMA, Facebook Prophet, and XGBoost compared side-by-side on MAE/RMSE/MAPE |
| 4 | **Segment-Level Forecasting** | Category & region-level forecasts using the best-performing model |
| 5 | **Anomaly Detection** | Isolation Forest + rolling Z-score, plus a multi-source merge exercise with the Video Game Sales dataset |
| 6 | **Product Segmentation** | K-Means clustering (elbow method) + PCA visualization, with relative (not fixed-threshold) cluster labeling |
| 7 | **Interactive Dashboard** | 4-page Streamlit app: Sales Overview, Forecast Explorer, Anomaly Report, Demand Segments |
| 8 | **Executive Report** | 2-page business-facing summary (`summary.docx`) — no code, no jargon |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data | Pandas, NumPy |
| Time Series | Statsmodels (SARIMA, decomposition, ADF test) |
| Forecasting | Facebook Prophet, XGBoost |
| ML | Scikit-learn (Isolation Forest, K-Means, PCA) |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Notebook | Jupyter |

---

## Repository Structure

```
SalesForecasting_Student/
├── analysis.ipynb            # Full analysis notebook (Tasks 1–6), self-installing & pre-executed
├── app.py                     # Streamlit dashboard (Task 7)
├── train.csv                  # Superstore Sales dataset
├── vgsales.csv                 # Video Game Sales dataset (multi-source merge exercise)
├── requirements.txt           # Dependencies for the dashboard
├── summary.docx                # Executive business report (Task 8)
├── model_comparison.csv        # SARIMA vs Prophet vs XGBoost metrics
├── cluster_assignments.csv     # Sub-category → demand cluster mapping
├── charts/                     # All chart PNGs exported from the notebook
└── README.md
```

---

## Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Run the notebook
`analysis.ipynb` installs its own dependencies in the first cell (`%pip install ...`), so you can just open it and run all cells top to bottom.
```bash
jupyter notebook analysis.ipynb
```
> First run only: if any packages were missing beforehand, restart the kernel once after the install cell finishes, then re-run from the top.

### 3. Run the dashboard
```bash
pip install -r requirements.txt
streamlit run app.py
```
Make sure you `cd` into this project folder first — `app.py` looks for `train.csv` sitting right next to it.

### 4. Deploy the dashboard (free)
1. Push this repo to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select this repo → set main file to `app.py` → **Deploy**.
3. Copy the live `*.streamlit.app` URL.

---

## Model Results

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| **SARIMA (recommended)** | **13,336** | **13,340** | **14.8%** |
| Prophet | 20,296 | 22,487 | 21.9% |
| XGBoost | 23,835 | 29,464 | 23.3% |

SARIMA performed best here because the monthly sales series turned out to already be statistically stationary (confirmed via ADF test) with a strong, regular seasonal pattern — exactly the setting a seasonal ARIMA model is designed for.

## Product Demand Clusters

Every sub-category grew year-over-year in this dataset (no outright decline), so clusters are labeled *relative to each other* rather than by fixed thresholds:

- **High Volume, Stable Demand** - Bookcases, Paper, Furnishings, Appliances, Art, Envelopes, Fasteners, Labels
- **Growing Demand** - Supplies
- **Slower, Steadier Demand** - Chairs, Phones, Tables, Storage, Binders, Accessories
- **Low Volume, High Volatility** - Copiers, Machines

---

## Challenges Faced

- **Date format mismatch** - the dataset used `DD/MM/YYYY`, not `MM/DD/YYYY`; pandas parsed it wrong silently for the first 12 days of each month instead of erroring out.
- **Choosing SARIMA's (p,d,q) parameters** - ran a small grid search on AIC instead of guessing blindly.
- **Comparing 3 different models fairly** - SARIMA, Prophet, and XGBoost each need different input formats, so ensuring all three were evaluated on the exact same held-out months took extra care.
- **Cluster labels breaking on fixed thresholds** - an initial "growth > 5% = Growing Demand" rule labeled every cluster identically since all sub-categories happened to grow; switched to relative labeling instead.
- **Isolation Forest vs Z-score disagreeing a lot** - the two methods agreed on only 1 anomaly out of ~15 flagged combined, a good reminder that "anomaly" isn't a single fixed definition.
- **Environment/setup friction** - missing packages, PowerShell not recognizing `!pip`, and `FileNotFoundError` from running Streamlit outside the project folder ate up more debugging time than the modeling itself.

