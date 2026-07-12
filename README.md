# Sales Forecasting & Demand Intelligence System

Superstore sales forecasting, anomaly detection, product segmentation, and
dashboard, course capstone project.

## Data
Both datasets are the real Kaggle files:
- `train.csv` - Superstore Sales (`rohitsahoo/sales-forecasting`), 9,800 rows, 2015-2018.
  **Note:** dates in this file are stored day-first (`DD/MM/YYYY`), which the
  notebook and `app.py` both parse accordingly (`format="%d/%m/%Y"`).
- `vgsales.csv` - Video Game Sales (`gregorut/videogamesales`), used in Task 5
  for the required multi-source merge exercise: Superstore `Technology`
  category sales are merged with yearly global video-game sales on the `Year`
  key. Only 3 years overlap (2015-2017) between the two datasets' coverage, so
  the correlation shown there is a methodology demonstration rather than a
  statistically robust signal — this caveat is called out directly in the
  notebook.

## Contents
| File | Task | Description |
|---|---|---|
| `analysis.ipynb` | 1-6 | Full analysis notebook, already executed with outputs & charts |
| `train.csv` | -- | Superstore sales dataset |
| `vgsales.csv` | 5 | Video Game Sales dataset, used for the multi-source merge exercise |
| `app.py` | 7 | Streamlit dashboard (4 pages) |
| `requirements.txt` | 7 | Dependencies for redeploying `app.py` |
| `summary.docx` | 8 | 2-page executive business report |
| `charts/` | 1-6 | All chart PNGs saved from the notebook |
| `model_comparison.csv` | 3 | SARIMA vs Prophet vs XGBoost metrics |
| `cluster_assignments.csv` | 6 | Sub-category → demand cluster mapping |

## Running locally
`analysis.ipynb` now installs its own dependencies - the very first code cell
runs `%pip install pandas numpy matplotlib seaborn statsmodels prophet xgboost
scikit-learn` automatically. Just open the notebook and run cells top to
bottom; no separate terminal `pip install` step is required for the notebook.
(If any of those packages weren't already installed, restart the kernel once
right after that first cell finishes, then re-run from the top - a fresh
install isn't picked up by an already-running kernel until it restarts.)

For the dashboard (`app.py`), install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
jupyter notebook analysis.ipynb     # re-run the analysis (self-installing)
streamlit run app.py                # launch the dashboard
```

## Deploying the dashboard (Streamlit Community Cloud, free)
1. Push this whole folder to a **public** GitHub repo (must include `app.py`,
   `train.csv`, and `requirements.txt` at minimum).
2. Go to https://share.streamlit.io → **New app** → select the repo, branch,
   and set the main file path to `app.py`.
3. Click **Deploy** - first build takes a few minutes (installs Prophet et al.
   from `requirements.txt`).
4. Copy the live `*.streamlit.app` URL for submission.

## Model results summary (on real Superstore data, 3-month held-out test)
| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| **SARIMA (recommended)** | **13,336** | **13,340** | **14.8%** |
| Prophet | 20,296 | 22,487 | 21.9% |
| XGBoost | 23,835 | 29,464 | 23.3% |

SARIMA won this time because the monthly sales series turned out to already
be statistically stationary (see the ADF test in Task 2) with a strong,
regular seasonal pattern, exactly the setting a seasonal ARIMA model is
built for. Prophet and XGBoost are kept in the notebook as cross-checks.

## Product demand clusters (Task 6)
Every sub-category grew year-over-year in this dataset (no outright decline),
so clusters are labeled by *relative* growth/stability rather than fixed
thresholds:
- **High Volume, Stable Demand** - Bookcases, Paper, Furnishings, Appliances, Art, Envelopes, Fasteners, Labels
- **Growing Demand** - Supplies
- **Slower, Steadier Demand** - Chairs, Phones, Tables, Storage, Binders, Accessories
- **Low Volume, High Volatility** - Copiers, Machines
