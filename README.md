# Toronto Island Ferry Ticket Demand Forecasting & Decision Support System

A production-grade, short-term predictive decision support system for the **Toronto Island Ferry Service**, built on over **10.6 years of historical 15-minute ticketing transaction data (2015–2025)**.

This system delivers multi-horizon predictive forecasts for **Ticket Sales** and **Ticket Redemptions** across **15-minute, 30-minute, 1-hour, and 2-hour** horizons, paired with empirical prediction intervals, holdout benchmark evaluations, and an interactive multi-page Streamlit application.

---

## 1. Project Objective

The Toronto Island ferry service experiences extreme volatility in passenger arrivals, characterized by rapid weather-induced demand surges, weekend leisure influxes, and seasonal peaks. Dispatchers and terminal managers require proactive visibility into incoming crowd volumes to:
- **Deploy auxiliary vessels** on appropriate lead times before docks become overwhelmed.
- **Adjust turnstile gate staffing** and open secondary queue lanes at the Jack Layton Ferry Terminal.
- **Mitigate platform crowding** and communicate accurate boarding wait times to passengers.

This system answers the core question:
> *"Based on historical ferry ticket activity and recent demand patterns, what volume of ticket sales and turnstile redemptions should we expect in the next 15 minutes, 30 minutes, 1 hour, and 2 hours?"*

---

## 2. Dataset Provenance & Profile

- **Source**: City of Toronto Open Data Portal (`Toronto Island Ferry Tickets.csv`).
- **Raw Observations**: 261,538 transactional entries spanning **May 1, 2015 to December 21, 2025**.
- **Continuous 15-Minute Timeline**: 372,509 intervals.
- **Missing Interval Handling**: Analysis revealed that zero entries in the raw ledger contained `Sales Count == 0` AND `Redemption Count == 0`. Missing intervals in the chronological grid correspond to overnight terminal closures (~23:00 to ~07:00) and off-season idle periods with true zero volume. The pipeline reindexes the timeline to a continuous 15-minute grid, filling off-hour gaps with zero transactions rather than distorting demand through artificial interpolation.
- **Data Quality**: 0 duplicate timestamps, 0 negative values, verified non-negative counts.

---

## 3. System Architecture

```text
ferry-demand-forecast/
│
├── data/
│   ├── raw/
│   │   └── Toronto_Island_Ferry_Tickets.csv   # Verified raw transaction dataset
│   └── processed/
│       ├── clean_15min_timeseries.parquet      # Reindexed continuous 15-min series (372,509 rows)
│       └── features_dataset.parquet           # 49 engineered features + multi-horizon targets
│
├── models/
│   └── saved/                                 # Serialized model binaries (.joblib)
│       ├── sales_15m_xgboost.joblib
│       ├── redemption_1h_xgboost.joblib
│       └── prediction_intervals.json          # Empirical conformal residual quantiles
│
├── notebooks/
│   └── eda_and_modeling.ipynb                 # Interactive Jupyter notebook for exploration
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py                     # Cleaning, chronological sort, 15-min reindexing
│   ├── feature_engineering.py                 # Lag generation, rolling windows, holidays, cyclical encodings
│   ├── models.py                              # Naive, MA, Ridge, Random Forest, HistGBR, XGBoost
│   ├── evaluation.py                          # MAE, RMSE, WAPE, sMAPE, Peak Miss Rate, conformal intervals
│   └── visualization.py                       # Plotly & Matplotlib figure generators
│
├── reports/
│   ├── figures/                               # Exported publication EDA figures
│   └── results/
│       ├── model_evaluation_metrics.csv       # Empirical benchmark results across 48 models
│       ├── model_evaluation_metrics.json
│       ├── executive_summary.md               # Stakeholder briefing
│       └── research_summary.md                # Research paper documentation
│
├── app.py                                     # Polished 6-page Streamlit decision support app
├── run_pipeline.py                            # End-to-end training and evaluation CLI
├── requirements.txt                           # Python package dependencies
└── README.md                                  # System documentation
```

---

## 4. Data Preprocessing & Validation

1. **Timestamp Normalization**: Parsed ISO-8601 strings into timezone-naive `datetime64[ns]`.
2. **Chronological Ordering**: Strictly sorted by timestamp.
3. **Regular Frequency Alignment**: Constructed complete 15-minute frequency index from `2015-05-01 13:30:00` to `2025-12-21 22:30:00`.
4. **Zero-Demand Imputation**: Off-operating gaps (111,651 intervals, 29.9%) mapped to $0$ tickets (no phantom demand).
5. **Operating Hours Flag**: Flagged active ferry schedule window (`07:00` to `23:30`).

---

## 5. Feature Engineering (49 Features)

To strictly prevent temporal data leakage, all features for time $t$ utilize exclusively information known at or before time $t$:
- **Lags (Sales & Redemptions)**: $lag_1$ (15m), $lag_2$ (30m), $lag_4$ (1h), $lag_8$ (2h), $lag_{12}$ (3h), $lag_{24}$ (6h), $lag_{96}$ (24h), $lag_{672}$ (7 days).
- **Rolling Windows**: Rolling means ($1h$, $2h$, $6h$, $24h$), rolling standard deviation ($2h$), and rolling min/max ($2h$).
- **Cross-Target Signals**: Sales-to-redemption ratio and differences at $lag_1$ and $lag_4$ to capture queue accumulation.
- **Calendar & Statutory Holidays**: `hour`, `minute`, `day_of_week`, `day_of_month`, `month`, `is_weekend`, `is_summer` (June–Aug), `is_shoulder` (May, Sept), and Ontario statutory holidays (Victoria Day, Canada Day, Civic Holiday, Labour Day, Thanksgiving, etc.).
- **Cyclical Trigonometric Encodings**: Sine/cosine transformations for 24-hour, 7-day, and 12-month periods.

---

## 6. Models & Multi-Horizon Strategy

Direct multi-horizon models ($t+1$, $t+2$, $t+4$, $t+8$) are trained independently to avoid autoregressive error accumulation:
1. **Naive Persistence Baseline**: Predicts the last observed value $\hat{y}_{t+h} = y_t$.
2. **Moving Average Baseline**: Predicts using the rolling 1-hour average of recent intervals.
3. **Linear Regression (Ridge)**: Standardized linear model with L2 regularization.
4. **Random Forest Regressor**: 50 trees, max depth 12, subsampling 50%.
5. **Gradient Boosting Regressor (HistGBR)**: Histogram-binned gradient boosted decision trees (100 iterations, learning rate 0.08).
6. **XGBoost Regressor**: Optimized tree ensemble using `tree_method='hist'` with feature subsampling.

---

## 7. Chronological Evaluation & Empirical Results

Evaluated on an **unseen holdout test set (June 1, 2025 – December 21, 2025; 19,571 intervals)** after training on 2015–2024:

### Ticket Sales Benchmark Summary
| Horizon | Model | MAE (tickets) | RMSE | WAPE (%) | Peak Miss Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **15m** | **XGBoost** | **16.84** | 62.82 | **25.78%** | **11.85%** |
| 15m | Random Forest | 16.91 | 62.77 | 25.88% | 11.81% |
| 15m | Naive Baseline | 22.62 | 87.32 | 34.62% | 17.42% |
| **1h** | **XGBoost** | **17.78** | 63.58 | **27.22%** | **13.36%** |
| 1h | Random Forest | 17.83 | 63.55 | 27.28% | 13.99% |
| 1h | Naive Baseline | 28.88 | 92.42 | 44.19% | 28.18% |
| **2h** | **XGBoost** | **19.11** | 66.14 | **29.24%** | **16.33%** |
| 2h | Random Forest | 19.13 | 64.83 | 29.28% | 17.59% |
| 2h | Naive Baseline | 39.36 | 102.57 | 60.23% | 41.29% |

### Ticket Redemptions Benchmark Summary
| Horizon | Model | MAE (tickets) | RMSE | WAPE (%) | Peak Miss Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **15m** | **Random Forest** | **17.18** | 64.26 | **26.28%** | **13.29%** |
| 15m | XGBoost | 17.26 | 64.20 | 26.40% | 13.13% |
| 15m | Naive Baseline | 23.81 | 89.69 | 36.43% | 20.08% |
| **1h** | **Random Forest** | **17.86** | 64.92 | **27.33%** | **15.11%** |
| 1h | XGBoost | 18.09 | 65.17 | 27.67% | 14.29% |
| 1h | Naive Baseline | 31.16 | 97.55 | 47.67% | 29.48% |
| **2h** | **Random Forest** | **18.84** | 66.04 | **28.81%** | **17.72%** |
| 2h | XGBoost | 19.40 | 68.02 | 29.68% | 17.39% |
| 2h | Naive Baseline | 42.89 | 110.87 | 65.61% | 41.66% |

**Key Finding**: As the forecast horizon extends to 2 hours, the Naive baseline degrades drastically (WAPE increases to >60%, Peak Miss Rate exceeds 41%). Tree-based ML models maintain steady precision (WAPE <30%, Peak Miss Rate <17%), demonstrating decisive operational value.

---

## 8. Prediction Intervals

Prediction uncertainty is modeled via **Conformal Residual Quantiles** computed on the independent validation split (July 2024 – May 2025):
$$\hat{y} \pm q_{1-lpha}(|y - \hat{y}|)$$
This yields empirical, statistically defensible 90% and 95% Prediction Intervals without unwarranted Gaussian assumptions.

---

## 9. Streamlit Application Pages

The application is structured across 6 dedicated pages:
1. **Page 1 — Overview**: Executive KPI cards, latest sales and redemptions, active demand trend indicator, and 48-hour context chart.
2. **Page 2 — Demand Analysis**: Interactive exploratory analytics covering diurnal cycles, weekday vs weekend surges, day-of-week boxplots, monthly seasonality, and sales-versus-redemption lead-lag dynamics.
3. **Page 3 — Forecast**: Interactive forecasting engine where users select Target (Sales/Redemptions), Horizon (15m, 30m, 1h, 2h), and Model architecture. Displays future tabular path and Plotly interactive chart with prediction intervals.
4. **Page 4 — Model Comparison**: Leaderboard comparing MAE, RMSE, WAPE, and Peak Miss Rates across all models and horizons with bar charts.
5. **Page 5 — Operational Insights**: Real-time surge detection against documented 95th percentile thresholds (182 tickets/15min), queue warnings, and decision support matrix.
6. **Page 6 — Data / Model Info**: Transparent provenance documentation, train/val/test split details, feature glossary, and operational disclaimers.

---

## 10. Installation & How to Run

### Prerequisites
Python 3.10+ (tested with Python 3.14 on Windows 64-bit).

### Setup Environment
```bash
cd ferry-demand-forecast
pip install -r requirements.txt
```

### (Optional) Retrain Full Pipeline
To rerun data cleaning, feature engineering, and model training:
```bash
python run_pipeline.py
```

### Launch Streamlit Application
Run the command below:
```bash
streamlit run app.py
```
Or with explicit Python executable:
```bash
python -m streamlit run app.py
```

---

## 11. Project Limitations & Future Improvements

- **Weather Integration**: While calendar and lag signals capture historical seasonal patterns, real-time hourly temperature and precipitation data would further improve rainy-day drop-off predictions.
- **Special Event Schedule**: Direct integration of Toronto Island amphitheatre concert schedules would refine extreme 99th percentile peak forecasting.
- **Online Learning**: Implementing rolling weekly model updates would capture subtle post-pandemic transit shifts.
#   S h o r t - T e r m - F e r r y - T i c k e t - D e m a n d - F o r e c a s t i n g - P r e d i c t i v e - D e c i s i o n - S u p p o r t - S y s t e m  
 