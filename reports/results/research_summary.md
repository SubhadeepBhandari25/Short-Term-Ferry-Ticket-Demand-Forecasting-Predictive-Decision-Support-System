# Research Paper Summary — Empirical Evaluation of Short-Term Ferry Demand Forecasting

## Abstract
Short-term passenger flow forecasting in multimodal maritime transit faces unique challenges including asymmetric arrival dynamics, sharp weather-induced demand surges, and high zero-demand frequency during off-peak hours. In this research, we evaluate multi-horizon direct machine learning forecasting models against historical baseline strategies on the comprehensive 10-year Toronto Island Ferry Ticket dataset (2015–2025).

## Methodology
- **Time Cadence**: Standardized 15-minute intervals, capturing operational rhythms without high-frequency noise.
- **Leakage Prevention**: Strictly causal lag structures ($lag_1$ to $lag_{672}$), rolling window statistics, Ontario statutory holiday calendar features, and cyclical trigonometric temporal encodings.
- **Horizons Evaluated**: $h \in \{1, 2, 4, 8\}$ corresponding to 15m, 30m, 1h, and 2h ahead.
- **Models Benchmarked**: Naive Persistence, 1-Hour Moving Average, Ridge Linear Regression, Random Forest, HistGradientBoosting, and XGBoost.
- **Uncertainty Quantification**: Conformal residual quantile estimation on independent validation holdouts.

## Benchmark Results Table (Test Holdout: June–Dec 2025)
Below is the empirical test set performance across all architectures and forecast horizons:

| Target      | Horizon   | Model             |   MAE |   RMSE |   WAPE |   Peak_Miss_Rate |   Margin_90 |
|:------------|:----------|:------------------|------:|-------:|-------:|-----------------:|------------:|
| Sales       | 15m       | Naive             | 22.62 |  87.32 |  34.62 |            17.42 |       37    |
| Sales       | 15m       | Moving Average    | 21.78 |  72.05 |  33.33 |            19.51 |       36.75 |
| Sales       | 15m       | Linear Regression | 18.85 |  64.08 |  28.85 |            14.61 |       29.85 |
| Sales       | 15m       | Random Forest     | 16.91 |  62.77 |  25.88 |            11.81 |       28.78 |
| Sales       | 15m       | Gradient Boosting | 17.03 |  62.82 |  26.07 |            11.93 |       28.88 |
| Sales       | 15m       | XGBoost           | 16.84 |  62.82 |  25.78 |            11.85 |       28.43 |
| Sales       | 30m       | Naive             | 24.19 |  88.64 |  37.02 |            20.23 |       41    |
| Sales       | 30m       | Moving Average    | 24.16 |  74.15 |  36.98 |            23.49 |       40.28 |
| Sales       | 30m       | Linear Regression | 19.82 |  64.9  |  30.33 |            16.5  |       31.05 |
| Sales       | 30m       | Random Forest     | 17.14 |  62.97 |  26.23 |            12.1  |       29.17 |
| Sales       | 30m       | Gradient Boosting | 17.2  |  62.97 |  26.33 |            12.1  |       29.57 |
| Sales       | 30m       | XGBoost           | 17.18 |  63.18 |  26.29 |            11.98 |       29.11 |
| Sales       | 1h        | Naive             | 28.88 |  92.42 |  44.19 |            28.18 |       48    |
| Sales       | 1h        | Moving Average    | 29.79 |  79.71 |  45.59 |            31.07 |       48.75 |
| Sales       | 1h        | Linear Regression | 22.31 |  66.69 |  34.15 |            19.85 |       34.22 |
| Sales       | 1h        | Random Forest     | 17.83 |  63.55 |  27.28 |            13.99 |       29.76 |
| Sales       | 1h        | Gradient Boosting | 17.9  |  63.47 |  27.39 |            13.9  |       30.13 |
| Sales       | 1h        | XGBoost           | 17.78 |  63.58 |  27.22 |            13.36 |       29.67 |
| Sales       | 2h        | Naive             | 39.36 | 102.57 |  60.23 |            41.29 |       65    |
| Sales       | 2h        | Moving Average    | 41.69 |  93.09 |  63.81 |            44.89 |       67    |
| Sales       | 2h        | Linear Regression | 27.47 |  71.25 |  42.04 |            32.96 |       41.97 |
| Sales       | 2h        | Random Forest     | 19.13 |  64.83 |  29.28 |            17.59 |       32.87 |
| Sales       | 2h        | Gradient Boosting | 19.13 |  64.62 |  29.28 |            16.62 |       32.58 |
| Sales       | 2h        | XGBoost           | 19.11 |  66.14 |  29.24 |            16.33 |       31.76 |
| Redemptions | 15m       | Naive             | 23.81 |  89.69 |  36.43 |            20.08 |       39    |
| Redemptions | 15m       | Moving Average    | 23.06 |  75.44 |  35.27 |            21.08 |       37.5  |
| Redemptions | 15m       | Linear Regression | 20.33 |  66.85 |  31.1  |            17.43 |       30.43 |
| Redemptions | 15m       | Random Forest     | 17.18 |  64.26 |  26.28 |            13.29 |       27.08 |
| Redemptions | 15m       | Gradient Boosting | 17.4  |  64.18 |  26.61 |            13.08 |       27.04 |
| Redemptions | 15m       | XGBoost           | 17.26 |  64.2  |  26.4  |            13.13 |       27.12 |
| Redemptions | 30m       | Naive             | 25.69 |  91.42 |  39.29 |            22.57 |       42    |
| Redemptions | 30m       | Moving Average    | 25.6  |  78.32 |  39.16 |            23.89 |       41.28 |
| Redemptions | 30m       | Linear Regression | 21.44 |  67.94 |  32.79 |            18.43 |       31.76 |
| Redemptions | 30m       | Random Forest     | 17.13 |  64.22 |  26.21 |            13.83 |       27.46 |
| Redemptions | 30m       | Gradient Boosting | 17.52 |  64.41 |  26.79 |            13.13 |       28.24 |
| Redemptions | 30m       | XGBoost           | 17.31 |  64.36 |  26.47 |            13.37 |       27.64 |
| Redemptions | 1h        | Naive             | 31.16 |  97.55 |  47.67 |            29.48 |       50    |
| Redemptions | 1h        | Moving Average    | 31.77 |  85.69 |  48.6  |            32.01 |       50.75 |
| Redemptions | 1h        | Linear Regression | 24.38 |  70.64 |  37.3  |            23.15 |       35.52 |
| Redemptions | 1h        | Random Forest     | 17.86 |  64.92 |  27.33 |            15.11 |       28.52 |
| Redemptions | 1h        | Gradient Boosting | 18.27 |  65.15 |  27.95 |            14.7  |       28.68 |
| Redemptions | 1h        | XGBoost           | 18.09 |  65.17 |  27.67 |            14.29 |       28.39 |
| Redemptions | 2h        | Naive             | 42.89 | 110.87 |  65.61 |            41.66 |       70    |
| Redemptions | 2h        | Moving Average    | 45.25 | 103.52 |  69.22 |            45.26 |       73    |
| Redemptions | 2h        | Linear Regression | 30.3  |  76.39 |  46.35 |            35.65 |       43.62 |
| Redemptions | 2h        | Random Forest     | 18.84 |  66.04 |  28.81 |            17.72 |       30.81 |
| Redemptions | 2h        | Gradient Boosting | 19.34 |  66.21 |  29.58 |            17.02 |       31.01 |
| Redemptions | 2h        | XGBoost           | 19.4  |  68.02 |  29.68 |            17.39 |       30.42 |

## Key Findings
1. **Machine Learning vs Baseline**: Gradient boosted ensembles (HistGBR and XGBoost) consistently outperform persistence baselines across all horizons, with an error reduction exceeding 30-40% at longer horizons (1h and 2h).
2. **Lead-Lag Queue Dynamics**: Feature importance demonstrates that sales lags provide high predictive power for redemption forecasting, reflecting the physical reality of ticketing preceding boarding.
3. **Peak Demand Capture**: Standard models tend to underpredict extreme summer holiday spikes; gradient boosted trees with depth constraints provide the lowest Peak Miss Rates.
