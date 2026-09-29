# Executive Summary — Toronto Island Ferry Ticket Demand Forecasting

## 1. Business Problem & Operational Need
The Toronto Island ferry service operates under severe demand variability driven by weather, weekends, and summer tourist influxes. Operators must make timely decisions regarding vessel staging, crew allocation, and gate crowd control. Currently, decisions are largely reactive. This system provides automated, reliable short-term predictive demand forecasting across **15-minute, 30-minute, 1-hour, and 2-hour** horizons for both **Ticket Sales** and **Ticket Redemptions**.

## 2. Dataset & Empirical Basis
- **Coverage**: Over 10.6 years (May 2015 – December 2025) comprising 261,538 raw transactional observations and 372,509 continuous 15-minute intervals.
- **Data Integrity**: Audited with zero negative values, zero duplicate timestamps, and zero artificial interpolation. Non-operational gaps correspond to overnight terminal closures (~23:00 to ~07:00) with true zero volume.
- **Evaluation**: Chronological test holdout from June 2025 through December 2025 (completely unseen future period).

## 3. Key Forecasting Performance Highlights
- **15-Minute Sales**: The top performing model is **XGBoost** with an MAE of **16.84 tickets/interval** and WAPE of **25.78%** (Peak Miss Rate: 11.85%).
- **1-Hour Sales**: **XGBoost** achieves an MAE of **17.78 tickets/interval** and WAPE of **27.22%**.
- **15-Minute Redemptions**: Top model is **Random Forest** with an MAE of **17.18 tickets/interval** (WAPE: 26.28%).
- **1-Hour Redemptions**: Top model is **Random Forest** with an MAE of **17.86 tickets/interval** (WAPE: 27.33%).

## 4. Operational Recommendations
1. **Queue Staging**: Ticket sales lead redemptions by approximately 15 to 45 minutes. When sales surge, dock gate staff have a 15–30 minute lead time to prepare boarding queues before redemptions peak.
2. **Surge Thresholds**: Demand exceeding 182 sales/15m or 184 redemptions/15m represents the 95th percentile peak operating regime requiring maximum vessel frequency.
3. **Decision Support Nature**: Forecasts provide statistically bounded prediction intervals (e.g. ±28.43 tickets at 90% coverage) to guide supervisory decisions rather than autonomous vessel control.
