# House_Price_Prediction

Predicting house prices from property features using a tuned Random Forest regression pipeline, with exploratory data analysis to identify key price drivers.

## Overview

This project builds an end-to-end regression pipeline to predict house prices from property and location features. It covers EDA, feature preprocessing, model training with hyperparameter tuning, and evaluation — a full workflow from raw data to a validated model.

## Dataset

**California Housing Dataset** (via `sklearn.datasets.fetch_california_housing`) — 20,640 records derived from 1990 U.S. Census data, one row per California census block group.

| Feature | Description |
|---|---|
| `MedInc` | Median income in block group (in $10,000s) |
| `HouseAge` | Median house age in block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average household occupancy |
| `Latitude` / `Longitude` | Location coordinates |
| `MedHouseVal` | **Target** — median house value (in $100,000s) |

## Approach

1. **Load data** — fetched directly via scikit-learn, no manual download required.
2. **EDA** — visualized price against income, room count, location (lat/long), house age, and population; built a correlation heatmap to identify the strongest price drivers.
3. **Preprocessing** — `StandardScaler` applied to all features (all-numeric dataset, no categorical encoding needed).
4. **Model** — `RandomForestRegressor` inside a scikit-learn `Pipeline`.
5. **Hyperparameter tuning** — `RandomizedSearchCV` (10 iterations, 3-fold CV) over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.
6. **Evaluation** — MAE, RMSE, R² on a held-out 20% test set.
7. **Visualization** — predicted-vs-actual scatter plot, residual distribution, and feature importance chart.

## Results

| Metric | Value |
|---|---|
| CV RMSE (best model, training) | 0.5010 |
| Test MAE | 0.3289 ($32,890) |
| Test RMSE | 0.4959 ($49,590) |
| Test R² | **0.8124** |

**Best hyperparameters:** `n_estimators=200`, `max_depth=20`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features='sqrt'`

The model explains ~81% of the variance in median house value. `MedInc` (median income) is the strongest price driver, consistent with the correlation analysis (~0.69 correlation with price).

## Key Findings from EDA

- **Median income** is the dominant driver of house price, by a wide margin over any other feature.
- **Location** (latitude/longitude) shows clear price clustering, consistent with coastal areas commanding higher values.
- **Average rooms** shows only a weak positive relationship with price once income is accounted for.
- **Population** and **average occupancy** show weak negative relationships with price.

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (`RandomForestRegressor`, `RandomizedSearchCV`, `Pipeline`, `StandardScaler`)
- matplotlib, seaborn

## Project Structure

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run

```bash
python house_price_prediction.py
```

Or open the notebook version in Google Colab — no dataset upload needed, it loads automatically via scikit-learn. Full run (including hyperparameter search) takes roughly 1–2 minutes on a standard Colab CPU instance.

## Notes on Dataset Selection

This project initially used a Kaggle "House Price Prediction Dataset" CSV (`Area`, `Bedrooms`, `Bathrooms`, `Floors`, `YearBuilt`, `Location`, `Condition`, `Garage`). EDA revealed near-zero correlation between every feature and price (max correlation: 0.056), and a tuned Random Forest scored **R² = -0.01** on it — worse than predicting the mean. This indicated the dataset's price values weren't generated with any real dependency on the listed features.

The project switched to California Housing, a dataset with genuine feature-price relationships, to demonstrate a working, evaluable pipeline. This is included here as part of the documented process — verifying a dataset has learnable signal is a standard and necessary step before investing in modeling it.

## License

MIT
