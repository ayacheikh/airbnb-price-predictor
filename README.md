# Airbnb Price Predictor

A supervised regression project that predicts nightly Airbnb listing prices from real market data. The goal is to estimate one price per listing in a held-out test set and minimize prediction error measured by RMSE.

## Problem

Hosts and guests on Airbnb need to answer a practical question: *what should a listing cost per night?* This project treats that as a **regression problem** — given listing features (location, property details, host info, reviews), predict the continuous target variable `price`.

| Dataset | Rows | Columns | Notes |
|---------|------|---------|-------|
| `listings_train.csv` | 42,234 | 36 | Includes `price` (target) |
| `listings_test_features.csv` | 8,199 | 35 | Features only; predictions required for every row |

Data files are not included in this repository due to size (~34 MB train, ~7 MB test). Download them from the links below and place them in the project root before running the notebook.

- [listings_train.csv](https://drive.google.com/file/d/11hFxV6bO0ccGiRlqaAaLywvkLmRHCx88/view?usp=sharing)
- [listings_test_features.csv](https://drive.google.com/file/d/1f1_Gh40YqUP4PaQT7GFYR7fQxgAh_qYU/view?usp=drive_link)

## Approach

The workflow in `A4 - Data Challenge.ipynb` follows the assignment structure: explore the data, engineer features, compare models on an internal validation split, then generate test predictions.

### 1. Exploratory Data Analysis

- **Problem framing:** Supervised regression with `price` as a continuous target.
- **Data quality:** 36 train columns vs. 35 test columns (only difference is `price`). 18 columns had missing values — mostly empty `neighbourhood_group` and `license`, review-related fields missing together for listings with no reviews, and a small number of missing bedroom/bathroom counts.
- **Distributions:** Price is strongly right-skewed (median **$141**, mean **$166**). Most numeric features share similar skew. Categorical columns range from low cardinality (`room_type`: 4 values) to unusably high (`name`: 41k+ unique values).

### 2. Feature Engineering

| Action | Columns |
|--------|---------|
| Dropped | `name`, `host_name`, `amenities`, `last_review`, `neighbourhood_group`, `license` |
| Median imputation | Numeric columns (medians fit on train, applied to test) |
| One-hot encoding | `room_type`, `host_has_profile_pic`, `host_identity_verified`, `neighbourhood` |

All transformations are fit on the training set and applied consistently to the test set using scikit-learn's `OneHotEncoder` with `handle_unknown='ignore'`.

### 3. Modeling

An 80/20 `train_test_split` (`random_state=42`) was used for internal model selection — the test file was never used for tuning.

| Model | Validation RMSE |
|-------|-----------------|
| Linear Regression | 97.97 |
| Decision Tree (`max_depth=10`) | 67.38 |
| Decision Tree (`max_depth=8`, best tuned) | 66.64 |
| **Random Forest** (`n_estimators=100`, `max_depth=15`) | **56.90** |

Random Forest outperformed simpler baselines and met the assignment's "Exceptional" threshold (RMSE < 65 for undergrad). The final model was retrained on the full training set to produce test predictions.

### 4. Output

Predictions are saved to `A4_predictions_ayacheikh.csv` — a single `price` column with one predicted nightly rate per test row (8,199 predictions).

## Tech Stack

- **Python** — pandas, NumPy, Matplotlib, Seaborn
- **scikit-learn** — `OneHotEncoder`, `train_test_split`, `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, `root_mean_squared_error`

## Project Structure

```
airbnb-price-predictor/
├── A4 - Data Challenge.ipynb   # Full pipeline: EDA → features → models → predictions
├── A4_predictions_ayacheikh.csv # Generated test predictions
└── README.md
```

## Getting Started

1. Clone the repository.
2. Download the two CSV files linked above into the project root.
3. Open and run `A4 - Data Challenge.ipynb` end-to-end.

**Requirements:** Python 3 with pandas, NumPy, matplotlib, seaborn, and scikit-learn installed. The notebook uses `micropip` to install seaborn in browser-based Jupyter environments; in a standard local setup, ensure seaborn is already installed (`pip install seaborn`).

## Key Takeaways

- **Location and property size matter most.** Neighbourhood (one-hot encoded), bedroom/bathroom counts, and room type carry strong signal for price.
- **High-cardinality text fields hurt more than they help** without NLP or embedding — dropping `name`, `host_name`, and `amenities` simplified the pipeline without sacrificing performance.
- **Ensemble methods beat linear models** on this messy, skewed real-world data. Random Forest handled non-linear interactions (e.g., neighbourhood × room type) that linear regression missed.
- **Consistent preprocessing is critical.** Fitting encoders and imputation statistics on train-only data, then applying them to test, avoids data leakage and ensures valid submissions.
