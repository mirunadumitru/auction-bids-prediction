# Auction Bids Prediction Dashboard

This project predicts and analyzes **daily auction bid counts** using multiple machine learning and time series models.  
The results are visualized in an **interactive Streamlit dashboard**, where performance metrics and plots for each model can be explored.

## Project Overview

The aim of this project was to compare the performance of different forecasting approaches on historical auction bid data and build an accessible interface for result visualization.

We trained and evaluated:
- **Naive Baseline** - simple last-value prediction
- **Linear Regression** - classical regression model
- **Random Forest** - ensemble-based regression
- **Prophet** - time-series model with trend & seasonality decomposition

## Dataset

The dataset contains **daily bid counts** from an online auction platform.  
The target variable: `num_bids` , total number of bids per day.


## Methods

1. **Data Preprocessing**
   - Parsed and indexed by date
   - Aggregated daily bid counts
   - Feature engineering (day of week, lag features, etc.)

2. **Model Training & Evaluation**
   - Models trained on a training set, tested on a holdout set
   - Metrics used:
     - MAE 
     - RMSE 
     - MAPE 

3. **Visualization**
   - Matplotlib/Seaborn for exploratory plots
   - Streamlit for interactive dashboard
   - Comparison bar charts and per-model scatter plots

## Results Summary

| Model             | MAE  | RMSE | MAPE (%) |
|-------------------|------|------|----------|
| Random Forest     | 1.45 | 1.78 | 8.55     |
| Linear Regression | 1.58 | 1.85 | 9.25     |
| Naive Baseline    | 3.24 | 3.84 | 19.39    |
| Prophet           | 6.47 | 8.28 | 9.08     |

**Key Takeaways:**
- Random Forest performed best overall in MAE and RMSE.
- Prophet had competitive percentage error (MAPE) but higher absolute errors.
- Naive Baseline performed worst, confirming the added value of the models.

## Dashboard Features

The **Streamlit dashboard** allows you to:
- View performance metrics in a table
- Compare models side-by-side in a **bar chart**
- Explore **per-model forecast vs. actual plots**
- Inspect Prophet’s **trend, seasonality, and daily components**

## How to Run

### Clone the repository
```bash
git clone https://github.com/mirunadumitru/auction-bids-prediction.git
cd auction-bids-prediction
