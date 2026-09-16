# 🛒 SmartKart Customer Churn Prediction

A complete, beginner-friendly **end-to-end machine learning pipeline** that predicts customer churn for a fictional e-commerce business ("SmartKart") using **Logistic Regression**. Built as a full-code walkthrough of the 15-step ML lifecycle — from raw, messy data to a business-ready churn risk report.

> 📘 Originally developed for an Introduction to AI & ML course (BBA AI/ML, Chitkara Business School), aligned with CLO02: apply data preprocessing, feature selection & ML models to business scenarios, and evaluate performance using appropriate metrics.

## 📊 Overview

SmartKart wants to know: **which customers are likely to churn (stop being customers), and why?**

This notebook takes a deliberately "dirty" 100-row customer dataset — with duplicates, missing values, invalid entries, and outliers — and walks through the full pipeline needed to turn it into an actionable, ranked list of at-risk customers.

## 🧩 Dataset

`SmartKart_dirty_100_rows.csv` — 100 customer records with 5 columns:

| Column | Description |
|---|---|
| `Customer_ID` | Unique customer identifier |
| `Age` | Customer age (contains messy/invalid entries) |
| `Monthly_Spend` | Average monthly spend (₹) |
| `Complaints` | Number of complaints raised |
| `Churn` | Target variable — 1 = churned, 0 = retained |

## 🔬 Pipeline Steps

1. **Data Collection** — load the raw CSV
2. **Data Understanding** — inspect shape, types, missingness, duplicates
3. **Data Cleaning** — strip whitespace, fix types, remove duplicates/invalid values, impute missing values with the median
4. **Outlier Detection & Treatment** — IQR-based capping for `Monthly_Spend` and `Complaints`
5. **Feature Selection** — keep `Age`, `Monthly_Spend`, `Complaints`; drop `Customer_ID`
6. **Define Target Variable** — `Churn`
7. **Encode Target Variable** — verify it's already numeric
8. **Train-Test Split** — 80/20, stratified
9. **Feature Standardisation** — `StandardScaler`, fit on train only
10. **Model Building** — instantiate `LogisticRegression`
11. **Model Training** — fit on scaled training data
12. **Prediction** — predict classes and churn probabilities on the test set
13. **Model Evaluation** — confusion matrix, accuracy, precision, recall, F1-score
14. **Model Interpretation** — read model coefficients to explain *why* customers churn
15. **Final Output** — export a ranked, business-ready churn risk report (CSV)

## 📈 Key Results

- **Recall ≈ 100%** — the model catches essentially every real churner in the test set, which matters most for a churn-prevention use case (a missed churner is a silently lost customer).
- **Accuracy ≈ 89–95%**, **Precision ≈ 83–91%** — a small number of false alarms is an acceptable trade-off against missing real churners.
- **Business insight:** `Monthly_Spend` strongly *reduces* churn risk (high spenders are more loyal), while `Complaints` strongly *increases* it — the clearest lever SmartKart's support team can act on.

## 🛠️ Tech Stack

- Python 3
- `pandas`, `numpy` — data handling
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — preprocessing, modeling, evaluation

## 🚀 Getting Started

1. Clone this repo:
   ```bash
   git clone https://github.com/anchitphutela-max/smartkart-churn-prediction.git
   cd smartkart-churn-prediction
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Launch the notebook:
   ```bash
   jupyter notebook SmartKart_Churn_Prediction_ML_Pipeline.ipynb
   ```
4. Run all cells in order. When prompted in Step 1, upload `SmartKart_dirty_100_rows.csv`.

## 📁 Output

Running the full notebook produces `smartkart_churn_risk_report.csv` — a ranked table of customers labeled **"Likely to Churn"** or **"Not Likely to Churn"**, sorted by churn probability, ready for a retention team to act on.


