# 📈 Yes Bank Stock Price Prediction 🔮💸

> Predicting the future is tough — but not with machine learning! 🚀  
> In this project, we forecast Yes Bank’s 📉 closing stock prices using data-driven ML models and analytics.

---

## 🧠 Overview

This project explores how various machine learning regression techniques can be applied to predict **Yes Bank’s monthly closing stock price** using its historical stock data 📊.

We walk through the full ML pipeline including:
- 🔍 Exploratory Data Analysis (EDA)
- 🧪 Hypothesis Testing
- 🛠️ Feature Engineering
- 🧮 Model Building (Linear, KNN, Random Forest)
- 🔁 Cross-Validation + Grid Search
- 🧾 Evaluation Metrics & Business Impact
- 💾 Model Saving and Reuse

---

## 📦 Dataset

- Features:
  - `Open`, `High`, `Low`, `Close_Price`, `Total_Trade_Quantity`, `Turnover`
  - + Engineered Features: `Year`, `Month`

---

## 🚀 Models Used

| Model               | Technique         | R² Score |
|--------------------|-------------------|----------|
| Linear Regression  | Baseline Model    | ~0.99    |
| KNN Regressor      | GridSearchCV (k=2)| ~0.99    |
| 🌳 Random Forest    | GridSearchCV + CV | **~0.98** |

---

## 🧪 Hypothesis Testing

Examples:
- H0: Trade Quantity does not affect Close Price  
- H1: Trade Quantity affects Close Price ➡️ **P-value < 0.05 ✅ Significant**

---

## 📊 Visualizations

> We followed the UBM rule:  
> **U**nivariate, **B**ivariate, **M**ultivariate Analysis ✅

✅ Boxplots  
✅ Line Charts  
✅ Correlation Heatmap  
✅ Pair Plots  
✅ Scatter Plots  
✅ KDE Distributions  
✅ Interactive insights 🚀

🖼️ *(Add your chart screenshots here!)*

---

## 🔐 Model Saving & Reloading

```python
import joblib

# Save the best model
joblib.dump(rf, 'random_forest_model.pkl')

# Load & predict
model = joblib.load('random_forest_model.pkl')
prediction = model.predict(x_test)
