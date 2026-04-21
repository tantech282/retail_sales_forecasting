# 🛒 Retail Sales Demand Forecasting

An end-to-end Machine Learning project for forecasting retail sales using historical data. This project demonstrates how predictive models can help businesses make data-driven decisions for inventory management and sales planning.

---

## 🚀 Overview

This project builds a complete pipeline for retail demand forecasting, including:

* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering for time-series data
* Model training and evaluation
* Future sales prediction

---

## 📊 Dataset

* Walmart Retail Sales Dataset (`train.csv`)
* Key features:

  * Store
  * Date
  * Weekly Sales
  * Holiday Flag

Additional features like month, day of week, and lag variables are engineered during preprocessing.

---

## 🧠 Machine Learning Models

The following models are used and compared:

* Linear Regression
* Random Forest Regressor ✅ *(Best Performing)*
* XGBoost Regressor

---

## ⚙️ Workflow

1. Data Loading
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation (RMSE, MAE, R²)
6. Model Selection
7. Future Forecasting (next 30 days)

---

## 📈 Results

* Random Forest achieved the best performance based on RMSE
* Model successfully predicts future sales trends
* Visualization included for actual vs predicted sales

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* XGBoost

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## 📌 Use Cases

* Retail demand forecasting
* Inventory optimization
* Sales trend analysis
* Business decision support

---

## 📎 Project Structure

```
retail_sales_forecasting/
├── main.py
├── requirements.txt
├── README.md
└── project_report.md
```

---

## ✨ Key Highlights

* End-to-end ML pipeline
* Time-series feature engineering
* Multiple model comparison
* Real-world dataset usage
* Clean and modular code

---

## 📬 Author

Developed as a Machine Learning project for academic and demonstration purposes.
