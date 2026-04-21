import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

print("🚀 Retail Sales Forecasting (REAL DATA)")

# =========================
# PREPROCESSING
# =========================
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna()
    df = df.sort_values('Date')
    return df

# =========================
# FEATURE ENGINEERING
# =========================
def create_features(df):
    df = df.copy()
    
    # Lag features
    for lag in [1, 7, 30]:
        df[f'Sales_lag_{lag}'] = df.groupby(['Store_ID'])['Sales'].shift(lag)
    
    # Rolling
    df['Sales_rolling_mean_7'] = df.groupby(['Store_ID'])['Sales'].rolling(7).mean().reset_index(0,drop=True)
    
    # Date features
    df['Month'] = df['Date'].dt.month
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
    
    # Encode store
    le = LabelEncoder()
    df['Store_ID_encoded'] = le.fit_transform(df['Store_ID'])
    
    return df, le

# =========================
# MODEL
# =========================
def train_models(X, y):
    split = int(len(X)*0.8)
    
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    models = {
         'Linear Regression': LinearRegression(),
         'Random Forest': RandomForestRegressor(n_estimators=20, random_state=42),
         'XGBoost': XGBRegressor(n_estimators=20, max_depth=3)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)


        results[name] = {"model": model, "rmse": rmse}
        
        print(f"{name} RMSE: {rmse:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   MAE : {mae:.2f}")
        print(f"   R²  : {r2:.3f}")
    
    return results

# =========================
# FORECAST
# =========================
def forecast(df, model, le, features):
    last = df.tail(30)
    last_date = df['Date'].max()
    
    future = []
    
    for i in range(30):
        date = last_date + timedelta(days=i+1)
        
        row = {}
        
        for col in features:
            if col == "Sales_lag_1":
                row[col] = last['Sales'].iloc[-1]
                
            elif col == "Sales_lag_7":
                row[col] = last['Sales'].iloc[-7]
                
            elif col == "Sales_lag_30":
                row[col] = last['Sales'].iloc[-30]
                
            elif col == "Sales_rolling_mean_7":
                row[col] = last['Sales'].tail(7).mean()
                
            elif col == "Month":
                row[col] = date.month
                
            elif col == "Day_of_Week":
                row[col] = date.dayofweek
                
            elif col == "Is_Weekend":
                row[col] = int(date.dayofweek >= 5)
                
            elif col == "Store_ID_encoded":
                row[col] = le.transform([last['Store_ID'].iloc[-1]])[0]
            
            else:
                # 🔥 Important: unknown features (Dept, IsHoliday etc.)
                row[col] = 0
        
        # Convert to DataFrame with correct column order
        row_df = pd.DataFrame([row])[features]
        
        pred = model.predict(row_df)[0]
        
        future.append([date, pred])
    
    return pd.DataFrame(future, columns=["Date","Predicted_Sales"])
# =========================
# MAIN
# =========================
def main():
    df = pd.read_csv("train.csv")
    df = df.sample(5000, random_state=42)
    
    df.rename(columns={
        "Store":"Store_ID",
        "Weekly_Sales":"Sales"
    }, inplace=True)
    
    df = preprocess_data(df)
    
    df, le = create_features(df)
    
    df = df.fillna(0)
    
    features = [c for c in df.columns if c not in ['Date','Sales','Store_ID']]
    
    X = df[features]
    y = df['Sales']
    
    results = train_models(X,y)
    
    best = min(results, key=lambda x: results[x]['rmse'])
    print("Best Model:", best)
    
    future = forecast(df, results[best]['model'], le, features)
    
    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'].tail(100), df['Sales'].tail(100), label="Actual")
    plt.plot(future['Date'], future['Predicted_Sales'], label="Forecast")
    plt.legend()
    plt.show()

main()