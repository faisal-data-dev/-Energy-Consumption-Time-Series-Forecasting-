# task_3_energy_forecasting.py

import numpy as np
import pandas as pd
import xgboost as xgb
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# 1. Generate Synthetic Household Power Consumption Dataset
print("Generating and parsing time series data...")
date_range = pd.date_range(start="2026-01-01", end="2026-03-31", freq="h")
data = pd.DataFrame({
    'Global_active_power': 2.0 + np.sin(date_range.hour / 24 * 2 * np.pi) + np.random.normal(0, 0.5, len(date_range))
}, index=date_range)

# 2. Engineer time-based features
data['hour'] = data.index.hour
data['dayofweek'] = data.index.dayofweek
data['is_weekend'] = data['dayofweek'].isin([5, 6]).astype(int)

# Resample to Daily for cleaner forecasting baseline[cite: 1]
df_daily = data[['Global_active_power']].resample('D').mean()
train_size = int(len(df_daily) * 0.8)
train_df, test_df = df_daily.iloc[:train_size], df_daily.iloc[train_size:]

print("Training models...")

# 3. Model 1: ARIMA[cite: 1]
arima_model = ARIMA(train_df['Global_active_power'], order=(5, 1, 0))
arima_fit = arima_model.fit()
arima_preds = arima_fit.forecast(steps=len(test_df)).values

# 4. Model 2: Prophet[cite: 1]
prophet_df = train_df.reset_index().rename(columns={'index': 'ds', 'Global_active_power': 'y'})
prophet_model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
prophet_model.fit(prophet_df)
future = prophet_model.make_future_dataframe(periods=len(test_df), freq='D')
forecast = prophet_model.predict(future)
prophet_preds = forecast['yhat'].iloc[train_size:].values

# 5. Model 3: XGBoost[cite: 1]
df_daily['dayofweek'] = df_daily.index.dayofweek
df_daily['target_lag1'] = df_daily['Global_active_power'].shift(1)
df_daily_dropna = df_daily.dropna()

X = df_daily_dropna[['dayofweek', 'target_lag1']]
y = df_daily_dropna['Global_active_power']

# Re-split due to dropped NaN from lag
X_tr, X_te = X.iloc[:train_size-1], X.iloc[train_size-1:]
y_tr, y_te = y.iloc[:train_size-1], y.iloc[train_size-1:]

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, random_state=42)
xgb_model.fit(X_tr, y_tr)
xgb_preds = xgb_model.predict(X_te)

# 6. Compare performance and evaluate (MAE, RMSE)[cite: 1]
print("\n--- Model Evaluation ---")
print(f"ARIMA   - MAE: {mean_absolute_error(test_df['Global_active_power'], arima_preds):.4f}, RMSE: {np.sqrt(mean_squared_error(test_df['Global_active_power'], arima_preds)):.4f}")
print(f"Prophet - MAE: {mean_absolute_error(test_df['Global_active_power'], prophet_preds):.4f}, RMSE: {np.sqrt(mean_squared_error(test_df['Global_active_power'], prophet_preds)):.4f}")
print(f"XGBoost - MAE: {mean_absolute_error(y_te, xgb_preds):.4f}, RMSE: {np.sqrt(mean_squared_error(y_te, xgb_preds)):.4f}")

# 7. Plot actual vs. forecasted energy usage[cite: 1]
plt.figure(figsize=(12, 6))
plt.plot(train_df.index, train_df['Global_active_power'], label='Train Data')
plt.plot(test_df.index, test_df['Global_active_power'], label='Actual Test Data')
plt.plot(test_df.index, arima_preds, label='ARIMA Forecast', linestyle='--')
plt.plot(test_df.index, prophet_preds, label='Prophet Forecast', linestyle='-.')
plt.plot(y_te.index, xgb_preds, label='XGBoost Forecast', linestyle=':')
plt.title('Actual vs Forecasted Household Energy Consumption')
plt.xlabel('Date')
plt.ylabel('Global Active Power')
plt.legend()
plt.tight_layout()
plt.savefig('energy_forecast_plot.png')
print("\nPlot saved successfully as 'energy_forecast_plot.png'")