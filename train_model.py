import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib
import os

# Load data
df = pd.read_csv("data/sales.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# Train ARIMA model
model = ARIMA(df["Sales"], order=(1,1,1))
model_fit = model.fit()

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model_fit, "model/arima_model.pkl")

print("✅ ARIMA model trained and saved")
