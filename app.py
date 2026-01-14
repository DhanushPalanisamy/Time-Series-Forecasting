from flask import Flask, render_template
import pandas as pd
import joblib
import plotly.graph_objs as go

app = Flask(__name__)

# Load data and model
df = pd.read_csv("data/sales.csv", parse_dates=["Date"])
model = joblib.load("model/arima_model.pkl")

@app.route("/")
def index():
    # Forecast next 6 months
    forecast = model.forecast(steps=6)

    # Create plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Sales"],
        mode="lines+markers",
        name="Historical Sales"
    ))

    fig.add_trace(go.Scatter(
        x=pd.date_range(df["Date"].iloc[-1], periods=7, freq="M")[1:],
        y=forecast,
        mode="lines+markers",
        name="Forecast"
    ))

    fig.update_layout(title="Sales Forecasting")

    graph_html = fig.to_html(full_html=False)

    return render_template("index.html", graph=graph_html)

if __name__ == "__main__":
    app.run(debug=True)
