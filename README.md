# Grok-Like Hourly Forecast App

A Streamlit app that analyzes hourly price data and forecasts future prices with a Grok-like assistant.

## How to Run Locally

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```

## How to Use

- Optionally upload a CSV with columns: UTC, Open, High, Low, Close, Volume.
- The app will estimate parameters and forecast hourly prices.
- Compare live vs predicted prices in a table and chart.

## Hosting

- You can deploy on [Streamlit Cloud](https://streamlit.io/cloud) for free.
- Or use any cloud VM with Python.
