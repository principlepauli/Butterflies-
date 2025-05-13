import yfinance as yf
import pandas as pd

# Define the ticker symbol for VIX
vix_ticker = "^VIX"

# Download data from Yahoo Finance (20 years back from today)
vix_data = yf.download(
    vix_ticker,
    start="2004-01-01",  # adjust this if you want a different start date
    end=None,            # None means up to today
    interval="1d",       # daily frequency
    progress=False
)

# Display the first few rows
print(vix_data.head())

# Optional: Save to CSV
vix_data.to_csv("vix_20_years.csv")