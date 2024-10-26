import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta
import numpy as np

# Set up Streamlit app
st.title("Enhanced Stock Price Chart Viewer with Prediction")

# User input for stock ticker
ticker = st.text_input("Enter a stock ticker (e.g., AAPL for Apple):", "AAPL")

# Define date range for the last year and the last week
today = date.today()
start_date = today - timedelta(days=365)
week_start_date = today - timedelta(days=7)

# Download stock data from Yahoo Finance
data = yf.download(ticker, start=start_date, end=today, interval="1d")
weekly_data = yf.download(ticker, start=week_start_date, end=today, interval="1d")

# Display line chart for the last year's Close prices
if not data.empty:
    st.write(f"Displaying the latest closing prices for {ticker}:")
    st.line_chart(data['Close'])

    # Display a data table with the last week's stock data (Open, Close, Volume, Adjusted Close)
    st.write(f"Stock Data for {ticker} (Last Week):")
    st.write(weekly_data[['Open', 'Close', 'Volume', 'Adj Close']])

    # Set up Polynomial Regression for prediction
    data['Days'] = (data.index - data.index[0]).days  # Convert date to numeric days for regression
    X = np.array(data['Days']).reshape(-1, 1)
    y = data['Close']
    
    # Polynomial Regression (Degree 3 for a better trend fit)
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predict future prices based on user input
    days_ahead = st.slider("Days to predict into the future:", 1, 30, 7)
    future_day = np.array([[X[-1][0] + days_ahead]])  # Predict from the last day in dataset
    future_day_poly = poly.transform(future_day)
    future_price = model.predict(future_day_poly)[0]
    
    # Display the prediction result as a float
    st.write(f"Predicted price of {ticker} in {days_ahead} days: ${float(future_price):.2f}")
else:
    st.error(f"No data found for ticker '{ticker}'. Please check the ticker symbol and try again.")
