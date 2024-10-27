
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Define a function to switch language based on button press
def translate_to_russian(is_russian):
    if is_russian:
        return {
            "title": "Просмотр графика и прогноз акций",
            "ticker_prompt": "Введите биржевой тикер (например, AAPL для Apple):",
            "chart_title": "Отображение цен закрытия за последний месяц:",
            "weekly_data_title": "Данные за последнюю неделю:",
            "prediction_prompt": "Дни для предсказания в будущем:",
            "prediction_result": "Предсказанная цена {} через {} дней: ${:.2f}",
            "mse_result": "Среднеквадратичная ошибка модели: {:.2f}",
            "translate_button": "Перевести на английский",
        }
    else:
        return {
            "title": "Stock Price Chart and Enhanced Prediction",
            "ticker_prompt": "Enter a stock ticker (e.g., AAPL for Apple):",
            "chart_title": "Displaying the last month's closing prices:",
            "weekly_data_title": "Stock Data for the Last Week:",
            "prediction_prompt": "Days to predict into the future:",
            "prediction_result": "Predicted price of {} in {} days: ${:.2f}",
            "mse_result": "Mean Squared Error of the model: {:.2f}",
            "translate_button": "Translate to Russian",
        }

# Track if the page should be in Russian
is_russian = st.session_state.get("is_russian", False)

# Button to toggle language
if st.button(translate_to_russian(is_russian)["translate_button"]):
    is_russian = not is_russian
    st.session_state["is_russian"] = is_russian

# Get translations based on the selected language
translations = translate_to_russian(is_russian)

# Set up Streamlit app with translated title
st.title(translations["title"])

# User input for stock ticker
ticker = st.text_input(translations["ticker_prompt"], "AAPL")

if ticker:
    today = date.today()
    month_start_date = today - timedelta(days=30)  # Last 30 days
    week_start_date = today - timedelta(days=7)    # Last week

    # Download last month's stock data
    data = yf.download(ticker, start=month_start_date, end=today, interval="1d")
    
    if not data.empty:
        # Chart for the last month's close prices
        st.write(translations["chart_title"])
        st.line_chart(data['Close'])

        # Weekly data for table display
        weekly_data = yf.download(ticker, start=week_start_date, end=today, interval="1d")
        if not weekly_data.empty:
            st.write(translations["weekly_data_title"])
            st.write(weekly_data[['Open', 'Close', 'Volume', 'Adj Close']])
        else:
            st.error("Failed to fetch weekly data.")

        # Enhanced Prediction Model (Polynomial Regression Degree 4)
        data['Days'] = (data.index - data.index.min()).days
        X = np.array(data['Days']).reshape(-1, 1)
        y = data['Close']
        
        poly = PolynomialFeatures(degree=4)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)

        # Calculate Mean Squared Error
        y_pred = model.predict(X_poly)
        mse = mean_squared_error(y, y_pred)
        st.write(translations["mse_result"].format(mse))

        # Future prediction based on user input
        days_ahead = st.slider(translations["prediction_prompt"], 1, 14, 7)  # Focus on 1-14 day predictions
        future_day = np.array([[X[-1][0] + days_ahead]])
        future_day_poly = poly.transform(future_day)
        future_price = model.predict(future_day_poly)[0]
        
        st.write(translations["prediction_result"].format(ticker, days_ahead, float(future_price)))
    else:
        st.error("No data found for the given ticker. Please check the ticker symbol and try again.")
else:
    st.error("Please enter a ticker symbol.")
