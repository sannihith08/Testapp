import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import plotly.express as px
from datetime import timedelta, time, date
from statsmodels.tsa.arima.model import ARIMA
import os 
import secrets
from ta.volatility import BollingerBands
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM, MultiHeadAttention, Input
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from company import company as company
from company import code as code
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
import traceback
from plotly.subplots import make_subplots
import ta
import pytz
from sklearn.metrics import mean_absolute_error, mean_squared_error
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import xgboost as xgb

tz = pytz.timezone("Asia/Kolkata")  # Indian time zone

PROJECT_FOLDER = ''
sma_100 = 0
sma_50 = 0
ema_100 = 0
ema_50 = 0
WINDOW = 5
pmae = 0
pmse = 0
prmse = 0
pmape = 0
accuracy = 0

if "previous_symbol" not in st.session_state:
    st.session_state.previous_symbol = ""

if "project_folder" not in st.session_state:
    st.session_state.project_folder = ""

if "last_menu_option" not in st.session_state:
    st.session_state.last_menu_option = None


def generate_market_time_range_1_minute(start_date, periods):
    """
    Generates a range of timestamps within stock market hours (9:15 AM to 3:30 PM) for a given number of periods.
    Each period corresponds to a 1-minute interval.
    """
    market_open = time(9, 15)  # 9:15 AM
    market_close = time(15, 30)  # 3:30 PM
    timestamps = []
    current_date = start_date.date()
    count = 0

    while count < periods:
        # Generate intraday 1-minute intervals within trading hours
        current_time = datetime.datetime.combine(current_date, market_open)
        while current_time.time() <= market_close and count < periods:
            timestamps.append(current_time)
            current_time += timedelta(minutes=1)
            count += 1
        current_date += timedelta(days=1)  # Move to the next day
    return pd.DatetimeIndex(timestamps)

def generate_market_time_range_5_minute(start_date, periods):
    """
    Generates a range of timestamps within stock market hours (9:15 AM to 3:30 PM) for a given number of periods.
    Each period corresponds to a 5-minute interval.
    """
    market_open = time(9, 15)  # 9:15 AM
    market_close = time(15, 30)  # 3:30 PM
    timestamps = []
    current_date = start_date.date()
    count = 0

    while count < periods:
        # Generate intraday 5-minute intervals within trading hours
        current_time = datetime.datetime.combine(current_date, market_open)
        while current_time.time() <= market_close and count < periods:
            timestamps.append(current_time)
            current_time += timedelta(minutes=5)
            count += 1
        current_date += timedelta(days=1)  # Move to the next day
    return pd.DatetimeIndex(timestamps)

def generate_market_time_range(start_date, periods, interval_minutes):
    """
    Generates a range of timestamps within stock market hours (9:15 AM to 3:30 PM) for a given number of periods.
    """
    market_open = time(9, 15)   # 9:15 AM
    market_close = time(15, 30) # 3:30 PM
    timestamps = []
    
    # Check if start_date has a timezone
    if start_date.tzinfo is not None:
        tz = start_date.tzinfo  # Get the timezone
    else:
        tz = None  # Naive datetime

    current_datetime = start_date
    count = 0

    while count < periods:
        # Ensure we start within market hours
        if current_datetime.time() < market_open:
            current_datetime = datetime.datetime.combine(current_datetime.date(), market_open)
        elif current_datetime.time() > market_close:
            current_datetime = datetime.datetime.combine(current_datetime.date() + timedelta(days=1), market_open)

        # Apply timezone if necessary
        if tz:
            current_datetime = current_datetime.replace(tzinfo=tz)

        # Generate timestamps within trading hours
        while current_datetime.time() <= market_close and count < periods:
            timestamps.append(current_datetime)
            current_datetime += timedelta(minutes=interval_minutes)
            count += 1
            
            # Stop if next timestamp exceeds market close
            if current_datetime.time() > market_close:
                current_datetime = datetime.datetime.combine(current_datetime.date() + timedelta(days=1), market_open)
                if tz:
                    current_datetime = current_datetime.replace(tzinfo=tz)

    return pd.DatetimeIndex(timestamps)


def generate_market_time_range_daily(start_date, days):
    """
    Generates a range of timestamps for the next `days` trading days.
    Each timestamp corresponds to the market close time (3:30 PM).
    """
    market_close = time(15, 30)  # 3:30 PM
    timestamps = []
    current_date = start_date.date()
    trading_days = 0

    while trading_days < days:
        # Skip weekends (Saturday and Sunday)
        if current_date.weekday() < 5:  # Monday to Friday are trading days
            timestamps.append(datetime.datetime.combine(current_date, market_close))
            trading_days += 1
        current_date += timedelta(days=1)  # Move to the next day
    
    return pd.DatetimeIndex(timestamps)


def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

def plotActualPredictedValue(test_predictions_baseline):
    try:
        fig = go.Figure()
        # Add predicted price line
        fig.add_trace(go.Scatter(
            x=test_predictions_baseline['Datetime'],
            y=test_predictions_baseline[stock_symbol + '_predicted'],
            mode='lines',
            name=stock_symbol + ' Predicted Price',
            line=dict(color='red')
        ))

        fig.add_trace(go.Scatter(
            x=test_predictions_baseline['Datetime'],
            y=test_predictions_baseline[stock_symbol + '_actual'],
            mode='lines',
            name=stock_symbol + ' Actual Price',
            line=dict(color='green')
        ))

        # Update layout
        fig.update_layout(
            title= stock_symbol + ' Prediction vs Actual',
            xaxis_title='Datetime',
            yaxis_title='Price',
            template='plotly_dark',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            height=600
        )

        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Error in plotting graph: {e}")

def EMA(Close_arr, n):
    a = 2 / (n + 1)
    EMA_n = np.zeros(len(Close_arr))
    EMA_n[:n] = np.nan

    # Initialize the first EMA value
    EMA_n[n] = np.mean(Close_arr[:n])

    # Calculate EMA for the rest of the values
    for i in range(n + 1, len(Close_arr)):
        EMA_n[i] = (Close_arr[i] - EMA_n[i - 1]) * a + EMA_n[i - 1]

    return EMA_n

def gains(Close_arr):
    gain_arr = np.diff(Close_arr)
    gain_arr[gain_arr < 0] = 0
    return gain_arr

def losses(Close_arr):
    loss_arr = np.diff(Close_arr)
    loss_arr[loss_arr > 0] = 0
    return np.abs(loss_arr)

def RSI(Close_arr, n=14):
    gain_arr = gains(Close_arr)
    loss_arr = losses(Close_arr)

    EMA_u = EMA(gain_arr, n)
    EMA_d = EMA(loss_arr, n)

    EMA_diff = EMA_u / EMA_d

    RSI_n = 100 - (100 / (1 + EMA_diff))
    RSI_n = np.concatenate((np.full(n, np.nan), RSI_n))  # Align lengths by padding initial values with NaN
    return RSI_n

def closeRSIPrediction(close_data):
    try:
        if 'Predicted Close Price' in close_data.columns:
            Close = close_data['Predicted Close Price'].values
            RSI3 = RSI(Close, n=3)
            
            # Ensure lengths match
            RSI3 = RSI3[:len(close_data)]
            
            close_data['RSI3'] = RSI3

            # Plot RSI using Plotly
            fig = go.Figure()

            # Add RSI Line
            fig.add_trace(go.Scatter(
                x=close_data['Datetime'],
                y=close_data['RSI3'],
                mode='lines',
                name='RSI (3-period)',
                line=dict(color='blue', width=2)
            ))

            # Add Overbought (70) and Oversold (30) Reference Lines
            fig.add_trace(go.Scatter(
                x=close_data['Datetime'],
                y=[70] * len(close_data),
                mode='lines',
                name='Overbought (70)',
                line=dict(color='red', width=1, dash='dash')
            ))

            fig.add_trace(go.Scatter(
                x=close_data['Datetime'],
                y=[30] * len(close_data),
                mode='lines',
                name='Oversold (30)',
                line=dict(color='green', width=1, dash='dash')
            ))

            # Layout settings
            fig.update_layout(
                title="RSI (Relative Strength Index)",
                xaxis_title="Date",
                yaxis_title="RSI Value",
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True, zeroline=True),
                legend=dict(x=0, y=1),
                height=500
            )

            # Display in Streamlit
            st.plotly_chart(fig)
        else:
            st.error("Column 'Predicted Close Price' not found in the dataset.")
    except Exception as e:
        print(f"Getting Error in predicting overbought and oversold: {e}")

def closeRSIPredictionForSwing(close_data):
    try:
        if 'Predicted Close Price' in close_data.columns:
            Close = close_data['Predicted Close Price'].values
            RSI4 = RSI(Close, n=4)
            
            # Ensure lengths match
            RSI4 = RSI4[:len(close_data)]
            
            close_data['RSI4'] = RSI4

            # Plot RSI using Plotly
            fig = go.Figure()

            # Add RSI Line
            fig.add_trace(go.Scatter(
                x=close_data['Datetime'],
                y=close_data['RSI4'],
                mode='lines',
                name='RSI (4-day)',
                line=dict(color='blue', width=2)
            ))

            # Add Overbought (70) and Oversold (30) Reference Lines
            fig.add_trace(go.Scatter(
                x=close_data['Datetime'],
                y=[70] * len(close_data),
                mode='lines',
                name='Overbought (70)',
                line=dict(color='red', width=1, dash='dash')
            ))

            fig.add_trace(go.Scatter(
                x=close_data['Datetime'],
                y=[30] * len(close_data),
                mode='lines',
                name='Oversold (30)',
                line=dict(color='green', width=1, dash='dash')
            ))

            # Layout settings
            fig.update_layout(
                title="RSI (Relative Strength Index)",
                xaxis_title="Date",
                yaxis_title="RSI Value",
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True, zeroline=True),
                legend=dict(x=0, y=1),
                height=500
            )

            # Display in Streamlit
            st.plotly_chart(fig)
        else:
            st.error("Column 'Predicted Close Price' not found in the dataset.")
    except Exception as e:
        print(f"Getting Error in predicting overbought and oversold: {e}")

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    data['RSI'] = rsi
    return data

# Calculate MACD
def calculate_macd(data, short=12, long=26, signal=9):
    data['EMA12'] = data['Close'].ewm(span=short, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=long, adjust=False).mean()
    
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=signal, adjust=False).mean()
    
    return data

# Calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std=2):
    data['SMA'] = data['Close'].rolling(window).mean()
    data['Upper_Band'] = data['SMA'] + (data['Close'].rolling(window).std() * num_std)
    data['Lower_Band'] = data['SMA'] - (data['Close'].rolling(window).std() * num_std)
    return data

# Calculate Stochastic Oscillator
def calculate_stochastic(data, k_window=14, d_window=3):
    low_min = data['Low'].rolling(k_window).min()
    high_max = data['High'].rolling(k_window).max()
    
    data['%K'] = 100 * (data['Close'] - low_min) / (high_max - low_min)
    data['%D'] = data['%K'].rolling(d_window).mean()
    return data

# Calculate ATR
def calculate_atr(data, atr_period=14):
    data['High-Low'] = data['High'] - data['Low']
    data['High-Close'] = abs(data['High'] - data['Close'].shift(1))
    data['Low-Close'] = abs(data['Low'] - data['Close'].shift(1))

    data['True Range'] = data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    data['ATR'] = data['True Range'].rolling(window=atr_period).mean()
    
    return data

# This should now generate Buy signals when:

# Strategy 1
# RSI is below 35.
# MACD is bullish.
# Stochastic %K crosses %D upwards.
# Price is near the lower Bollinger Band.
# And Sell signals when:

# RSI is above 65.
# MACD is bearish.
# Stochastic %K crosses %D downwards.
# Price is near the upper Bollinger Band.

# def generate_signals(data):
#     buy_signals = []
#     sell_signals = []
    
#     for i in range(len(data)):
#         if (data['RSI'][i] < 35 and data['MACD'][i] > data['Signal_Line'][i] and 
#             data['%K'][i] < 20 and data['%K'][i] > data['%D'][i] and
#             data['Close'][i] <= data['Lower_Band'][i]):
#             buy_signals.append(data['Close'][i])
#             sell_signals.append(np.nan)
#         elif (data['RSI'][i] > 65 and data['MACD'][i] < data['Signal_Line'][i] and 
#               data['%K'][i] > 80 and data['%K'][i] < data['%D'][i] and
#               data['Close'][i] >= data['Upper_Band'][i]):
#             sell_signals.append(data['Close'][i])
#             buy_signals.append(np.nan)
#         else:
#             buy_signals.append(np.nan)
#             sell_signals.append(np.nan)
    
#     data['Buy_Signal'] = buy_signals
#     data['Sell_Signal'] = sell_signals
#     return data

#Strategy 2
# The strategy uses the Stochastic Oscillator to generate buy signals when `%K` crosses above `%D` below 20, 
# confirming with higher volume, and sell signals when `%K` crosses below `%D` above 80, also 
# confirmed with higher volume. It includes stop-loss and take-profit based on ATR values to limit losses 
# and secure profits.

# Generate buy/sell signals
def generate_signals(data):
    signals = []
    is_trade_active = False
    entry_price = None
    stop_loss = None
    take_profit = None
    volume_ma = data['Volume'].rolling(window=20).mean()  # 20-period volume moving average

    for i in range(len(data)):
        if np.isnan(data['ATR'][i]) or np.isnan(data['%K'][i]) or np.isnan(data['%D'][i]):
            signals.append("None")
            continue

        atr_value = data['ATR'][i]
        current_price = data['Close'][i]
        current_volume = data['Volume'][i]
        current_volume_ma = volume_ma[i]

        # Buy Signal (Stochastic Oscillator and Volume Confirmation)
        if data['%K'][i] < 20 and data['%K'][i] > data['%D'][i] and not is_trade_active:
            if current_volume > current_volume_ma:  # Confirm with volume
                entry_price = current_price
                stop_loss = entry_price - (1.5 * atr_value)
                take_profit = entry_price + (2 * atr_value)
                signals.append("Strong BUY")
                is_trade_active = True
            else:
                signals.append("Neutral")

        # Sell Signal (Stochastic Oscillator and Volume Confirmation)
        elif data['%K'][i] > 80 and data['%K'][i] < data['%D'][i] and is_trade_active:
            if current_volume > current_volume_ma:  # Confirm with volume
                signals.append("SELL")
                is_trade_active = False
            else:
                signals.append("Neutral")

        # Stop-Loss or Take-Profit conditions
        elif is_trade_active:
            if current_price <= stop_loss:
                signals.append("Stop-Loss Hit")
                is_trade_active = False
            elif current_price >= take_profit:
                signals.append("Take-Profit Hit")
                is_trade_active = False
            else:
                signals.append("Neutral")
        else:
            signals.append("Neutral")

    data['B/S_Signal'] = signals  # Add new column for signals
    return data

def time_feature(data):
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data['Weekday'] = data['Datetime'].dt.weekday  # 0=Monday, 6=Sunday
    data['Month'] = data['Datetime'].dt.month  # 1-12
    data['Quarter'] = data['Datetime'].dt.quarter  # 1-4
    data['Year'] = (data['Datetime'].dt.year - data['Datetime'].dt.year.min()) / (data['Datetime'].dt.year.max() - data['Datetime'].dt.year.min())  # Normalize year
    return data

def get_accuracy(actual, predicted):
    # Calculate MAE
    pmae = mean_absolute_error(actual, predicted)
    print(f"Mean Absolute Error (MAE): {pmae}")

    # Calculate MSE
    pmse = mean_squared_error(actual, predicted)
    print(f"Mean Squared Error (MSE): {pmse}")

    # Calculate RMSE
    prmse = pmse ** 0.5
    print(f"Root Mean Squared Error (RMSE): {prmse}")

    # Calculate MAPE
    pmape = np.mean(np.abs((actual - predicted) / actual)) * 100
    print(f"Mean Absolute Percentage Error (MAPE): {pmape:.2f}%")

    # Calculate accuracy
    accuracy = 100 - pmape
    print(f"Accuracy: {accuracy:.2f}%")
    st.markdown(
        f"<p style='font-size:12px;'>"
        f"<b>Mean Absolute Error (MAE):</b> {pmae:.4f} | "
        f"<b>Mean Squared Error (MSE):</b> {pmse:.4f} | "
        f"<b>Root Mean Squared Error (RMSE):</b> {prmse:.4f} | "
        f"<b>Mean Absolute Percentage Error (MAPE):</b> {pmape:.2f}% | "
        f"<b>Accuracy:</b> {accuracy:.2f}%"
        f"</p>",
        unsafe_allow_html=True
    )

# Header formatting using Markdown and CSS
header_style = """
    <style>
        .header {
            color: #0077b6;
            font-size: 48px;
            text-shadow: 2px 2px 2px rgba(0, 0, 0, 0.15);
        }
        .subheader {
            color: #0077b6;
            font-size: 30px;
        }
    </style>
"""

# Apply header formatting to the title
st.markdown(header_style, unsafe_allow_html=True)
st.markdown('<h1 class="header">Stock Price Forecast</h1>', unsafe_allow_html=True)

# Sidebar with prediction options
# Sidebar header with styling
st.sidebar.markdown("""
    <h2 style="color: #0077b6; text-align: center;">Prediction Strategy</h2>
    <p style="font-size: 14px; color: #4a4a4a; text-align: justify;">
        Choose a prediction strategy to forecast stock prices. Each strategy has its strengths:
        <ul>
            <li><b>Day Trading Forcast</b></li>
            <li><b>Swing Trading Forcast</b></li>
        </ul>
    </p>
""", unsafe_allow_html=True)

# Add the radio button menu with icons
menu_option = st.sidebar.radio(
    "🔮 Select Prediction Strategy",
    ("🧠 Day Trading Forcast", "📈 Swing Trading Forcast"),
    help="Select one of the prediction trading to forecast stock prices."
)

# if st.session_state.last_menu_option != menu_option:
#     st.session_state.last_menu_option = menu_option
#     st.experimental_rerun()  # Trigger a rerun when the option is switched


# Parse the selected option to extract the model
if "Day" in menu_option:
    menu_option = "Day Trading Forcast"
elif "Swing" in menu_option:
    menu_option = "Swing Trading Forcast"


# User input for stock symbol
selected_company = st.selectbox("\n\nSearch for a Company", company)
st.write("---")
n = company.index(selected_company)
stock_symbol = code[n]
st.write(stock_symbol)


EPOCHS = 64
BATCH_SIZE = 32
TIME_STEPS = 60
PROJECT_FOLDER = st.session_state.project_folder
# print("PROJECT_FOLDER: ", PROJECT_FOLDER)
scaler = MinMaxScaler(feature_range=(0, 1))

def stockInfo(stock_symbol):
    try:
        # Quick Stock Quote Assessment
        st.markdown('<h2 class="subheader">Stock Quote</h2>', unsafe_allow_html=True)

        # Fetch stock quote using yfinance
        quote_data = yf.Ticker(stock_symbol)
        quote_info = quote_data.info
        earnings_dates = ticker.calendar.get('Earnings Date')
        print("earnings_dates: ",earnings_dates)
        # Display relevant information in a tabular format
        quote_table = {
            "Category": ["Company Name", "Current Stock Price", "Change Perecentage", "Open Price", "High Price", "Low Price",
                        "Volume", "Market Capitalization", "52-Week Range", "Dividend Yield", "P/E", "EPS"],
            "Value": [quote_info.get('longName', 'N/A'),
                    f"${quote_info.get('currentPrice', 'N/A'):.2f}" if isinstance(quote_info.get('currentPrice'), float) else 'N/A',
                    f"{quote_info.get('regularMarketChangePercent', 'N/A'):.2%}" if quote_info.get('regularMarketChangePercent') is not None else 'N/A',
                    f"${quote_info.get('open', 'N/A'):.2f}" if isinstance(quote_info.get('open'), float) else 'N/A',
                    f"${quote_info.get('dayHigh', 'N/A'):.2f}" if isinstance(quote_info.get('dayHigh'), float) else 'N/A',
                    f"${quote_info.get('dayLow', 'N/A'):.2f}" if isinstance(quote_info.get('dayLow'), float) else 'N/A',
                    f"{quote_info.get('regularMarketVolume', 'N/A') / 1000000:.2f}M" if isinstance(quote_info.get('regularMarketVolume'), int) else 'N/A',
                    f"${quote_info.get('marketCap', 'N/A'):,}" if isinstance(quote_info.get('marketCap'), int) else 'N/A',
                    f"${quote_info.get('fiftyTwoWeekLow', 'N/A'):.2f} - ${quote_info.get('fiftyTwoWeekHigh', 'N/A'):.2f}" if isinstance(quote_info.get('fiftyTwoWeekLow'), float) and isinstance(quote_info.get('fiftyTwoWeekHigh'), float) else 'N/A',
                    f"{quote_info.get('dividendYield', 'N/A'):.2%}" if quote_info.get('dividendYield') is not None else 'N/A',
                    quote_info.get('trailingPE', 'N/A'),
                    quote_info.get('trailingEps', 'N/A')]
        }

        quote_table_df = pd.DataFrame(quote_table)
        quote_table_df.index = range(1, len(quote_table_df) + 1)
        st.table(quote_table_df)
    except Exception as e:
        st.warning(f"An error occured while fetching stock data: '{e}' . Please enter a valid ticker.")

def stockGraph(stock_data):
    try:
        # Visualize Stock Price
        st.markdown('<h2 class="subheader">Stock Prices Over Time</h2>', unsafe_allow_html=True)
        fig = px.line(stock_data, x=stock_data['Datetime'], y='Close')
        st.plotly_chart(fig)
    except Exception as e:
        st.warning(f"An error occured while creating stock graph: '{e}' . Please enter a valid ticker.")

def stockIndicatorForSwing(stock_data):
    try:
        # Create a horizontal slider to navigate through different indicators
        st.markdown('<h2 style="color: #0077b6; font-size: 28px;">Select a Technical Indicator</h2>', unsafe_allow_html=True)

        # Add an explanation about the indicators
        st.markdown("""
            <p style="font-size: 16px; color: #4a4a4a;">
                Choose a technical indicator from the dropdown to visualize it. Here's what each option means:
            </p>
            <ul style="font-size: 14px; color: #4a4a4a;">
                <li><b>SMA</b>: Simple Moving Average - Helps smooth out price data to identify trends.</li>
                <li><b>EMA</b>: Exponential Moving Average - Similar to SMA but gives more weight to recent prices.</li>
                <li><b>RSI</b>: Relative Strength Index - Indicates overbought or oversold conditions.</li>
                <li><b>MACD</b>: Moving Average Convergence Divergence - Highlights momentum and trend strength.</li>
                <li><b>ATR</b>: Average True Range - Average range of price movement for a stock over a specified period. </li>
            </ul>
        """, unsafe_allow_html=True)
        # Add the selectbox with icons
        selected_indicator = st.selectbox(
            "📈 Choose an indicator",
            ["📊 SMA - Simple Moving Average", 
            "📉 EMA - Exponential Moving Average", 
            "🔍 RSI - Relative Strength Index", 
            "📶 MACD - Moving Average Convergence Divergence", 
            "📏 ATR - Average True Range"],
            help="Select one of the technical indicators to display its visualization."
        )

        if selected_indicator == "📊 SMA - Simple Moving Average":
            fig_sma = go.Figure()

            # Add Candlestick Chart
            fig_sma.add_trace(go.Candlestick(
                x=stock_data['Datetime'],
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Candlestick'
            ))

            # Add Short-term SMA (20-period)
            fig_sma.add_trace(go.Scatter(
                x=stock_data['Datetime'],
                y=stock_data['SMA'],
                mode='lines',
                name='SMA',
                line=dict(color='green', width=2)
            ))

            # Update Layout
            fig_sma.update_layout(
                title='Simple Moving Averages (SMA)',
                xaxis_title='Datetime',
                yaxis_title='Price',
                template='plotly_dark',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                height=600,
                xaxis_rangeslider_visible=False
            )

            # Display in Streamlit
            st.plotly_chart(fig_sma)

        elif selected_indicator == "📉 EMA - Exponential Moving Average":
            # Plot EMA Chart
            fig_ema = go.Figure()

            # Add Candlestick Chart
            fig_ema.add_trace(go.Candlestick(
                x=stock_data['Datetime'],
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Candlestick'
            ))

            # Add 50-day EMA
            fig_ema.add_trace(go.Scatter(
                x=stock_data['Datetime'],
                y=stock_data['EMA12'],
                mode='lines',
                name='12-day EMA',
                line=dict(color='green', width=2)
            ))

            # Add 100-day EMA
            fig_ema.add_trace(go.Scatter(
                x=stock_data['Datetime'],
                y=stock_data['EMA26'],
                mode='lines',
                name='26-day EMA',
                line=dict(color='orange', width=2)
            ))

            # Update Layout
            fig_ema.update_layout(
                title='Exponential Moving Averages (EMA)',
                xaxis_title='Datetime',
                yaxis_title='Price',
                template='plotly_dark',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                height=600,
                xaxis_rangeslider_visible=False
            )

            # Display in Streamlit
            st.plotly_chart(fig_ema, use_container_width=True)

        elif selected_indicator == "🔍 RSI - Relative Strength Index":
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=stock_data['Datetime'], y=stock_data['RSI'], mode='lines', name=f'RSI'))
            fig_rsi.add_trace(go.Scatter(x=stock_data['Datetime'], y=[70] * len(stock_data), mode='lines', name='Overbought (70)', line=dict(color='red', dash='dash')))
            fig_rsi.add_trace(go.Scatter(x=stock_data['Datetime'], y=[30] * len(stock_data), mode='lines', name='Oversold (30)', line=dict(color='green', dash='dash')))
            fig_rsi.update_layout(title='Relative Strength Index (RSI)', xaxis_title='Date', yaxis_title='RSI Value')
            st.plotly_chart(fig_rsi)

        elif selected_indicator == "📶 MACD - Moving Average Convergence Divergence":
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=stock_data['Datetime'], y=stock_data['MACD'], mode='lines', name='MACD Line'))
            fig_macd.add_trace(go.Scatter(x=stock_data['Datetime'], y=stock_data['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='orange')))
            fig_macd.add_trace(go.Scatter(x=stock_data['Datetime'], y=[0] * len(stock_data), mode='lines', name='Zero Line', line=dict(color='black', dash='dash')))
            fig_macd.update_layout(title='Moving Average Convergence Divergence (MACD)', xaxis_title='Date', yaxis_title='MACD Value')
            st.plotly_chart(fig_macd)

        elif selected_indicator == "📏 ATR - Average True Range":
            # Plot ATR Chart
            fig_atr = go.Figure()

            # Add ATR line
            fig_atr.add_trace(go.Scatter(
                x=stock_data['Datetime'],
                y=stock_data['ATR'],
                mode='lines',
                name='ATR',
                line=dict(color='orange', width=2)
            ))

            # Add Candlestick Chart
            fig_atr.add_trace(go.Candlestick(
                x=stock_data['Datetime'],
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Candlestick'
            ))

            # Update layout for better visuals
            fig_atr.update_layout(
                title='Average True Range (ATR)',
                xaxis_title='Datetime',
                yaxis_title='Value',
                template='plotly_dark',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                height=600,
                xaxis_rangeslider_visible=False
            )

            # Display in Streamlit
            st.plotly_chart(fig_atr)
    
    except Exception as e:
        st.warning(f"An error occured while working on indicator '{e}' . Please enter a valid ticker.")

def stockIndicatorForIntraday(stock_data):
    try:
        # Create a horizontal slider to navigate through different indicators
        st.markdown('<h2 style="color: #0077b6; font-size: 28px;">Select a Technical Indicator</h2>', unsafe_allow_html=True)

        # Add an explanation about the indicators
        st.markdown("""
            <p style="font-size: 16px; color: #4a4a4a;">
                Choose a technical indicator from the dropdown to visualize it. Here's what each option means:
            </p>
            <ul style="font-size: 14px; color: #4a4a4a;">
                <li><b>SMA</b>: Simple Moving Average - Helps smooth out price data to identify trends.</li>
                <li><b>EMA</b>: Exponential Moving Average - Similar to SMA but gives more weight to recent prices.</li>
                <li><b>RSI</b>: Relative Strength Index - Indicates overbought or oversold conditions.</li>
                <li><b>MACD</b>: Moving Average Convergence Divergence - Highlights momentum and trend strength.</li>
                <li><b>VWAP</b>: Volume Weighted Average Price - True average price based on its liquidity.</li>
                <li><b>ATR</b>: Average True Range - Average range of price movement for a stock over a specified period. </li>
            </ul>
        """, unsafe_allow_html=True)
        # Add the selectbox with icons
        selected_indicator = st.selectbox(
            "📈 Choose an indicator",
            ["📊 SMA - Simple Moving Average", 
            "📉 EMA - Exponential Moving Average", 
            "🔍 RSI - Relative Strength Index", 
            "📶 MACD - Moving Average Convergence Divergence", 
            "📐 VWAP - Volume Weighted Average Price", 
            "📏 ATR - Average True Range"],
            help="Select one of the technical indicators to display its visualization."
        )

        if selected_indicator == "📊 SMA - Simple Moving Average":
            fig_sma = go.Figure()

            # Add Candlestick Chart
            fig_sma.add_trace(go.Candlestick(
                x=stock_data['Datetime'],
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Candlestick'
            ))

            # Add Short-term SMA (20-period)
            fig_sma.add_trace(go.Scatter(
                x=stock_data['Datetime'],
                y=sma_20,
                mode='lines',
                name='20-Period SMA',
                line=dict(color='green', width=2)
            ))

            # Add Long-term SMA (5-period)
            fig_sma.add_trace(go.Scatter(
                x=stock_data['Datetime'],
                y=sma_5,
                mode='lines',
                name='5-Period SMA',
                line=dict(color='blue', width=2)
            ))

            # Update Layout
            fig_sma.update_layout(
                title='Simple Moving Averages (SMA)',
                xaxis_title='Datetime',
                yaxis_title='Price',
                template='plotly_dark',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                height=600,
                xaxis_rangeslider_visible=False
            )

            # Display in Streamlit
            st.plotly_chart(fig_sma)

        elif selected_indicator == "📉 EMA - Exponential Moving Average":
            # Plot EMA Chart
            fig_ema = go.Figure()

            # Add Candlestick Chart
            fig_ema.add_trace(go.Candlestick(
                x=stock_data['Datetime'],
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Candlestick'
            ))

            # Add 20-period EMA
            fig_ema.add_trace(go.Scatter(
                x=stock_data['Datetime'],
                y=ema_20,
                mode='lines',
                name='20-period EMA',
                line=dict(color='green', width=2)
            ))

            # Add 5-period EMA
            fig_ema.add_trace(go.Scatter(
                x=stock_data['Datetime'],
                y=ema_5,
                mode='lines',
                name='5-period EMA',
                line=dict(color='orange', width=2)
            ))

            # Update Layout
            fig_ema.update_layout(
                title='Exponential Moving Averages (EMA)',
                xaxis_title='Datetime',
                yaxis_title='Price',
                template='plotly_dark',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                height=600,
                xaxis_rangeslider_visible=False
            )

            # Display in Streamlit
            st.plotly_chart(fig_ema, use_container_width=True)

        elif selected_indicator == "🔍 RSI - Relative Strength Index":
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=stock_data['Datetime'], y=stock_data['RSI'], mode='lines', name=f'RSI'))
            fig_rsi.add_trace(go.Scatter(x=stock_data['Datetime'], y=[70] * len(stock_data), mode='lines', name='Overbought (70)', line=dict(color='red', dash='dash')))
            fig_rsi.add_trace(go.Scatter(x=stock_data['Datetime'], y=[30] * len(stock_data), mode='lines', name='Oversold (30)', line=dict(color='green', dash='dash')))
            fig_rsi.update_layout(title='Relative Strength Index (RSI)', xaxis_title='Date', yaxis_title='RSI Value')
            st.plotly_chart(fig_rsi)

        elif selected_indicator == "📶 MACD - Moving Average Convergence Divergence":
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=stock_data['Datetime'], y=stock_data['MACD'], mode='lines', name='MACD Line'))
            fig_macd.add_trace(go.Scatter(x=stock_data['Datetime'], y=macd_signal, mode='lines', name='Signal Line', line=dict(color='orange')))
            fig_macd.add_trace(go.Scatter(x=stock_data['Datetime'], y=[0] * len(stock_data), mode='lines', name='Zero Line', line=dict(color='black', dash='dash')))
            fig_macd.update_layout(title='Moving Average Convergence Divergence (MACD)', xaxis_title='Date', yaxis_title='MACD Value')
            st.plotly_chart(fig_macd)

        elif selected_indicator == "📐 VWAP - Volume Weighted Average Price":
            fig_vwap = go.Figure()
            fig_vwap.add_trace(go.Candlestick(x=stock_data['Datetime'],
                open=stock_data['Open'],  # Assuming 'Open' is same as 'High' for simplicity
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Candlestick'
            ))
            fig_vwap.add_trace(go.Scatter(
                x=stock_data['Datetime'],
                y=stock_data['VWAP'],
                mode='lines',
                line=dict(color='orange', width=2),
                name='VWAP'
            ))
            fig_vwap.update_layout(
                title='VWAP and Candlestick Chart',
                xaxis_title='Datetime',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                height=600,
            )
            # Show the plot
            st.plotly_chart(fig_vwap)

        elif selected_indicator == "📏 ATR - Average True Range":
            # Plot ATR Chart
            fig_atr = go.Figure()

            # Add ATR line
            fig_atr.add_trace(go.Scatter(
                x=stock_data['Datetime'],
                y=stock_data['ATR'],
                mode='lines',
                name='ATR',
                line=dict(color='orange', width=2)
            ))

            # Add Candlestick Chart
            fig_atr.add_trace(go.Candlestick(
                x=stock_data['Datetime'],
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Candlestick'
            ))

            # Update layout for better visuals
            fig_atr.update_layout(
                title='Average True Range (ATR)',
                xaxis_title='Datetime',
                yaxis_title='Value',
                template='plotly_dark',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                height=600,
                xaxis_rangeslider_visible=False
            )

            # Display in Streamlit
            st.plotly_chart(fig_atr)
    
    except Exception as e:
        st.warning(f"An error occured while working on indicator '{e}' . Please enter a valid ticker.")


def build_model(hp):
    model = Sequential()
    print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32), return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=256, step=32)))  # Unique name for each parameter
    model.add(Dropout(hp.Float('dropout_2', 0.1, 0.5, step=0.1)))
    model.add(Dense(1, activation='linear'))

    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        )
    )
    return model

def build_model_intraday(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32), return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=256, step=32)))  # Unique name for each parameter
    model.add(Dropout(hp.Float('dropout_2', 0.1, 0.5, step=0.1)))
    model.add(Dense(1, activation='linear'))

    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        )
    )
    return model


# Validate the stock symbol
if stock_symbol:
    ticker = yf.Ticker(stock_symbol)
    try:
        ticker_info = ticker.info
        if stock_symbol != st.session_state.previous_symbol:
            TODAY_RUN = datetime.datetime.today().strftime("%Y%m%d")
            TOKEN = stock_symbol + '_' + TODAY_RUN + '_' + secrets.token_hex(16)
            PROJECT_FOLDER = os.path.join(os.getcwd(), TOKEN)
            sma_100 = 0
            sma_50 = 0
            ema_100 = 0
            ema_50 = 0
            if not os.path.exists(PROJECT_FOLDER):
                os.makedirs(PROJECT_FOLDER)
            st.session_state.previous_symbol = stock_symbol
            st.session_state.project_folder = os.path.join(os.getcwd(), TOKEN)
    except:
        st.warning(f"'{stock_symbol}' is not a valid stock ticker symbol. Please enter a valid ticker.")


if stock_symbol != "":
    if menu_option == "Day Trading Forcast":
        try:
            # Fetch data for the last 8 days with a 1-minute interval
            end_date = datetime.datetime.now()
            start_date = end_date - timedelta(days=59)
            last_15_day = end_date - timedelta(days=15)
            validation_date = pd.to_datetime(last_15_day)  

            stock_data = yf.download(tickers=stock_symbol, period="1mo", interval="5m")
            if stock_data.empty:
                st.error("No data found for the given interval. Please try a different symbol.")
            else:
                stock_data.index = pd.to_datetime(stock_data.index)
                stock_data.index = stock_data.index.tz_convert("Asia/Kolkata")
                stock_data.reset_index(inplace=True)
                stock_data = stock_data.drop(index=stock_data.index[0])
                stock_data.to_csv(os.path.join(PROJECT_FOLDER, 'downloaded_data_'+ stock_symbol+'.csv'), index=False)
                stock_data=pd.read_csv(os.path.join(PROJECT_FOLDER, 'downloaded_data_'+ stock_symbol +'.csv'))
                stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
                st.success(f"Download last 30 days of precise data (5-minute interval)")

                stockInfo(stock_symbol)
                stockGraph(stock_data)

                stock_data['High']= round(pd.to_numeric(stock_data['High'], errors='coerce'),2)
                stock_data['Low'] = round(pd.to_numeric(stock_data['Low'], errors='coerce'),2)
                stock_data['Close']= round(pd.to_numeric(stock_data['Close'], errors='coerce'),2)
                stock_data['Open']= round(pd.to_numeric(stock_data['Open'], errors='coerce'),2)
                stock_data["Dynamic_Support"] = stock_data["Low"].rolling(window=14).min()
                stock_data["Dynamic_Resistance"] = stock_data["High"].rolling(window=14).max()

                # Pivot-based levels
                stock_data["Pivot"] = (stock_data["High"] + stock_data["Low"] + stock_data["Close"]) / 3
                stock_data["R1"] = (2 * stock_data["Pivot"]) - stock_data["Low"]
                stock_data["S1"] = (2 * stock_data["Pivot"]) - stock_data["High"]
                stock_data["R2"] = stock_data["Pivot"] + (stock_data["High"] - stock_data["Low"])
                stock_data["S2"] = stock_data["Pivot"] - (stock_data["High"] - stock_data["Low"])

                stock_data['O-C']= round(stock_data['Close']-stock_data['Open'],2)


                stock_data['Delta'] = stock_data['Close'].diff() 
                
                # Plot Relative Strength Index (RSI)
                stock_data['RSI'] = round(RSIIndicator(close=stock_data['Close'], window=14).rsi(),2)

                # Plot 5 and 20 minute Simple Moving Averages
                sma_5 = stock_data['Close'].rolling(window=5).mean()
                sma_20 = stock_data['Close'].rolling(window=20).mean()
                stock_data['SMA_5'] = sma_5
                stock_data['SMA_20'] = sma_20

                # Plot 5 and 20-minute Exponential Moving Averages
                ema_5 = stock_data['Close'].ewm(span=5, adjust=False).mean()
                ema_20 = stock_data['Close'].ewm(span=20, adjust=False).mean()
                stock_data['EMA_5'] = ema_5
                stock_data['EMA_20'] = ema_20

                # Step 3: Generate Buy/Sell Signals
                stock_data["EMA_Signal"] = 0
                stock_data.loc[stock_data["EMA_5"] > stock_data["EMA_20"], "EMA_Signal"] = 1  # Buy Signal
                stock_data.loc[stock_data["EMA_5"] <= stock_data["EMA_20"], "EMA_Signal"] = -1  # Sell Signal

                stock_data['factorchange'] = np.where(
                    (stock_data['Open'] > 100) & (stock_data['Open'] < 500), 2,
                    np.where(
                        (stock_data['Open'] > 500) & (stock_data['Open'] < 1000), 3,
                        np.where(
                            (stock_data['Open'] > 1000) & (stock_data['Open'] < 4000), 5,
                            10  # Default case if no conditions are met
                        )
                    )
                )

                # Assign labels based on conditions
                stock_data['OC'] = np.where(
                    stock_data['O-C'] < stock_data['factorchange'], "Sell",  # If O-C < -5
                    np.where(
                        stock_data['O-C'] > stock_data['factorchange'], "Strong Buy",  # If O-C > 5
                        np.where(
                            (stock_data['O-C'] < stock_data['factorchange']) & (stock_data['O-C'] > 0), "Buy",  # If O-C < 5
                            "HOLD"  # Default condition if O-C > -5
                        )
                    )
                )

                # Plot Moving Average Convergence Divergence (MACD)
                macd = MACD(close=stock_data['Close'], window_slow=26, window_fast=12, window_sign=9)
                stock_data['MACD'] = macd.macd()
                stock_data['MACD_signal'] = macd.macd_signal()
                macd_signal = macd.macd_signal()

                #create instance of SES
                # SES reduces this noise by giving more weight to recent data and
                # less weight to older data, creating a smoothed version of the stock's price movement.
                # ses = SimpleExpSmoothing(stock_data['Close'])
                # #fit SES to data
                # alpha = 0.7
                # res = ses.fit(smoothing_level=alpha, optimized=False)
                # stock_data['SES'] = res.fittedvalues

                # Add Lag Features
                for lag in [1, 3, 5, 10]:
                    stock_data[f'Close_lag_{lag}'] = stock_data['Close'].shift(lag)

                # Add Bollinger Bands
                bollinger = BollingerBands(close=stock_data['Close'], window=20)
                stock_data['BB_upper'] = round( bollinger.bollinger_hband(),2)
                stock_data['BB_middle'] = round( bollinger.bollinger_mavg(),2)
                stock_data['BB_lower'] =round( bollinger.bollinger_lband(),2)

                stock_data['Volume'] = pd.to_numeric(stock_data['Volume'], errors='coerce')

                
                # VWAP (Volume-Weighted Average Price)
                stock_data['Cum_Price_Volume'] = (stock_data['Close'] * stock_data['Volume']).cumsum()
                stock_data['Cum_Volume'] = stock_data['Volume'].cumsum()
                stock_data['VWAP'] = stock_data['Cum_Price_Volume'] / stock_data['Cum_Volume']
                stock_data.drop(columns=['Cum_Price_Volume', 'Cum_Volume'], inplace=True)

                stock_data['Avg_Volume'] = round(stock_data['Volume'].rolling(window=20).mean(),0)
                stock_data['RVOL'] = round(stock_data['Volume'] / stock_data['Avg_Volume'],0)

                def identify_signal(row):
                    # Strong Buy
                    if row['RVOL'] >= 2 and row['Close'] < row['BB_lower'] and row['RSI'] < 30:
                        return "Strong Buy"
                    # Strong Sell
                    elif row['RVOL'] >= 2 and row['Close'] > row['BB_upper'] and row['RSI'] > 70:
                        return "Strong Sell"
                    else :
                        return "Netutral"
                    return None

                stock_data['Vol_Sell_Signal'] = stock_data.apply(identify_signal, axis=1)

                stock_data['CheckVolume'] = np.where(
                    (stock_data['Volume'] > stock_data['Avg_Volume']) & (stock_data['O-C'] < 0),
                    "High Vol Sold",
                    np.where(
                        (stock_data['Volume'] > stock_data['Avg_Volume']) & (stock_data['O-C'] > 0),
                        "High Vol Bought",
                        "Neutral"
                    )
                )

                # Combined Signal: 1 for Buy, -1 for Sell, 0 for Hold
                stock_data['B/S_Signal'] = np.where(
                    (stock_data['RSI'] < 30) & (stock_data['MACD'] > stock_data['MACD_signal']) & (stock_data['CheckVolume'] == 'High Vol Bought'),
                    "Strong BUY",  # Buy
                    np.where(
                        (stock_data['RSI'] > 70) & (stock_data['MACD'] < stock_data['MACD_signal']) & (stock_data['CheckVolume'] == 'High Vol Sold'),
                        "SELL",  # Sell Signal
                        "HOLD"  # Hold
                    )
                )

                def identify_reversal(row):
                    # Bearish Reversal
                    if row['Close'] < row['BB_lower'] and row['RSI'] < 30 and row['Close'] < row['VWAP']:
                        return "Buy"
                    # Bearish Reversal
                    elif row['Close'] > row['BB_upper'] and row['RSI'] > 70 and row['Close'] > row['VWAP']:
                        return "Sell"
                    else :
                        return "Neutral"
                    return None

                stock_data['Reversal'] = stock_data.apply(identify_reversal, axis=1)

                conditions = [
                    (stock_data['Close'] > stock_data['VWAP']) & (stock_data['Volume'] > stock_data['Volume'].rolling(window=5).mean()),  # Buy Signal
                    (stock_data['Close'] < stock_data['VWAP']) & (stock_data['Volume'] > stock_data['Volume'].rolling(window=5).mean())   # Sell Signal
                ]

                # Values for corresponding conditions (1 for Buy, -1 for Sell, 0 for no signal)
                values = ["Buy", "Sell"]

                # Apply the conditions and create the signal column
                stock_data['VwapSign'] = np.select(conditions, values, default="NoSignal")

                stock_data['LargeVol'] = stock_data['Volume'] > (2 * stock_data['Avg_Volume'])
                stock_data['3xLargeVol']= stock_data['Volume'] > (3 * stock_data['Avg_Volume'])

                stock_data['DynVol'] = stock_data['Volume'] > (1.5 * stock_data['Avg_Volume'])


                                
                stock_data['3xLargeVol']=stock_data['3xLargeVol'].apply(lambda x:"T" if x else "F" )
                stock_data['LargeVol']=  stock_data['LargeVol'].apply(lambda x:"T" if x else "F" )

                stock_data['DynVol']= stock_data['DynVol'].apply(lambda x: "T" if x else "F") + stock_data['LargeVol']+stock_data['3xLargeVol']

                # Buy and Sell Ranges
                stock_data["Buy_Range"] = stock_data[["Dynamic_Support", "S1", "S2"]].min(axis=1)  # Lowest of all supports
                stock_data["Sell_Range"] = stock_data[["Dynamic_Resistance", "R1", "R2"]].max(axis=1)  # Highest of all resistances
                
                stock_data['Date'] = pd.to_datetime(stock_data['Datetime'])
                stock_data['Date'] = pd.to_datetime(stock_data['Datetime']).dt.date

                # ATR (Average True Range)
                atr = AverageTrueRange(high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close'], window=7)
                stock_data['ATR'] = atr.average_true_range()
                support_level = stock_data['Low'].iloc[:15].min()
                resistance_level = stock_data['High'].iloc[:15].max()

                stock_data['Previous_Close'] = stock_data['Close'].shift(1)  # Shift close price by 1 to get previous close
                stock_data['TR'] = stock_data.apply(
                    lambda row: max(
                        row['High'] - row['Low'],  # High - Low
                        abs(row['High'] - row['Previous_Close']),  # High - Previous Close
                        abs(row['Low'] - row['Previous_Close'])  # Low - Previous Close
                    ),
                    axis=1
                )

                def calculate_avg_price(prices, volumes):
                    prices = np.array(prices)  # Convert to NumPy array
                    volumes = np.array(volumes)  # Convert to NumPy array
                    total_value = np.sum(prices * volumes)
                    total_qty = np.sum(volumes)
                    return total_value / total_qty if total_qty else 0

                # Assuming 'data' is a DataFrame with columns 'Close', 'Open', and 'Volume'
                volumes = stock_data['Volume'].values
                
                                #Calculate Buyers' and Sellers' Average Prices per row
                stock_data['BAP'] = stock_data.apply(lambda row: calculate_avg_price([row['Close']], [row['Volume']]), axis=1)
                stock_data['SAP'] = stock_data.apply(lambda row: calculate_avg_price([row['Open']], [row['Volume']]), axis=1)

                stock_data['Volume_MA'] = stock_data['Volume'].rolling(window=10).mean()
                stock_data['Volume_Trend'] = np.where(stock_data['Volume'] > stock_data['Volume_MA'], 1, -1)  # 1: Increasing, -1: Decreasing
                                
                stock_data['Strong_Sign'] = np.where(
                    (stock_data['BAP'] > stock_data['SAP']) &
                    (stock_data['Close'] > stock_data['VWAP']) &
                    (stock_data['Volume_Trend'] == 1) &
                    (stock_data['RSI'] < 40) &
                    (stock_data['MACD'] > stock_data['MACD_signal']), "StrongBUY",  # Strong Buy

                    np.where(
                        (stock_data['SAP'] > stock_data['BAP']) &
                        (stock_data['Close'] < stock_data['VWAP']) &
                        (stock_data['Volume_Trend'] == 1) &
                        (stock_data['RSI'] > 60) &
                        (stock_data['MACD'] < stock_data['MACD_signal']), "StrongSELL",  # Strong Sell

                        "Neutral"  # No Signal
                    )
                )

                # # Define Breakout & Breakdown Conditions
                stock_data['B/S-Press'] = np.where(
                    (stock_data['Close'] > stock_data['VWAP']) & (stock_data['BAP'] > stock_data['SAP']) & (stock_data['Volume_Trend'] == 1), "BuyPress",  # Buy Signal
                    np.where(
                        (stock_data['Close'] < stock_data['VWAP']) & (stock_data['SAP'] > stock_data['BAP']) & (stock_data['Volume_Trend'] == -1), "SellPress",  # Sell Signal
                        "Neutral"  # No Signal
                    )
                )
                # Step 3: Define Multipliers for Entry, Stop-Loss, and Target
                atr_multiplier = 1.5
                target_multiplier = 2.0

                # Step 4: Calculate Buy Range
                stock_data['Buy_Entry'] = resistance_level + (stock_data['ATR'] * atr_multiplier)
                stock_data['Buy_Stop_Loss'] = stock_data['Buy_Entry'] - (stock_data['ATR'] * atr_multiplier)
                stock_data['Buy_Target'] = stock_data['Buy_Entry'] + (stock_data['ATR'] * target_multiplier)

                # Step 5: Calculate Sell Range
                stock_data['Sell_Entry'] = support_level + (stock_data['ATR'] * atr_multiplier)
                stock_data['Sell_Stop_Loss'] = stock_data['Sell_Entry'] + (stock_data['ATR'] * atr_multiplier)
                stock_data['Sell_Target'] = stock_data['Sell_Entry'] - (stock_data['ATR'] * target_multiplier)

                # Step 6: Filter Buy and Sell Signals
                buy_signals = stock_data[stock_data['Close'] > stock_data['Buy_Entry']]
                sell_signals = stock_data[stock_data['Close'] < stock_data['Sell_Entry']]

                daily_rsi_stats = stock_data.groupby('Date')['RSI'].agg(['max', 'min']).reset_index()
                daily_rsi_stats.rename(columns={'max': 'RSI_Max', 'min': 'RSI_Min'}, inplace=True)

                daily_high_low = stock_data.groupby("Date").agg(Daily_High=("High", "max"), Daily_Low=("Low", "min")).reset_index()
                stock_data = pd.merge(stock_data, daily_rsi_stats, on='Date', how='left')

                stock_data['Log_Close'] = np.log(stock_data['Close'])  # Stabilize fluctuations
                stock_data['Diff_Close'] = stock_data['Close'].diff()  # Remove trends
                
                prediction_size_menu = st.sidebar.radio(
                    "Select Prediction time limit",
                    ("Next 15-minute prediction", "Next 1-hour prediction", "Today prediction", "Next day prediction", "Next 2-day prediction"),
                    help="Select one of the prediction timeline to forecast stock prices."
                )

                # Display the selected model in the sidebar
                st.sidebar.markdown(f"""
                    <div style="margin-top: 20px; text-align: center;">
                        <h3 style="color: #0077b6;">You selected:</h3>
                        <h3 style="color: #023e8a;">{menu_option} for {prediction_size_menu}</h3>
                    </div>
                """, unsafe_allow_html=True)

                # You are predicting for the next 3 days. 
                # Each day has 6 hours of trading time (from 9:15 AM to 3:30 PM),
                # and each hour has 12 intervals (5 minutes each). For 15 minute of prediction, PREDICTED_TIME is 3
                # So, 6 * 12 = 72 intervals per day, and for 3 days, the total periods are 216
                
                if prediction_size_menu == "Next 15-minute prediction":
                    PREDICT_START_DATE = datetime.datetime.now(tz)
                    PREDICTED_TIME = 3
                    stock_data['SMA'] = sma_5
                    WINDOW = 5
                elif prediction_size_menu == "Next 1-hour prediction":
                    PREDICT_START_DATE = datetime.datetime.now(tz)
                    PREDICTED_TIME = 12
                    stock_data['SMA'] = sma_5
                    WINDOW = 5
                elif prediction_size_menu == "Today prediction":
                    PREDICT_START_DATE = datetime.datetime.today()
                    PREDICTED_TIME = 72
                    stock_data['SMA'] = sma_5
                    WINDOW = 20
                elif prediction_size_menu == "Next day prediction":
                    PREDICT_START_DATE = datetime.datetime.today() + timedelta(days=1)
                    PREDICTED_TIME = 144
                    stock_data['SMA'] = sma_20
                    WINDOW = 20
                elif prediction_size_menu == "Next 2-day prediction":
                    PREDICT_START_DATE = datetime.datetime.today() + timedelta(days=2)
                    PREDICTED_TIME = 216
                    stock_data['SMA'] = sma_20
                    WINDOW = 20
                else:
                    PREDICT_START_DATE = datetime.datetime.now(tz)
                    PREDICTED_TIME = 3
                    stock_data['SMA'] = sma_5
                    WINDOW = 20

                # Drop rows with NaN values
                stock_data=stock_data[['Datetime','Date','Open','High','Low','Close','TR','B/S-Press','CheckVolume','DynVol','B/S_Signal','Reversal','Vol_Sell_Signal','Volume','Avg_Volume','VWAP','RSI','MACD', 'MACD_signal', 'SMA','ATR','BB_lower', 'BB_upper', 'Log_Close', 'Diff_Close']]
                stock_data.dropna(inplace=True)
                
                stock_data.to_csv(os.path.join(PROJECT_FOLDER, 'data_'+stock_symbol+'.csv'), index=False)

                st.markdown('<h2 class="subheader">Stock Prices with Indicator Data</h2>', unsafe_allow_html=True)


                # Display paginated data
                stock_data["DynVol"] = stock_data["DynVol"].astype(str)
                st.dataframe(stock_data.tail(50).sort_values(by=stock_data.columns[0], ascending=False))

                # Create figure
                fig = go.Figure()

                # Add Candlestick chart
                fig.add_trace(go.Candlestick(
                    x=stock_data['Datetime'],
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    name='Candlestick'
                ))

                # Add SMA and EMA
                fig.add_trace(go.Scatter(x=stock_data['Datetime'], y=stock_data['SMA'], mode='lines', name='SMA', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=stock_data['Datetime'], y=stock_data['VWAP'], mode='lines', name='VWAP', line=dict(color='purple')))

                # Add Bollinger Bands
                fig.add_trace(go.Scatter(x=stock_data['Datetime'], y=stock_data['BB_upper'], mode='lines', name='BB Upper', line=dict(color='red', dash='dot')))
                fig.add_trace(go.Scatter(x=stock_data['Datetime'], y=stock_data['BB_lower'], mode='lines', name='BB Lower', line=dict(color='red', dash='dot')))

                # Add Buy/Sell Signals
                buy_signals = stock_data[stock_data['B/S_Signal'] == "Strong BUY"]
                sell_signals = stock_data[stock_data['B/S_Signal'] == "SELL"]

                fig.add_trace(go.Scatter(x=buy_signals['Datetime'], y=buy_signals['Close'], mode='markers+text', name='Buy Signal', marker=dict(color='green', size=8, symbol='triangle-up'), text="BUY", textposition="top center", textfont=dict(size=12, color="green")))
                fig.add_trace(go.Scatter(x=sell_signals['Datetime'], y=sell_signals['Close'], mode='markers+text', name='Sell Signal', marker=dict(color='red', size=8, symbol='triangle-down'), text="SELL", textposition="top center", textfont=dict(size=12, color="red")))
                fig.add_trace(go.Scatter(x=buy_signals['Datetime'], y=buy_signals['Close'], mode='markers', name='Glow Effect', marker=dict(color='rgba(0,255,0,0.3)', size=18, symbol='circle-open'), showlegend=False ))
                fig.add_trace(go.Scatter(x=sell_signals['Datetime'], y=sell_signals['Close'], mode='markers', name='Glow Effect', marker=dict(color='rgba(0,255,0,0.3)', size=18, symbol='circle-open'), showlegend=False ))
                
                # Customize layout
                fig.update_layout(
                    title=f"{stock_symbol} Stock Data",
                    xaxis_title="Time",
                    yaxis_title="Stock Price",
                    xaxis_rangeslider_visible=False,template="plotly_dark")

                # Show plot
                st.plotly_chart(fig, use_container_width=True)

                # stockIndicatorForIntraday(stock_data)

                menu_option_trained = st.radio(
                    "Choose an option:", 
                    ["None Selected","Train New Model", "Use Pre-Trained Model"], 
                    index=0,
                    )
                stock_data['Datetime'] = pd.to_datetime(stock_data['Datetime'], errors='coerce')
                training_data = stock_data[stock_data['Datetime'] < pd.Timestamp(validation_date).tz_localize("Asia/Kolkata")].copy()
                test_data = stock_data[stock_data['Datetime'] >= pd.Timestamp(validation_date).tz_localize('Asia/Kolkata')].copy()
                training_data = training_data.set_index('Datetime')
                # Set the data frame index using column Date
                test_data = test_data.set_index('Datetime')

                train_scaled = scaler.fit_transform(training_data[['Open','High','Low','Close','Volume','VWAP','RSI','MACD','SMA','ATR','BB_lower', 'BB_upper', 'TR', 'Avg_Volume', 'Log_Close', 'Diff_Close', 'MACD_signal']])
                
                # Training Data Transformation
                x_train = []
                y_train = []
                for i in range(TIME_STEPS, len(train_scaled)):
                    x_train.append(train_scaled[i - TIME_STEPS:i])
                    y_train.append(train_scaled[i, 3]) 

                x_train, y_train = np.array(x_train), np.array(y_train)
                total_data = pd.concat((training_data, test_data), axis=0)
                inputs = total_data[len(total_data) - len(test_data) - TIME_STEPS:]
                test_scaled = scaler.transform(inputs[['Open','High','Low','Close','Volume','VWAP','RSI','MACD','SMA','ATR','BB_lower', 'BB_upper', 'TR', 'Avg_Volume', 'Log_Close', 'Diff_Close', 'MACD_signal']])
                
                # Testing Data Transformation
                x_test = []
                y_test = []
                for i in range(TIME_STEPS, len(test_scaled)):
                    x_test.append(test_scaled[i - TIME_STEPS:i])
                    y_test.append(test_scaled[i, 3])

                x_test, y_test = np.array(x_test), np.array(y_test)

                print(f"Feature count - Training: {train_scaled.shape[1]}, Testing: {test_scaled.shape[1]}")

                if menu_option_trained == "None Selected":
                    print("None Selected")
                elif menu_option_trained == "Train New Model":

                    tuner = kt.RandomSearch(
                        build_model_intraday, 
                        objective='val_loss', 
                        max_trials=1, 
                        executions_per_trial=1,
                        directory='hyperparam_tuning',  # Save tuning results
                        project_name='lstm_stock_forecast_intraday'
                    )

                    # Perform hyperparameter search
                    tuner.search(
                        x_train, 
                        y_train, 
                        epochs=EPOCHS, 
                        validation_data=(x_test, y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
                    )

                    # Get the best model
                    model = tuner.get_best_models(num_models=1)[0]
                    print("saving weights")
                    model.save(os.path.join(PROJECT_FOLDER, 'close_model_weights.h5'))
                    test_predictions_baseline = model.predict(x_test)

                    test_predictions_baseline_padded = np.zeros((test_predictions_baseline.shape[0], x_test.shape[2]))
                    test_predictions_baseline_padded[:, 3] = test_predictions_baseline.flatten()

                    # Perform inverse transform
                    x_test_predictions_baseline = scaler.inverse_transform(test_predictions_baseline_padded)[:, 3]  # Extract only the first column

                    test_predictions_baseline = pd.DataFrame({
                            f'{stock_symbol}_actual': test_data['Close'],  # Actual Close price
                            f'{stock_symbol}_lstm_predicted': x_test_predictions_baseline  # Predicted Close price
                        })
                    test_predictions_baseline[f'{stock_symbol}_residual'] = test_predictions_baseline[f'{stock_symbol}_actual'] - test_predictions_baseline[f'{stock_symbol}_lstm_predicted']

                    X_train, X_test, y_train, y_test = train_test_split(test_predictions_baseline[[f'{stock_symbol}_lstm_predicted']], test_predictions_baseline[f'{stock_symbol}_residual'], test_size=0.2, shuffle=False)
                    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
                    xgb_model.fit(X_train, y_train)

                    test_predictions_baseline[f'{stock_symbol}_xgb_residual_correction'] = xgb_model.predict(test_predictions_baseline[[f'{stock_symbol}_lstm_predicted']])
                    test_predictions_baseline[f'{stock_symbol}_predicted'] = test_predictions_baseline[f'{stock_symbol}_lstm_predicted'] + test_predictions_baseline[f'{stock_symbol}_xgb_residual_correction']
                    test_predictions_baseline = test_predictions_baseline.drop(columns=[f'{stock_symbol}_xgb_residual_correction', f'{stock_symbol}_lstm_predicted', f'{stock_symbol}_residual'])

                    test_predictions_baseline.index = test_data.index
                    test_predictions_baseline = test_predictions_baseline.reset_index()  # Moves index to a column

                    test_predictions_baseline.to_csv(os.path.join(PROJECT_FOLDER, 'predictions.csv'))

                    st.markdown('<h2 class="subheader">Predicted and Actual Data</h2>', unsafe_allow_html=True)
                    get_accuracy(test_predictions_baseline[f'{stock_symbol}_actual'].values, test_predictions_baseline[f'{stock_symbol}_predicted'].values)
                    

                    st.write(test_predictions_baseline.tail(50).sort_values(by=test_predictions_baseline.columns[0], ascending=False)) 
                    # print(test_predictions_baseline)
                    plotActualPredictedValue(test_predictions_baseline.tail(50))

                    model = tf.keras.models.load_model(os.path.join(PROJECT_FOLDER, "close_model_weights.h5"), custom_objects={"mse": mse})

                    # Adjust the number of features dynamically
                    num_features = x_test.shape[2]  # Number of features in the dataset

                    # Perform inverse scaling on predicted values
                    predictions = model.predict(x_test)

                    if prediction_size_menu == "Next 15-minute prediction" or prediction_size_menu == "Next 1-hour prediction":
                        predicted_dates = generate_market_time_range(PREDICT_START_DATE, PREDICTED_TIME, 5)
                    else:
                        predicted_dates = generate_market_time_range_5_minute(PREDICT_START_DATE, PREDICTED_TIME)

                    predictions = predictions[-PREDICTED_TIME:]
                    y_test = y_test[-PREDICTED_TIME:]

                    # Inverse scaling for 'Close' price
                    predicted_close = scaler.inverse_transform(np.column_stack((
                        np.zeros((len(predictions), 3)),  # Placeholders for Open, High, Low
                        predictions,  # Predicted Close (assuming it is the first column of predictions)
                        np.zeros((len(predictions), 13))  # Placeholder for remaining features
                    )))[:, 3]  # Here we select index 3 for 'Close' if 'Close' is the fourth column
                    
                    # Add predictions and actual values to test_data
                    close_data = pd.DataFrame({
                        'Datetime': predicted_dates,
                        'Predicted Close Price': predicted_close,
                    })
                    close_data.to_csv(os.path.join(PROJECT_FOLDER, "predicted_data.csv"), index=False)

                    close_data['SMA'] = close_data['Predicted Close Price'].rolling(WINDOW).mean()
                    close_data['Upper_Band'] = close_data['SMA'] + (close_data['Predicted Close Price'].rolling(WINDOW).std() * 2)
                    close_data['Lower_Band'] = close_data['SMA'] - (close_data['Predicted Close Price'].rolling(WINDOW).std() * 2)

                    close_data['Trend'] = np.where(close_data['Predicted Close Price'] > close_data['Upper_Band'], 'Uptrend',
                                np.where(close_data['Predicted Close Price'] < close_data['Lower_Band'], 'Downtrend', 'Neutral'))

                    st.markdown('<h2 class="subheader">Predicted Close Price</h2>', unsafe_allow_html=True)
                    st.write(close_data)
                    # Create the plot using Plotly
                    close_fig = go.Figure()

                    # Add the predicted close price line
                    close_fig.add_trace(go.Scatter(
                        x=predicted_dates,
                        y=predicted_close,
                        mode='lines',
                        name='Predicted Close Price',
                        line=dict(color='red', width=2)
                    ))

                    # Add Bollinger Bands
                    close_fig.add_trace(go.Scatter(
                        x=close_data['Datetime'],
                        y=close_data['Upper_Band'],
                        mode='lines',
                        name='Upper Bollinger Band',
                        line=dict(color='blue', width=1, dash='dash')
                    ))

                    close_fig.add_trace(go.Scatter(
                        x=close_data['Datetime'],
                        y=close_data['Lower_Band'],
                        mode='lines',
                        name='Lower Bollinger Band',
                        line=dict(color='blue', width=1, dash='dash')
                    ))

                    # Highlight Trends (Uptrend = Green, Downtrend = Red)
                    close_fig.add_trace(go.Scatter(
                        x=close_data['Datetime'][close_data['Trend'] == 'Uptrend'],
                        y=close_data['Predicted Close Price'][close_data['Trend'] == 'Uptrend'],
                        mode='markers+text',
                        name='Uptrend',
                        marker=dict(color='green', size=8, symbol='triangle-up'),
                        text="UP",
                        textposition="top center"
                    ))

                    close_fig.add_trace(go.Scatter(
                        x=close_data['Datetime'][close_data['Trend'] == 'Downtrend'],
                        y=close_data['Predicted Close Price'][close_data['Trend'] == 'Downtrend'],
                        mode='markers+text',
                        name='Downtrend',
                        marker=dict(color='red', size=8, symbol='triangle-down'),
                        text="DOWN",
                        textposition="top center"
                    ))

                    # Customize layout
                    close_fig.update_layout(
                        title="Predicted Close Prices with Bollinger Bands & Trend Detection",
                        xaxis=dict(title="Datetime"),
                        yaxis=dict(title="Predicted Close Price"),
                        template="plotly_white",
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                        height=600
                    )


                    # Add gridlines and rotation for x-axis labels
                    close_fig.update_xaxes(
                        tickangle=45,  # Rotate x-axis labels by 45 degrees
                        showgrid=True
                    )
                    close_fig.update_yaxes(showgrid=True)
                    st.plotly_chart(close_fig, use_container_width=True)

                    closeRSIPrediction(close_data)

                elif menu_option_trained == "Use Pre-Trained Model":
                    file_name = "close_model_weights.h5"

                    # Check if the file exists in the specified directory
                    file_path = os.path.join(PROJECT_FOLDER, file_name)

                    if os.path.isfile(file_path):
                        print(f"File '{file_name}' exists in the directory '{PROJECT_FOLDER}'.")        
                    else:
                        st.success("Please wait...")
                        print(f"File '{file_name}' does NOT exist in the directory '{PROJECT_FOLDER}'.")
                        tuner = kt.RandomSearch(
                            build_model_intraday, 
                            objective='val_loss', 
                            max_trials=1, 
                            executions_per_trial=1,
                            directory='hyperparam_tuning',  # Save tuning results
                            project_name='lstm_stock_forecast_intraday'
                        )

                        # Perform hyperparameter search
                        tuner.search(
                            x_train, 
                            y_train, 
                            epochs=EPOCHS, 
                            validation_data=(x_test, y_test),
                            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
                        )

                        # Get the best model
                        model = tuner.get_best_models(num_models=1)[0]
                        
                        print("saving weights")
                        model.save(os.path.join(PROJECT_FOLDER, 'close_model_weights.h5'))
                        test_predictions_baseline = model.predict(x_test)

                        test_predictions_baseline_padded = np.zeros((test_predictions_baseline.shape[0], x_test.shape[2]))
                        test_predictions_baseline_padded[:, 3] = test_predictions_baseline.flatten()

                        # Perform inverse transform
                        x_test_predictions_baseline = scaler.inverse_transform(test_predictions_baseline_padded)[:, 3]  # Extract only the first column

                        predicted_value = pd.DataFrame({
                                f'{stock_symbol}_actual': test_data['Close'],  # Actual Close price
                                f'{stock_symbol}_lstm_predicted': x_test_predictions_baseline  # Predicted Close price
                            })
                        predicted_value[f'{stock_symbol}_residual'] = predicted_value[f'{stock_symbol}_actual'] - predicted_value[f'{stock_symbol}_lstm_predicted']

                        X_train, X_test, y_train, y_test = train_test_split(predicted_value[[f'{stock_symbol}_lstm_predicted']], predicted_value[f'{stock_symbol}_residual'], test_size=0.2, shuffle=False)
                        xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
                        xgb_model.fit(X_train, y_train)

                        predicted_value[f'{stock_symbol}_xgb_residual_correction'] = xgb_model.predict(predicted_value[[f'{stock_symbol}_lstm_predicted']])
                        predicted_value[f'{stock_symbol}_predicted'] = predicted_value[f'{stock_symbol}_lstm_predicted'] + predicted_value[f'{stock_symbol}_xgb_residual_correction']
                        predicted_value = predicted_value.drop(columns=[f'{stock_symbol}_xgb_residual_correction', f'{stock_symbol}_lstm_predicted', f'{stock_symbol}_residual'])

                        predicted_value.index = test_data.index
                        predicted_value = predicted_value.drop(columns=['Unnamed: 0', 'Datetime.1'], errors='ignore')
                        predicted_value = predicted_value.reset_index()  # Moves index to a column
                        # print(predicted_value)

                        predicted_value.to_csv(os.path.join(PROJECT_FOLDER, 'predictions.csv'))
                        st.success("Model training success")
                        print("st.session_state: ", st.session_state)

                        

                    predicted_value=pd.read_csv(os.path.join(PROJECT_FOLDER, 'predictions.csv'))
                    predicted_value = predicted_value.drop(columns=['Unnamed: 0', 'Datetime.1'], errors='ignore')
                    predicted_value = predicted_value.tail(50)
                    st.markdown('<h2 class="subheader">Predicted and Actual Data</h2>', unsafe_allow_html=True)
                    get_accuracy(predicted_value[f'{stock_symbol}_actual'].values, predicted_value[f'{stock_symbol}_predicted'].values)
                    
                    st.write(predicted_value.sort_values(by=predicted_value.columns[0], ascending=False))

                    plotActualPredictedValue(predicted_value)

                    print("prediction is finished")

                    model = tf.keras.models.load_model(os.path.join(PROJECT_FOLDER, "close_model_weights.h5"), custom_objects={"mse": mse})

                    # Adjust the number of features dynamically
                    num_features = x_test.shape[2]  # Number of features in the dataset

                    # Perform inverse scaling on predicted values
                    predictions = model.predict(x_test)
                    # predictions = predictions[:PREDICTED_TIME]
                    # y_test = y_test[:PREDICTED_TIME]
                    predictions = predictions[-PREDICTED_TIME:]
                    y_test = y_test[-PREDICTED_TIME:]

                    # Inverse scaling for 'Close' price
                    predicted_close = scaler.inverse_transform(np.column_stack((
                        np.zeros((len(predictions), 3)),  # Placeholders for Open, High, Low
                        predictions,  # Predicted Close (assuming it is the first column of predictions)
                        np.zeros((len(predictions), 13))  # Placeholder for remaining features
                    )))[:, 3]  # Here we select index 3 for 'Close' if 'Close' is the fourth column

                    if prediction_size_menu == "Next 15-minute prediction" or prediction_size_menu == "Next 1-hour prediction":
                        predicted_dates = generate_market_time_range(PREDICT_START_DATE, PREDICTED_TIME, 5)
                    else:
                        predicted_dates = generate_market_time_range_5_minute(PREDICT_START_DATE, PREDICTED_TIME)

                    # Add predictions and actual values to test_data
                    close_data = pd.DataFrame({
                        'Datetime': predicted_dates,
                        'Predicted Close Price': predicted_close,
                    })
                    close_data.to_csv(os.path.join(PROJECT_FOLDER, "predicted_data.csv"), index=False)
                    close_data['SMA'] = close_data['Predicted Close Price'].rolling(WINDOW).mean()
                    close_data['Upper_Band'] = close_data['SMA'] + (close_data['Predicted Close Price'].rolling(WINDOW).std() * 2)
                    close_data['Lower_Band'] = close_data['SMA'] - (close_data['Predicted Close Price'].rolling(WINDOW).std() * 2)

                    close_data['Trend'] = np.where(close_data['Predicted Close Price'] > close_data['Upper_Band'], 'Uptrend',
                                np.where(close_data['Predicted Close Price'] < close_data['Lower_Band'], 'Downtrend', 'Neutral'))


                    st.markdown('<h2 class="subheader">Predicted Close Price</h2>', unsafe_allow_html=True)
                    st.write(close_data)

                    # Create the plot using Plotly
                    close_fig = go.Figure()

                    # Add the predicted close price line
                    close_fig.add_trace(go.Scatter(
                        x=predicted_dates,
                        y=predicted_close,
                        mode='lines',
                        name='Predicted Close Price',
                        line=dict(color='red', width=2)
                    ))

                    # Add Bollinger Bands
                    close_fig.add_trace(go.Scatter(
                        x=close_data['Datetime'],
                        y=close_data['Upper_Band'],
                        mode='lines',
                        name='Upper Bollinger Band',
                        line=dict(color='blue', width=1, dash='dash')
                    ))

                    close_fig.add_trace(go.Scatter(
                        x=close_data['Datetime'],
                        y=close_data['Lower_Band'],
                        mode='lines',
                        name='Lower Bollinger Band',
                        line=dict(color='blue', width=1, dash='dash')
                    ))

                    # Highlight Trends (Uptrend = Green, Downtrend = Red)
                    close_fig.add_trace(go.Scatter(
                        x=close_data['Datetime'][close_data['Trend'] == 'Uptrend'],
                        y=close_data['Predicted Close Price'][close_data['Trend'] == 'Uptrend'],
                        mode='markers+text',
                        name='Uptrend',
                        marker=dict(color='green', size=8, symbol='triangle-up'),
                        text="UP",
                        textposition="top center"
                    ))

                    close_fig.add_trace(go.Scatter(
                        x=close_data['Datetime'][close_data['Trend'] == 'Downtrend'],
                        y=close_data['Predicted Close Price'][close_data['Trend'] == 'Downtrend'],
                        mode='markers+text',
                        name='Downtrend',
                        marker=dict(color='red', size=8, symbol='triangle-down'),
                        text="DOWN",
                        textposition="top center"
                    ))

                    # Customize layout
                    close_fig.update_layout(
                        title="Predicted Close Prices with Bollinger Bands & Trend Detection",
                        xaxis=dict(title="Datetime"),
                        yaxis=dict(title="Predicted Close Price"),
                        template="plotly_white",
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                        height=600
                    )

                    # Add gridlines and rotation for x-axis labels
                    close_fig.update_xaxes(
                        tickangle=45,  # Rotate x-axis labels by 45 degrees
                        showgrid=True
                    )
                    close_fig.update_yaxes(showgrid=True)
                    st.plotly_chart(close_fig, use_container_width=True)
                    closeRSIPrediction(close_data)

        except Exception as e:
            error_details = traceback.format_exc()  # Full traceback
            st.warning(f"⚠️ Error: {e} \n\nPossible causes: API data is null or incorrect.")  
            st.text_area("Detailed Error Log", error_details, height=150)
    else:
        try:
            six_months_ago = datetime.date.today() - datetime.timedelta(days=6*30)  # Approximate 6 months

            start_date = st.date_input("Enter the start date:", datetime.date(2018, 1, 1), min_value=datetime.date(2017, 1, 1), max_value=six_months_ago)
        
            end_date = datetime.date.today()
            # Calculate the duration from start_date to today
            today = datetime.datetime.today().date()
            total_duration = (today - start_date).days  # Total days between start_date and today

            # Calculate 70% of the total duration
            validation_duration = int(0.7 * total_duration)

            # Calculate validation_date
            validation_date = start_date + timedelta(days=validation_duration)

            PREDICTED_TIME = 30
            PREDICT_START_DATE = datetime.datetime.today()

            if start_date > end_date:
                st.error("Start date must be earlier than end date.")
            else:
                stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
                if stock_data.empty:
                    st.error("No data found for the given date range. Please try a different range.")
                else:
                    stock_data.index = pd.to_datetime(stock_data.index)
                    stock_data.index = stock_data.index.tz_localize("Asia/Kolkata")
                    stock_data.reset_index(inplace=True)
                    stock_data = stock_data.drop(index=stock_data.index[0])
                    # Rename 'Date' to 'Datetime'
                    if 'Date' in stock_data.columns:
                        stock_data.rename(columns={'Date': 'Datetime'}, inplace=True)
                    stock_data.to_csv(os.path.join(PROJECT_FOLDER, 'downloaded_data_'+ stock_symbol+'.csv'), index=False)
                    stock_data=pd.read_csv(os.path.join(PROJECT_FOLDER, 'downloaded_data_'+ stock_symbol +'.csv'))
                    stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
                    st.success(f"Downloaded data from {start_date} to {end_date}.")
                    
                    stockInfo(stock_symbol)
                    stockGraph(stock_data)

                    prediction_model_menu = st.radio(
                        "Select Model for Prediction",
                        ("ARIMA Model", "LSTM Model"),
                        help="Select one of the prediction model to forecast stock prices.", index=1
                    )
                    if prediction_model_menu == "ARIMA Model":
                        st.markdown('<h2 class="subheader">ARIMA Prediction</h2>', unsafe_allow_html=True)
                        p, d, q = 5, 1, 0
                        data = np.asarray(stock_data['Close'][-300:]).reshape(-1, 1)
                        model = ARIMA(data, order=(p,d,q))
                        fitted = model.fit(method_kwargs={'maxiter': 3000})
                        fitted = model.fit(method_kwargs={'xtol': 1e-6})

                        # Generate a daily interval range
                        predicted_dates = pd.date_range(
                            start=PREDICT_START_DATE,
                            periods=PREDICTED_TIME,
                            freq='D'  # 'D' stands for daily frequency
                        )

                        # Predict for the generated date range
                        steps = len(predicted_dates)  # Number of steps to forecast
                        forecast_values = fitted.forecast(steps=steps)  # Directly returns a numpy array

                        # Visualize ARIMA Forecasted Prices
                        forecast_arima_df = pd.DataFrame({'Datetime': predicted_dates, 'Predicted Close': forecast_values})
                        fig = px.line(forecast_arima_df, x='Datetime', y='Predicted Close')
                        st.plotly_chart(fig)
                    else:
                        stock_data['Volume'] = pd.to_numeric(stock_data['Volume'], errors='coerce')
            
                        # ATR (Average True Range)
                        stock_data['High'] = pd.to_numeric(stock_data['High'], errors='coerce')
                        stock_data['Low'] = pd.to_numeric(stock_data['Low'], errors='coerce')
                        stock_data['Open'] = pd.to_numeric(stock_data['Open'], errors='coerce')

                        stock_data = calculate_rsi(stock_data)
                        stock_data = calculate_macd(stock_data)
                        stock_data = calculate_bollinger_bands(stock_data)
                        stock_data = calculate_atr(stock_data)  
                        stock_data = calculate_stochastic(stock_data)
                        stock_data = generate_signals(stock_data)


                        stock_data['VWAP'] = (stock_data['Volume'] * (stock_data['High'] + stock_data['Low'] + stock_data['Close']) / 3).cumsum() / stock_data['Volume'].cumsum()
                        stock_data['Avg_Volume'] = round(stock_data['Volume'].rolling(window=10).mean(),0)
                        stock_data['ATR'] = stock_data['True Range'].rolling(window=10).mean() 

                        stock_data['Log_Close'] = np.log(stock_data['Close'])  # Stabilize fluctuations
                        stock_data['Diff_Close'] = stock_data['Close'].diff()  # Remove trends

                        stock_data.dropna(inplace=True)

                        st.markdown('<h2 class="subheader">Stock Prices with Indicator Data</h2>', unsafe_allow_html=True)

                        stock_data.to_csv(os.path.join(PROJECT_FOLDER, 'data_'+stock_symbol+'.csv'), index=False)
                        
                        st.dataframe(stock_data.tail(50).sort_values(by=stock_data.columns[0], ascending=False))


                        # Create a subplot figure with 3 rows
                        fig = go.Figure()

                        # Candlestick Chart
                        fig.add_trace(go.Candlestick(
                            x=stock_data["Datetime"],
                            open=stock_data["Open"],
                            high=stock_data["High"],
                            low=stock_data["Low"],
                            close=stock_data["Close"],
                            name="Candlestick"
                        ))

                        fig.add_trace(go.Scatter(x=stock_data['Datetime'], y=stock_data['Upper_Band'], name="Upper Band", line=dict(color='red', dash='dot')))
                        fig.add_trace(go.Scatter(x=stock_data['Datetime'], y=stock_data['Lower_Band'], name="Lower Band", line=dict(color='blue', dash='dot')))
                        fig.add_trace(go.Scatter(x=stock_data['Datetime'], y=stock_data['SMA'], name="SMA", line=dict(color='orange')))
                        
                        buy_signals = stock_data[stock_data['B/S_Signal'] == "Strong BUY"]
                        sell_signals = stock_data[stock_data['B/S_Signal'] == "Sell"]
                        stop_loss_hits = stock_data[stock_data['B/S_Signal'] == "Stop-Loss Hit"]
                        take_profit_hits = stock_data[stock_data['B/S_Signal'] == "Take-Profit Hit"]

                        fig.add_trace(go.Scatter(
                            x=buy_signals['Datetime'], 
                            y=buy_signals['Close'], 
                            mode='markers+text', name='Buy Signal', marker=dict(color='green', size=8, symbol='triangle-up'), text=["BUY"] * len(buy_signals), textposition="top center", textfont=dict(size=12, color="green")
                        ))
                        fig.add_trace(go.Scatter(
                            x=sell_signals['Datetime'], 
                            y=sell_signals['Close'], 
                            mode='markers+text', name='Sell Signal', marker=dict(color='red', size=8, symbol='triangle-down'), text=["SELL"] * len(sell_signals), textposition="top center", textfont=dict(size=12, color="red")
                        ))
                        fig.add_trace(go.Scatter(
                            x=stop_loss_hits['Datetime'], 
                            y=stop_loss_hits['Close'], 
                            mode='markers', name='Stop-Loss Hit', marker=dict(color='orange', size=10, symbol='x'), showlegend=True))
                        fig.add_trace(go.Scatter(
                            x=take_profit_hits['Datetime'], 
                            y=take_profit_hits['Close'], 
                            mode='markers', name='Take-Profit Hit', marker=dict(color='blue', size=10, symbol='star'), showlegend=True
                        ))
                        fig.update_layout(title=f"{stock_symbol} Price & Bollinger Bands", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False)

                        st.plotly_chart(fig)

                        # stockIndicatorForSwing(stock_data)

                        menu_option_trained = st.radio(
                            "Choose an option:", 
                            ["None Selected","Train New Model", "Use Pre-Trained Model"], 
                            index=0,
                            )
                        stock_data['Datetime'] = pd.to_datetime(stock_data['Datetime'], errors='coerce')
                        training_data = stock_data[stock_data['Datetime'] < pd.Timestamp(validation_date).tz_localize("Asia/Kolkata")].copy()
                        test_data = stock_data[stock_data['Datetime'] >= pd.Timestamp(validation_date).tz_localize('Asia/Kolkata')].copy()
                        training_data = training_data.set_index('Datetime')
                        # Set the data frame index using column Date
                        test_data = test_data.set_index('Datetime')

                        train_scaled = scaler.fit_transform(training_data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Signal_Line', '%K', '%D', 'Upper_Band', 'Lower_Band', 'ATR', 'Log_Close', 'Diff_Close', 'SMA' ]])
                        
                        # Training Data Transformation
                        x_train = []
                        y_train = []
                        for i in range(TIME_STEPS, len(train_scaled)):
                            x_train.append(train_scaled[i - TIME_STEPS:i])
                            y_train.append(train_scaled[i, 3]) 

                        x_train, y_train = np.array(x_train), np.array(y_train)
                        total_data = pd.concat((training_data, test_data), axis=0)
                        inputs = total_data[len(total_data) - len(test_data) - TIME_STEPS:]
                        test_scaled = scaler.transform(inputs[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Signal_Line', '%K', '%D', 'Upper_Band', 'Lower_Band', 'ATR', 'Log_Close', 'Diff_Close', 'SMA']])
                        
                        # Testing Data Transformation
                        x_test = []
                        y_test = []
                        for i in range(TIME_STEPS, len(test_scaled)):
                            x_test.append(test_scaled[i - TIME_STEPS:i])
                            y_test.append(test_scaled[i, 3])

                        x_test, y_test = np.array(x_test), np.array(y_test)
                        if menu_option_trained == "None Selected":
                            print("None Selected")
                        elif menu_option_trained == "Train New Model":
                            # Define the tuner
                            tuner = kt.RandomSearch(
                                build_model, 
                                objective='val_loss', 
                                max_trials=1, 
                                executions_per_trial=1,
                                directory='hyperparam_tuning',  # Save tuning results
                                project_name='lstm_stock_forecast'
                            )

                            # Perform hyperparameter search
                            tuner.search(
                                x_train, 
                                y_train, 
                                epochs=EPOCHS, 
                                validation_data=(x_test, y_test),
                                callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
                            )

                            # Get the best model
                            model = tuner.get_best_models(num_models=1)[0]
                            print("saving weights")
                            model.save(os.path.join(PROJECT_FOLDER, 'close_model_weights.h5'))
                            test_predictions_baseline = model.predict(x_test)

                            test_predictions_baseline_padded = np.zeros((test_predictions_baseline.shape[0], x_test.shape[2]))
                            test_predictions_baseline_padded[:, 3] = test_predictions_baseline.flatten()

                            # Perform inverse transform
                            x_test_predictions_baseline = scaler.inverse_transform(test_predictions_baseline_padded)[:, 3]  # Extract only the first column

                            test_predictions_baseline = pd.DataFrame({
                                    f'{stock_symbol}_actual': test_data['Close'],  # Actual Close price
                                    f'{stock_symbol}_lstm_predicted': x_test_predictions_baseline  # Predicted Close price
                                })
                            test_predictions_baseline[f'{stock_symbol}_residual'] = test_predictions_baseline[f'{stock_symbol}_actual'] - test_predictions_baseline[f'{stock_symbol}_lstm_predicted']

                            X_train, X_test, y_train, y_test = train_test_split(test_predictions_baseline[[f'{stock_symbol}_lstm_predicted']], test_predictions_baseline[f'{stock_symbol}_residual'], test_size=0.2, shuffle=False)
                            xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
                            xgb_model.fit(X_train, y_train)

                            test_predictions_baseline[f'{stock_symbol}_xgb_residual_correction'] = xgb_model.predict(test_predictions_baseline[[f'{stock_symbol}_lstm_predicted']])
                            test_predictions_baseline[f'{stock_symbol}_predicted'] = test_predictions_baseline[f'{stock_symbol}_lstm_predicted'] + test_predictions_baseline[f'{stock_symbol}_xgb_residual_correction']
                            test_predictions_baseline = test_predictions_baseline.drop(columns=[f'{stock_symbol}_xgb_residual_correction', f'{stock_symbol}_lstm_predicted', f'{stock_symbol}_residual'])

                            test_predictions_baseline.index = test_data.index
                            test_predictions_baseline = test_predictions_baseline.reset_index()  # Moves index to a column

                            test_predictions_baseline.to_csv(os.path.join(PROJECT_FOLDER, 'predictions.csv'))

                            st.markdown('<h2 class="subheader">Predicted and Actual Data</h2>', unsafe_allow_html=True)
                            get_accuracy(test_predictions_baseline[f'{stock_symbol}_actual'].values, test_predictions_baseline[f'{stock_symbol}_predicted'].values)
                    
                            
                            st.write(test_predictions_baseline.tail(50).sort_values(by=test_predictions_baseline.columns[0], ascending=False)) 
                            # print(test_predictions_baseline)
                            plotActualPredictedValue(test_predictions_baseline.tail(50))

                            model = tf.keras.models.load_model(os.path.join(PROJECT_FOLDER, "close_model_weights.h5"), custom_objects={"mse": mse})

                            # Adjust the number of features dynamically
                            num_features = x_test.shape[2]  # Number of features in the dataset

                            # Perform inverse scaling on predicted values
                            predictions = model.predict(x_test)

                            predictions = predictions[-PREDICTED_TIME:]
                            y_test = y_test[-PREDICTED_TIME:]

                            # Inverse scaling for 'Close' price
                            predicted_close = scaler.inverse_transform(np.column_stack((
                                np.zeros((len(predictions), 3)),  # Placeholders for Open, High, Low
                                predictions,  # Predicted Close (assuming it is the first column of predictions)
                                np.zeros((len(predictions), 12))  # Placeholder for remaining features
                            )))[:, 3]  # Here we select index 3 for 'Close' if 'Close' is the fourth column
                        
                            predicted_dates = generate_market_time_range_daily(PREDICT_START_DATE, PREDICTED_TIME)
                            
                            # Add predictions and actual values to test_data
                            close_data = pd.DataFrame({
                                'Datetime': predicted_dates,
                                'Predicted Close Price': predicted_close,
                            })
                            close_data.to_csv(os.path.join(PROJECT_FOLDER, "predicted_data.csv"), index=False)
                            close_data['SMA_Short'] = close_data['Predicted Close Price'].rolling(window=5).mean()
                            close_data['SMA_Long'] = close_data['Predicted Close Price'].rolling(window=10).mean()

                            close_data['Trend'] = np.where(close_data['SMA_Short'] > close_data['SMA_Long'], 'Uptrend',
                                        np.where(close_data['SMA_Short'] < close_data['SMA_Long'], 'Downtrend', 'Neutral'))


                            st.markdown('<h2 class="subheader">Predicted Close Price</h2>', unsafe_allow_html=True)
                            st.write(close_data)

                            # Create the plot using Plotly
                            close_fig = go.Figure()

                            # Add the predicted close price line
                            close_fig.add_trace(go.Scatter(
                                x=predicted_dates,
                                y=predicted_close,
                                mode='lines',
                                name='Predicted Close Price',
                                line=dict(color='red', width=2)
                            ))

                            # Plot SMA Short (5-day)
                            close_fig.add_trace(go.Scatter(
                                x=close_data['Datetime'], y=close_data['SMA_Short'],
                                mode='lines', name='SMA Short (5-day)',
                                line=dict(color='blue', width=1, dash='dot')
                            ))

                            # Plot SMA Long (10-day)
                            close_fig.add_trace(go.Scatter(
                                x=close_data['Datetime'], y=close_data['SMA_Long'],
                                mode='lines', name='SMA Long (10-day)',
                                line=dict(color='green', width=1, dash='dot')
                            ))

                            # Highlight Uptrends & Downtrends
                            close_fig.add_trace(go.Scatter(
                                x=close_data['Datetime'][close_data['Trend'] == 'Uptrend'],
                                y=close_data['Predicted Close Price'][close_data['Trend'] == 'Uptrend'],
                                mode='markers+text',
                                name='Uptrend',
                                marker=dict(color='green', size=8, symbol='triangle-up'),
                                text="UP",
                                textposition="top center"
                            ))

                            close_fig.add_trace(go.Scatter(
                                x=close_data['Datetime'][close_data['Trend'] == 'Downtrend'],
                                y=close_data['Predicted Close Price'][close_data['Trend'] == 'Downtrend'],
                                mode='markers+text',
                                name='Downtrend',
                                marker=dict(color='red', size=8, symbol='triangle-down'),
                                text="DOWN",
                                textposition="top center"
                            ))

                            # Customize Layout
                            close_fig.update_layout(
                                title="Predicted Close Prices with SMA Trend Detection",
                                xaxis=dict(title="Datetime"),
                                yaxis=dict(title="Close Price"),
                                template="plotly_white",
                                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                                height=600
                            )

                            # Add gridlines and rotation for x-axis labels
                            close_fig.update_xaxes(
                                tickangle=45,  # Rotate x-axis labels by 45 degrees
                                showgrid=True
                            )
                            close_fig.update_yaxes(showgrid=True)
                            st.plotly_chart(close_fig, use_container_width=True)

                            closeRSIPredictionForSwing(close_data)

                        elif menu_option_trained == "Use Pre-Trained Model":
                            file_name = "close_model_weights.h5"

                            # Check if the file exists in the specified directory
                            file_path = os.path.join(PROJECT_FOLDER, file_name)

                            if os.path.isfile(file_path):
                                print(f"File '{file_name}' exists in the directory '{PROJECT_FOLDER}'.")        
                            else:
                                st.success("Please wait...")
                                print(f"File '{file_name}' does NOT exist in the directory '{PROJECT_FOLDER}'.")
                                tuner = kt.RandomSearch(
                                    build_model, 
                                    objective='val_loss', 
                                    max_trials=1, 
                                    executions_per_trial=1,
                                    directory='hyperparam_tuning',  # Save tuning results
                                    project_name='lstm_stock_forecast'
                                )

                                # Perform hyperparameter search
                                tuner.search(
                                    x_train, 
                                    y_train, 
                                    epochs=EPOCHS, 
                                    validation_data=(x_test, y_test),
                                    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
                                )

                                # Get the best model
                                model = tuner.get_best_models(num_models=1)[0]
                                print("saving weights")
                                model.save(os.path.join(PROJECT_FOLDER, 'close_model_weights.h5'))
                                test_predictions_baseline = model.predict(x_test)

                                test_predictions_baseline_padded = np.zeros((test_predictions_baseline.shape[0], x_test.shape[2]))
                                test_predictions_baseline_padded[:, 3] = test_predictions_baseline.flatten()

                                # Perform inverse transform
                                x_test_predictions_baseline = scaler.inverse_transform(test_predictions_baseline_padded)[:, 3]  # Extract only the first column

                                predicted_value = pd.DataFrame({
                                        f'{stock_symbol}_actual': test_data['Close'],  # Actual Close price
                                        f'{stock_symbol}_lstm_predicted': x_test_predictions_baseline  # Predicted Close price
                                    })
                                predicted_value[f'{stock_symbol}_residual'] = predicted_value[f'{stock_symbol}_actual'] - predicted_value[f'{stock_symbol}_lstm_predicted']

                                X_train, X_test, y_train, y_test = train_test_split(predicted_value[[f'{stock_symbol}_lstm_predicted']], predicted_value[f'{stock_symbol}_residual'], test_size=0.2, shuffle=False)
                                xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
                                xgb_model.fit(X_train, y_train)

                                predicted_value[f'{stock_symbol}_xgb_residual_correction'] = xgb_model.predict(predicted_value[[f'{stock_symbol}_lstm_predicted']])
                                predicted_value[f'{stock_symbol}_predicted'] = predicted_value[f'{stock_symbol}_lstm_predicted'] + predicted_value[f'{stock_symbol}_xgb_residual_correction']
                                predicted_value = predicted_value.drop(columns=[f'{stock_symbol}_xgb_residual_correction', f'{stock_symbol}_lstm_predicted', f'{stock_symbol}_residual'])

                                predicted_value.index = test_data.index
                                predicted_value = predicted_value.drop(columns=['Unnamed: 0', 'Datetime.1'], errors='ignore')
                                predicted_value = predicted_value.reset_index()  # Moves index to a column
                                # print(predicted_value)

                                predicted_value.to_csv(os.path.join(PROJECT_FOLDER, 'predictions.csv'))
                                st.success("Model training success")
                                print("st.session_state: ", st.session_state)

                                

                            predicted_value=pd.read_csv(os.path.join(PROJECT_FOLDER, 'predictions.csv'))
                            predicted_value = predicted_value.drop(columns=['Unnamed: 0', 'Datetime.1'], errors='ignore')
                            st.markdown('<h2 class="subheader">Predicted and Actual Data</h2>', unsafe_allow_html=True)
                            get_accuracy(predicted_value[f'{stock_symbol}_actual'].values, predicted_value[f'{stock_symbol}_predicted'].values)
                    
                            st.write(predicted_value.tail(50).sort_values(by=predicted_value.columns[0], ascending=False)) 

                            plotActualPredictedValue(predicted_value.tail(50))

                            print("prediction is finished")

                            model = tf.keras.models.load_model(os.path.join(PROJECT_FOLDER, "close_model_weights.h5"), custom_objects={"mse": mse})

                            # Adjust the number of features dynamically
                            num_features = x_test.shape[2]  # Number of features in the dataset

                            # Perform inverse scaling on predicted values
                            predictions = model.predict(x_test)
                            # predictions = predictions[:PREDICTED_TIME]
                            # y_test = y_test[:PREDICTED_TIME]
                            predictions = predictions[-PREDICTED_TIME:]
                            y_test = y_test[-PREDICTED_TIME:]

                            # Inverse scaling for 'Close' price
                            predicted_close = scaler.inverse_transform(np.column_stack((
                                np.zeros((len(predictions), 3)),  # Placeholders for Open, High, Low
                                predictions,  # Predicted Close (assuming it is the first column of predictions)
                                np.zeros((len(predictions), 12))  # Placeholder for remaining features
                            )))[:, 3]  # Here we select index 3 for 'Close' if 'Close' is the fourth column

                            predicted_dates = generate_market_time_range_daily(PREDICT_START_DATE, PREDICTED_TIME)
                            
                            # Add predictions and actual values to test_data
                            close_data = pd.DataFrame({
                                'Datetime': predicted_dates,
                                'Predicted Close Price': predicted_close,
                            })
                            close_data.to_csv(os.path.join(PROJECT_FOLDER, "predicted_data.csv"), index=False)
                            close_data.to_csv(os.path.join(PROJECT_FOLDER, "predicted_data.csv"), index=False)
                            close_data['SMA_Short'] = close_data['Predicted Close Price'].rolling(window=5).mean()
                            close_data['SMA_Long'] = close_data['Predicted Close Price'].rolling(window=10).mean()

                            close_data['Trend'] = np.where(close_data['SMA_Short'] > close_data['SMA_Long'], 'Uptrend',
                                        np.where(close_data['SMA_Short'] < close_data['SMA_Long'], 'Downtrend', 'Neutral'))


                            st.markdown('<h2 class="subheader">Predicted Close Price</h2>', unsafe_allow_html=True)
                            st.write(close_data)
                            # Create the plot using Plotly
                            close_fig = go.Figure()

                            # Add the predicted close price line
                            close_fig.add_trace(go.Scatter(
                                x=predicted_dates,
                                y=predicted_close,
                                mode='lines',
                                name='Predicted Close Price',
                                line=dict(color='red', width=2)
                            ))

                                                        # Plot SMA Short (5-day)
                            close_fig.add_trace(go.Scatter(
                                x=close_data['Datetime'], y=close_data['SMA_Short'],
                                mode='lines', name='SMA Short (5-day)',
                                line=dict(color='blue', width=1, dash='dot')
                            ))

                            # Plot SMA Long (10-day)
                            close_fig.add_trace(go.Scatter(
                                x=close_data['Datetime'], y=close_data['SMA_Long'],
                                mode='lines', name='SMA Long (10-day)',
                                line=dict(color='green', width=1, dash='dot')
                            ))

                            # Highlight Uptrends & Downtrends
                            close_fig.add_trace(go.Scatter(
                                x=close_data['Datetime'][close_data['Trend'] == 'Uptrend'],
                                y=close_data['Predicted Close Price'][close_data['Trend'] == 'Uptrend'],
                                mode='markers+text',
                                name='Uptrend',
                                marker=dict(color='green', size=8, symbol='triangle-up'),
                                text="UP",
                                textposition="top center"
                            ))

                            close_fig.add_trace(go.Scatter(
                                x=close_data['Datetime'][close_data['Trend'] == 'Downtrend'],
                                y=close_data['Predicted Close Price'][close_data['Trend'] == 'Downtrend'],
                                mode='markers+text',
                                name='Downtrend',
                                marker=dict(color='red', size=8, symbol='triangle-down'),
                                text="DOWN",
                                textposition="top center"
                            ))

                            # Customize Layout
                            close_fig.update_layout(
                                title="Predicted Close Prices with SMA Trend Detection",
                                xaxis=dict(title="Datetime"),
                                yaxis=dict(title="Close Price"),
                                template="plotly_white",
                                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                                height=600
                            )

                            # Add gridlines and rotation for x-axis labels
                            close_fig.update_xaxes(
                                tickangle=45,  # Rotate x-axis labels by 45 degrees
                                showgrid=True
                            )
                            close_fig.update_yaxes(showgrid=True)
                            st.plotly_chart(close_fig, use_container_width=True)
                            closeRSIPredictionForSwing(close_data)


        except Exception as e:
            error_details = traceback.format_exc()  # Full traceback
            st.warning(f"⚠️ Error: {e} \n\nPossible causes: API data is null or incorrect.")  
            st.text_area("Detailed Error Log", error_details, height=150)
