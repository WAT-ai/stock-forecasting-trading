from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ==== GETTING DATA FROM API ==== 
API_KEY = 'DNWQMLFC43J1PHDI'
TS = TimeSeries(key=API_KEY, output_format='pandas')


def get_stock_data(symbol, ts, api_key, days_back=1):
    """
    Fetches historical stock data and technical indicators for a given stock symbol.
    Input: Symbol of the stock (string), TimeSeries object, API key (string)
    Output: DataFrame of stock data (pd.DataFrame)
    """

    # Fetch historical stock data
    historical_data, _ = ts.get_daily(symbol=symbol, outputsize='full')

    #and grab the last 5 elements
    historical_data = historical_data.head(days_back)

    #reverse the list so el 5 becomes el 1
    historical_data = historical_data.iloc[::-1]

    # print(historical_data.index)

    # Fetch technical indicators
    ti = TechIndicators(key=api_key, output_format='pandas')
    rsi, _ = ti.get_rsi(symbol=symbol, interval='daily', time_period=14)
    macd, _ = ti.get_macd(symbol=symbol, interval='daily', series_type='close')
    adx, _ = ti.get_adx(symbol=symbol, interval='daily', time_period=14)
    cci, _ = ti.get_cci(symbol=symbol, interval='daily', time_period=14)

    # print(macd.index[:5][::-1])
    # print(rsi.index[-5:])
    # Check if all dataframes have the same index
    if not all(df.index.equals(historical_data.index) for df in [rsi[-days_back:], adx[-days_back:], cci[-days_back:], macd[:days_back][::-1]]):
        raise ValueError("Indices of DataFrames do not match.")

    # Combine all data into one DataFrame
    df = pd.concat([historical_data['1. open'], historical_data['4. close'], rsi[-days_back:], adx[-days_back:], cci[-days_back:],  macd[:days_back][::-1]], axis=1)

    return df


# # ====  EVALUATION METRICS ====  
# def get_sharpe_ratio(symbol, ts, RFR = 0.05):
#     """
#     Gets the sharpe ratio of a stocks 
#     Input: Symbol of the stock (string), TimeSeries object, Risk Free Rate (float)
#     Output: Sharpe Ratio (float)

#     Note(s): 
#     free to modify risk free rate, just took 0.05 a a basis from website below (3 month t-bill)
#     https://ycharts.com/indicators/3_month_t_bill#:~:text=Basic%20Info,long%20term%20average%20of%204.19%25.
#     """

#     # Fetch historical stock data (last 100 data points for example)
#     data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
#     close_prices = data['4. close']

#     # Calculate daily returns
#     daily_returns = close_prices.pct_change().dropna()

#     # Calculate the Sharpe Ratio
#     excess_returns = daily_returns - RFR / 252 #252 trading days in a year
#     sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)  #sqrt to annualize sharpe ratio

#     return sharpe_ratio


def get_percentage_return(symbol, api_key):
    """
    Fetches the daily percentage returns of a stock.
    Input: Symbol of the stock (string), API key (string)
    Output: daily percentage returns (pandas Series)
    
    """
    # Initialize TimeSeries with your Alpha Vantage API Key
    ts = TimeSeries(key=api_key, output_format='pandas')

    # Fetch historical stock data (daily)
    data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
    
    # Extract closing prices
    closing_prices = data['4. close']

    # Calculate daily percentage returns
    daily_returns = closing_prices.pct_change().dropna() * 100

    return daily_returns






#==== SAMPLE OUTPUT DATA ====

stock_data = get_stock_data('MCD', TS, API_KEY, 2462)
stock_data.to_csv('data.csv')
# daily = get_percentage_return('GOOG', API_KEY)
# print(daily)

