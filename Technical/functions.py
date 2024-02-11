#combine as python for Combined.ipynb

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def calculate_volatility(symbol, per):
    # historical data from Yahoo Finance
    stock_data = symbol.history(period = per)
    
    stock_data['Returns'] = stock_data['Close'].pct_change()

    # volatility -> standard deviation
    volatility = np.std(stock_data['Returns'])

    return volatility

def calculate_bollinger_bands(symbol, per, window=20, num_std=2):

    stock_data = symbol.history(period=per)

    stock_data['Returns'] = stock_data['Close'].pct_change()

    #also get volatility
    stock_data['Volatility'] = stock_data['Returns'].rolling(window=window).std()

    # calculate the rolling mean and standard deviation for Bollinger Bands
    stock_data['SMA'] = stock_data['Close'].rolling(window=window).mean()
    stock_data['Upper Band'] = stock_data['SMA'] + (num_std * stock_data['Close'].rolling(window=window).std())
    stock_data['Lower Band'] = stock_data['SMA'] - (num_std * stock_data['Close'].rolling(window=window).std())

    return stock_data[['Close', 'SMA', 'Upper Band', 'Lower Band', 'Volatility']]

def macd_calc(symbol, per):
    stock_data = symbol.history(period=per)
    stock_data.get('Volume')
    stock_data.get('Close')

    stock_data['EMA12'] = stock_data['Close'].ewm(span=12, min_periods=0, adjust=False).mean()
    stock_data['EMA26'] = stock_data['Close'].ewm(span=26, min_periods=0, adjust=False).mean()
    stock_data['MACD'] = stock_data['EMA12'] - stock_data['EMA26']
    stock_data['Signal'] = stock_data['MACD'].ewm(span=9, min_periods=0, adjust=False).mean()
    stock_data['Histogram'] = stock_data['MACD'] - stock_data['Signal']

    return stock_data[['EMA12', 'EMA26', 'MACD', 'Signal', 'Histogram']]


def sma_calc(symbol, per):
    stock_data = symbol.history(period=per)
    stock_data['SMA50'] = stock_data['Close'].rolling(50).mean()
    stock_data['SMA200'] = stock_data['Close'].rolling(200).mean()

    return stock_data [['SMA50', 'SMA200']]


# 
def rsi_cal(symbol, per):
    stock_data = symbol.history(period=per)
    stock_data['RSI'] = 100 - 100 / (
            1 + (stock_data['Close'].diff() / stock_data['Close'].shift(1)).rolling(14).mean())

    return stock_data ['RSI']

# 
def volume_calc(symbol, per):
    stock_data = symbol.history(period = per)

    return  stock_data['Volume']


#
def combine_all (symbol, per):

    volatility = calculate_volatility(symbol, per)
    bollinger_bands_data = calculate_bollinger_bands(symbol, per)
    macd = macd_calc(symbol, per)
    sma = sma_calc(symbol, per)
    rsi = rsi_cal(symbol, per)
    volume = volume_calc(symbol, per)

    stock_data = bollinger_bands_data.join([macd, sma, rsi, volume])
    stock_data = stock_data.dropna()


    return stock_data

def normalize_df(df):
     #normalize data using sklearns MinMax Scaler
    date_col = df.index
    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(df)

    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    scaled_df.insert(0, 'date', date_col)

    return scaled_df

if __name__ == "__main__":


    symbol = yf.Ticker('YUM')
    per = '10y'

    # volatility = calculate_volatility(symbol, per)
    # bollinger_bands_data = calculate_bollinger_bands(symbol, per)
    # macd = macd_calc(symbol, per)
    # sma = sma_calc(symbol, per)
    # rsi = rsi_cal(symbol, per)
    # volume = volume_calc(symbol, per)

    # stock_data_df = bollinger_bands_data.join([macd, sma, rsi, volume])

    stock_data_df = combine_all(symbol, per)
    stock_data_df = normalize_df(stock_data_df)
    
    

    print(stock_data_df)





