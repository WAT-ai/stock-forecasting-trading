import yfinance as yf

# MACD and SMA Calculations
def get_stock_data(ticker):
    YUM = yf.Ticker('YUM')
    YUM_history = YUM.history(period='10y') # 10y history for now, changeable later
    YUM_history['Close'] = YUM_history['Close'].to_frame() 
    YUM_history.reset_index(drop=False, inplace=True)

    # MACD Calculation and add to df
    YUM_history['EMA12'] = YUM_history['Close'].ewm(span=12, min_periods=0, adjust=False).mean()
    YUM_history['EMA26'] = YUM_history['Close'].ewm(span=26, min_periods=0, adjust=False).mean()
    YUM_history['MACD'] = YUM_history['EMA12'] - YUM_history['EMA26']
    YUM_history['Signal'] = YUM_history['MACD'].ewm(span=9, min_periods=0, adjust=False).mean()
    YUM_history['Histogram'] = YUM_history['MACD'] - YUM_history['Signal']

    # SMA calculation and add to df
    YUM_history['SMA50'] = YUM_history['Close'].rolling(50).mean()
    YUM_history['SMA200'] = YUM_history['Close'].rolling(200).mean()
    return YUM_history