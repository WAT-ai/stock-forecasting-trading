from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ==== GETTING DATA FROM API ==== 
class Indicators:
    def __init__(self, api_key):
        self.api_key = api_key
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        self.df = pd.DataFrame()
        self.df_normalized = pd.DataFrame

    def get_stock_data(self, symbol, days_back=1):
        """
        Fetches historical stock data and technical indicators for a given stock symbol.
        Input: Symbol of the stock (string), TimeSeries object, API key (string)
        Output: DataFrame of stock data (pd.DataFrame)
        """
        # Fetch historical stock data
        historical_data, _ = self.ts.get_daily(symbol=symbol, outputsize='full')
        # Filter data to only include the last n days
        historical_data = historical_data.head(days_back)
        # Reverse the data so that it's in chronological order
        historical_data = historical_data.iloc[::-1]

        # Fetch technical indicators
        ti = TechIndicators(key=self.api_key, output_format='pandas')
        rsi, _ = ti.get_rsi(symbol=symbol, interval='daily', time_period=14)
        macd, _ = ti.get_macd(symbol=symbol, interval='daily', series_type='close')
        adx, _ = ti.get_adx(symbol=symbol, interval='daily', time_period=14)
        cci, _ = ti.get_cci(symbol=symbol, interval='daily', time_period=14)

        # Check if indices of DataFrames match
        if not all(df.index.equals(historical_data.index) for df in [rsi[-days_back:], adx[-days_back:], cci[-days_back:], macd[:days_back][::-1]]):
            raise ValueError("Indices of DataFrames do not match.")

        # Combine historical data and technical indicators into one DataFrame
        self.df = pd.concat([historical_data['1. open'], historical_data['4. close'], rsi[-days_back:], adx[-days_back:], cci[-days_back:],  macd[:days_back][::-1]], axis=1)
        return self.df
            
    def preprocess_stock_data(self):
        """
        Preprocesses the stock data DataFrame by adding lagged features, moving averages,
        daily returns, handling outliers, and normalizing the data.
        Input: DataFrame of stock data (pd.DataFrame)
        Output: preprocessed data (np.array)
        """

        if len(self.df) == 0:
            print("Error: Empty DataFrame provided.")
            return None

    
        # Drop rows with NaN values created by shifts and moving averages
        self.df.dropna(inplace=True)
        
        # Check if there's any data left after preprocessing
        if len(self.df) == 0:
            print("Error: No data left after preprocessing.")
            return None
        
        # ==== NORMALIZATION ====
        # Min-max normalization
        index = self.df.index
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_values = self.df.values
        df_normalized_values = scaler.fit_transform(df_values[:, 2:])
        
        # Concatenate normalized data with index
        self.df_normalized = pd.DataFrame(df_normalized_values, columns=self.df.columns[2:], index=index)
        self.df_normalized.insert(0, self.df.columns[0], self.df[self.df.columns[0]])  # Insert closing price
        self.df_normalized.insert(1, self.df.columns[1], self.df[self.df.columns[1]]) #Insert opening price 
        
        
        # Convert to numpy array for model
        
        return self.df_normalized

        

def load_csv(filename): 
    "get data from a csv file, save it to a dataframe with a date column and a sentiment column"
    df = pd.read_csv(filename, index_col='Date')
    return df
        


#==== SAMPLE OUTPUT DATA ====
API_KEY = 'DNWQMLFC43J1PHDI'
analyzer = Indicators(api_key=API_KEY)
analyzer.get_stock_data('MCD', 2)
analyzer.preprocess_stock_data()
analyzer.df_normalized.to_csv('test2.csv')
# daily = get_percentage_return('GOOG', API_KEY)
# print(daily)

