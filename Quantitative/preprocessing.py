from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import data_extractor
import csv 


API_KEY = 'YOUR API KEY HERE'
TS = TimeSeries(key=API_KEY, output_format='pandas')
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_stock_data(df):
    """
    Preprocesses the stock data DataFrame by adding lagged features, moving averages,
    daily returns, handling outliers, and normalizing the data.
    Input: DataFrame of stock data (pd.DataFrame)
    Output: preprocessed data (np.array)
    """

    if len(df) == 0:
        print("Error: Empty DataFrame provided.")
        return None

  
    # Drop rows with NaN values created by shifts and moving averages
    df.dropna(inplace=True)
    
    # Check if there's any data left after preprocessing
    if len(df) == 0:
        print("Error: No data left after preprocessing.")
        return None
    
    # ==== NORMALIZATION ====
    # Min-max normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    # Convert to numpy array for model
    
    
    return df_normalized


def preprocess_sentiment(): 
    "get data from a csv file, save it to a dataframe with a date column and a sentiment column"
    df = pd.read_csv('Quantitative/MCD_Sentiment_Data.csv', index_col='Date')
    return df

### Example usage
# stock_data = data_extractor.get_stock_data('AAPL', TS, API_KEY, days_back=50)
# print(stock_data)

# preprocessed_data = preprocess_stock_data(stock_data, 2)

# # Print preprocessed data
# print(preprocessed_data)

# run preprocessed_sentiment() to get the sentiment data    
sentiment_data = preprocess_sentiment()
print(sentiment_data)