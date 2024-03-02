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
def main (): 
    data = pd.read_csv("MCD_Sentiment_Data.csv", index_col='Date')

    # Parse the dates from the index
    data.index = pd.to_datetime(data.index)

    # Create a date range for all trading days
    full_date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='B')

    # Reindex the DataFrame with the complete date range
    data = data.reindex(full_date_range)

    # Interpolate missing sentiment scores
    data['Sentiment'] = data['Sentiment'].interpolate(method='linear')

    # Calculate the average sentiment score for each missing trading day
    missing_dates = data[data['Sentiment'].isnull()].index
    average_sentiment = data['Sentiment'].mean()

    for date in missing_dates:
        data.loc[date, 'Sentiment'] = average_sentiment

    df = pd.read_csv("sentiments.csv")

    # Define a list of holiday dates if applicable
    holidays = ["2023-12-25", "2023-12-31", "2024-01-01", "2024-02-19", "2024-01-15"]  # Add your holiday dates here
    
    print(df.head())
    # Filter out holidays
    df = df[~df['Date'].isin(holidays)]

    # Convert the 'date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter out weekends (Saturday and Sunday)
    df = df[df['Date'].dt.dayofweek < 5]

    indi_df = pd.read_csv("indicators.csv")
    indi_df['Date'] = pd.to_datetime(indi_df['Date'])

    df_close = indi_df[['Date', '4. close']].copy()

    # Create a dataframe with date and every column after 4. close
    df_indicators = indi_df[['Date'] + list(indi_df.columns[indi_df.columns.get_loc('4. close')+1:])].copy()

    tech_df = pd.read_csv("tech.csv")
    tech_df['Date'] = pd.to_datetime(tech_df['Date'])

    # Filter tech_df to include only dates within the range of df
    tech_df = tech_df[(tech_df['Date'] >= df['Date'].min()) & (tech_df['Date'] <= df['Date'].max())]
    df_close = df_close[(df_close['Date'] >= df['Date'].min()) & (df_close['Date'] <= df['Date'].max())]
    df_indicators = df_indicators[(df_indicators['Date'] >= df['Date'].min()) & (df_indicators['Date'] <= df['Date'].max())]

    df.set_index('Date', inplace=True)
    df_close.set_index('Date', inplace=True)
    df_indicators.set_index('Date', inplace=True)
    tech_df.set_index('Date', inplace=True)


    df.to_csv('_sentiments_df.csv', index=True)

    # Save df_close to CSV
    df_close.to_csv('_df_close.csv', index=True)

    # Save df_indicators to CSV
    df_indicators.to_csv('_df_indicators.csv', index=True)

    # Save tech_df to CSV
    tech_df.to_csv('_tech_df.csv', index=True)




if __name__ == "__main__":
    main()