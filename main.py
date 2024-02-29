

import numpy as np
import pandas as pd
import queue as Queue

import gym 

from datetime import datetime

from Quantitative import model as quant_model
from Quantitative.quant import Indicators
from Quantitative.evaluation_loop import Stock
import Quantitative.evaluation_loop as eval

from Technical.object_oriented.model import LstmModel

from Sentimental.senti import SentimentClassifier
from Sentimental import sentiment_main as sentiment 



def main():
    #CSV PATHS 
    SENTIMENT_TRAINING_CSV_PATH = "Sentimental/all-data.csv"
    SENTIMENT_HISTORICAL_CSV_PATH= ""
    TECHNICAL_HISTORICAL_CSV_PATH = ""
    CURRENT_DATE = datetime.now().date()

    #USER INPUTS
    TICKER = input("Enter ticker ")
    ALPHA_API = 'DNWQMLFC43J1PHDI'

    #RL HYPERPARAMETERS
    LEARNING_RATE = 0.01
    HOLDING_THRESHOLD = 0.01

    #OTHER PARAMETERS (for the compile function)
    CAPITAL = 10000 
    NUM_ASSETS = 10 
    NUM_INDICATORS = 0 
    NUM_SENTIMENTS = 0
    NUM_PREDICTIONS = 0


    #Historical Sentiment and Technical (from csv)
    # historical_senti_df = pd.read_csv(SENTIMENT_HISTORICAL_CSV_PATH, index_col='Date')
    # historical_technical_df = pd.read_csv(TECHNICAL_HISTORICAL_CSV_PATH, index_col='Date')

    #===INDICATORS===#
    stock_data = Indicators(api_key=ALPHA_API)
    stock_data.get_stock_data(TICKER, 2400)
    stock_data.preprocess_stock_data()

    #split the data into training and for predictions 
    historical_indicators_df = stock_data.df_normalized.iloc[:-1]
    current_indicators_df = stock_data.df_normalized.iloc[ -1] #include type casting if necessary 

    ##====TECHNICAL====##
    model = LstmModel(TICKER)
    model.download_data()
    model.preprocess_data()
    model.build_model()
    model.train_model()
    pred = model.predict_future() #include type casting if necessary

    

    ##====SENTIMENT====##
    sent_class = SentimentClassifier(SENTIMENT_TRAINING_CSV_PATH)
    example = sent_class.get_sentiment("NVIDIA forecasts increased demand for their chips, sales up 50%")
    print(example)
    news_sentiment = sentiment.get_news_sentiment(TICKER, sent_class)
    current_sentiment_df = pd.DataFrame([news_sentiment], index=[CURRENT_DATE], columns=['Sentiment'])
    print(current_sentiment_df)


    ##====DDPG MODEL===#
    rl_env = quant_model.StockRLNNTrainingEnv(LR)
    agent = rl_env.compile(...)

    agent.learn(total_timestamps = 1000, log_interval = 10)
    agent.predict()

    observation = rl_env.reset()
    action = agent.predict(observation)
    print(action)


    ##====EVALUATION===###
    portfolio_data = {'MCD': Queue()}
    

    ##WE NEED OUR MODEL TO RETURN THESE
    result = 'sell'
    num_stocks = 10

    ##ONLY REUTRNS METRICS ON SELL 
    sharpe_ratio, total_percent_return, win_loss_ratio = eval.evaluate_stock(portfolio_data, 'AAPL', result, num_stocks)

    return None 

if __name__ == '__main__':
    main()



    