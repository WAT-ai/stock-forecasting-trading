

import numpy as np
import pandas as pd
import queue as Queue

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
from gym import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import DDPG


from datetime import datetime

from Quantitative import model as quant_model
from Quantitative.quant import Indicators
from Quantitative.evaluation_loop import Stock
import Quantitative.evaluation_loop as eval

from Technical.object_oriented.model import LstmModel

# from Sentimental.senti import SentimentClassifier
# from Sentimental import sentiment_main as sentiment 



def main():
    #CSV PATHS 
    SENTIMENT_TRAINING_CSV_PATH = "MCD_Sentiment_data.csv"

    SENTIMENT_HISTORICAL_CSV_PATH= "_sentiments_df.csv"
    TECHNICAL_HISTORICAL_CSV_PATH = "_tech_df.csv"
    INDICATORS_HISTORICAL_CSV_PATH = "_df_indicators.csv"
    CLOSING_HISTORICAL_CSV_PATH = "_df_close.csv"

    CURRENT_DATE = datetime.now().date()

    #USER INPUTS
    ALPHA_API = 'DNWQMLFC43J1PHDI'

    #RL HYPERPARAMETERS
    LEARNING_RATE = 0.01
    HOLDING_THRESHOLD = 0.01

    #OTHER PARAMETERS (for the compile function)
    CAPITAL = 10000 
    NUM_INDICATORS = 6
    NUM_SENTIMENTS = 1
    NUM_PREDICTIONS = 1


    #Historical Sentiment and Technical (from csv)
    historical_senti_df = pd.read_csv(SENTIMENT_HISTORICAL_CSV_PATH, index_col='Date')
    historical_technical_df = pd.read_csv(TECHNICAL_HISTORICAL_CSV_PATH, index_col='Date')
    historical_indicators_df = pd.read_csv(INDICATORS_HISTORICAL_CSV_PATH, index_col='Date')
    historical_closing_df = pd.read_csv(CLOSING_HISTORICAL_CSV_PATH, index_col='Date')

    ##====TECHNICAL====##
    # model = LstmModel(TICKER)
    # model.download_data()
    # model.preprocess_data()
    # model.build_model()
    # model.train_model()
    # pred = model.predict_future() #include type casting if necessary

    

    ##====SENTIMENT====##
    # sent_class = SentimentClassifier(SENTIMENT_TRAINING_CSV_PATH)
    # news_sentiment = sentiment.get_news_sentiment(TICKER, sent_class)
    # current_sentiment_df = pd.DataFrame([news_sentiment], index=[CURRENT_DATE], columns=['Sentiment'])


    ##====DDPG MODEL===#
    rl_env = quant_model.StockRLNNTrainingEnv(LEARNING_RATE)
    agent = rl_env.compile(
        CAPITAL, 
        1,
        HOLDING_THRESHOLD,
        NUM_INDICATORS,
        NUM_PREDICTIONS,
        NUM_SENTIMENTS,
        historical_closing_df, 
        historical_indicators_df,
        historical_technical_df,
        historical_senti_df
    )

    agent.learn(total_timesteps = 6, log_interval = 10)
    agent.predict()

    observation = rl_env.reset()
    action = agent.predict(observation)
    print("action")
    print(action)

    next_obs, reward, done, info = rl_env.step(action)
    print("observations, rewards, status, info")
    print(next_obs, reward, done, info)


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



    