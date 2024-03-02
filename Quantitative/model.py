import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
from gym import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import DDPG


class StockRLFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Dict, num_indicators=0, num_price_predictions=0, num_sentiments=0, hidden_size=0):
        super().__init__(observation_space, num_indicators + num_price_predictions + num_sentiments)
        self.indicator_network = [
            nn.LSTM(num_indicators, 20),
            nn.LSTM(20, 20),
            nn.Linear(20, hidden_size),
        ]

        self.price_prediction_network = [
            nn.LSTM(num_price_predictions, 20),
            nn.LSTM(20, 20),
            nn.Linear(20, hidden_size)
        ]

        self.sentiment_network = [
            nn.LSTM(num_sentiments, 20),
            nn.Linear(20, hidden_size)
        ]
    
    def indicator_forward(self, indicators):
        x = indicators
        for layer in self.indicator_network[:-1]:
            x, _ = layer(x)
            x = F.relu(x)
        return F.relu(self.indicator_network[-1](x))

    def price_prediction_forward(self, price_predictions):
        x = price_predictions
        for layer in self.price_prediction_network[:-1]:
            x, _ = layer(x)
            x = F.relu(x)
        return F.relu(self.price_prediction_network[-1](x))
    
    def sentiment_forward(self, sentiments):
        x = sentiments
        for layer in self.sentiment_network[:-1]:
            x, _ = layer(x)
            x = F.relu(x)
        return F.relu(self.sentiment_network[-1](x))

    def forward(self, observations):
        x = torch.cat((self.indicator_forward(observations["indicators"]), 
                      self.price_prediction_forward(observations["price_predictions"]), 
                      self.sentiment_forward(observations["sentiments"])))
        
        if x.shape[0] == 3:
            return torch.flatten(x)
        
        return torch.cat((self.indicator_forward(observations["indicators"]), 
                      self.price_prediction_forward(observations["price_predictions"]), 
                      self.sentiment_forward(observations["sentiments"])), 1)


class StockRLNNTrainingEnv(gym.Env):
    def __init__(self, learning_rate):
        super(StockRLNNTrainingEnv, self).__init__()

        # RL specific variables
        self.learning_rate = learning_rate
        self.current_step = 0

        #stock trading environment-specific variables
        self.num_assets = 0
        self.transaction_cost_percentage = 0

        # stock trading bot-specific variables
        self.capital = 0
        self.left_over_balance = 0
        self.hold_threshold = 0
        self.asset_quantities = None

        self.prev_left_over_balance = None
        self.prev_asset_quantities = None
        self.prev_closing_prices = None

        self.num_indicators = None
        self.num_price_predictions = None
        self.num_sentiments = None

        self.action_space = None
        self.observation_space = None

        # default values for reset
        self.base_left_over_balance = 0
        self.base_prev_closing_prices = None
        self.base_capital = None
        self.base_asset_quantities = None
        self.base_prev_left_over_balance = None
        self.base_prev_asset_quantities = None

        # dataset storage
        self.X_closing_prices = None
        self.X_indicators = None
        self.X_price_predictions = None
        self.X_sentiments = None
    
    def compile(self, capital, num_assets, hold_threshold, num_indicators, num_price_predictions, num_sentiments, X_closing_prices: pd.DataFrame, X_indicators: pd.DataFrame, X_price_predictions: pd.DataFrame, X_sentiments: pd.DataFrame):
        self.num_assets = num_assets
        self.capital = capital
        self.hold_threshold = hold_threshold

        self.num_indicators = num_indicators
        self.num_price_predictions = num_price_predictions
        self.num_sentiments = num_sentiments
        
        self.X_closing_prices = X_closing_prices
        self.X_indicators = X_indicators
        self.X_price_predictions = X_price_predictions
        self.X_sentiments = X_sentiments

        self.asset_quantities = np.zeros(self.num_assets)

        self.prev_closing_prices = self.X_closing_prices.iloc[0]

        self.base_prev_closing_prices = self.prev_closing_prices
        self.base_asset_quantities = np.zeros(self.num_assets)
        self.base_capital = self.capital

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_assets,))
        self.observation_space = spaces.Dict({
            "indicators": spaces.Box(low=0, high=10000000, shape=(self.num_indicators,)),
            "price_predictions": spaces.Box(low=0, high=10000000, shape=(self.num_price_predictions,)),
            "sentiments": spaces.Box(low=0, high=10000000, shape=(self.num_sentiments,))
        })

        return self.build_model()

    def sample_data(self, index=None):
        if index is None:
            return np.array(self.X_closing_prices.iloc[self.current_step]), \
                    {
                        "indicators": np.array(self.X_indicators.iloc[self.current_step]),
                        "price_predictions": np.array(self.X_price_predictions.iloc[self.current_step]), 
                        "sentiments": np.array(self.X_sentiments.iloc[self.current_step])
                    }
        return np.array(self.X_closing_prices.iloc[index]), \
                {
                    "indicators": np.array(self.X_indicators.iloc[index]),
                    "price_predictions": np.array(self.X_price_predictions.iloc[index]), 
                    "sentiments": np.array(self.X_sentiments.iloc[index])
                }
    
    def calc_portfolio_value(self, portfolio: np.array, left_over_balance: float, closing_prices: np.array):
        # calculates portfolio value based on closing prices
        
        return np.dot(closing_prices, portfolio) + left_over_balance
    
    def create_denormalized_action_space(self, closing_prices: np.array):
        # creates a vector containing the maximum number of shares
        # that can be purchased with the current total portfolio value

        return self.calc_portfolio_value(self.asset_quantities, self.left_over_balance, closing_prices) / closing_prices
    
    def calc_reward(self, closing_prices: np.array):
        # calculates reward for the current state given closing prices

        if self.prev_asset_quantities is not None and self.prev_left_over_balance  is not None and self.prev_closing_prices is not None:
            return self.calc_portfolio_value(self.asset_quantities, self.left_over_balance, closing_prices) - \
                self.calc_portfolio_value(self.prev_asset_quantities, self.prev_left_over_balance, self.prev_closing_prices) - \
                self.transaction_cost_percentage * np.dot(abs(self.asset_quantities - self.prev_asset_quantities), closing_prices)
        raise ValueError("Cannot create reward when previous (asset quantites, left over balance, closing prices) not present.")
    
    def build_model(self):
        agent = DDPG(
            env=self,
            policy="MultiInputPolicy",
            learning_rate=self.learning_rate,
            policy_kwargs=dict(
                net_arch=[3 * self.num_assets, 32, self.num_assets],
                features_extractor_class=StockRLFeatureExtractor,
                features_extractor_kwargs=dict(
                    num_indicators=self.num_indicators,
                    num_price_predictions=self.num_price_predictions,
                    num_sentiments=self.num_sentiments,
                    hidden_size=self.num_assets
                )
            )
        )

        return agent
    
    def get_buy_sell_holds(self):
        # action: array of numbers all between -1 and 1
        # action[i]:
        # -self.hold_threshold < action[i] < self.hold_threshold: hold
        # -1 < action[i] < -self.hold_threshold: sell
        # self.hold_threshold < action[i]: buy

        pass
        
    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.X_closing_prices.index):
            self.current_step = 1
        closing_prices, obs = self.sample_data()
        self.prev_closing_prices, _ = self.sample_data(self.current_step - 1)

        self.prev_left_over_balance = self.left_over_balance
        self.prev_asset_quantities = self.asset_quantities

        self.capital = self.calc_portfolio_value(self.asset_quantities, self.left_over_balance, closing_prices)
        
        # sell and hold stocks based on the action
        for i, act in enumerate(action):
            if abs(act) < self.hold_threshold:
                self.asset_quantities[i] = self.prev_asset_quantities[i]
            elif act < 0:
                sell_amt = -int(act * self.asset_quantities[i])
                self.asset_quantities[i] -= sell_amt
                self.left_over_balance += sell_amt * closing_prices[i]
        
        max_actions = self.create_denormalized_action_space(closing_prices)

        # buy stocks based on the action
        for i, act in enumerate(action):
            if act > 0:
                buy_amt = int(act * max_actions[i])
                if buy_amt * closing_prices[i] > self.left_over_balance:
                    while buy_amt * closing_prices[i] > self.left_over_balance:
                        buy_amt -= 1
                self.asset_quantities[i] += buy_amt
                self.left_over_balance -= buy_amt * closing_prices[i]

        self.left_over_balance = self.capital - np.dot(closing_prices, self.asset_quantities)
        
        reward = self.calc_reward(closing_prices)
        done = self.calc_portfolio_value(self.asset_quantities, self.left_over_balance, closing_prices) <= 0
        return obs, reward, done, {}
    
    def reset(self):
        self.prev_closing_prices = self.base_prev_closing_prices
        self.capital = self.base_capital
        self.left_over_balance = self.base_left_over_balance
        self.asset_quantities = self.base_asset_quantities
        self.prev_left_over_balance = self.base_prev_left_over_balance
        self.prev_asset_quantities = self.base_prev_asset_quantities
        self.current_step = random.randint(1, len(self.X_closing_prices.index) - 1)
        _, new_data = self.sample_data()
        return new_data

    def render(self):
        pass


def main (): 
     # Create sample data frames for closing prices, indicators, price predictions, and sentiments
    num_days = 100
    num_assets = 5

    # Sample closing prices data frame
    closing_prices_df = pd.DataFrame(np.random.rand(num_days, num_assets) * 100, columns=[f"Asset_{i+1}" for i in range(num_assets)])


    # Sample indicators data frame
    indicators_df = pd.DataFrame(np.random.rand(num_days, num_assets) * 100, columns=[f"Indicator_{i+1}" for i in range(num_assets)])  

    # Sample price predictions data frame
    price_predictions_df = pd.DataFrame(np.random.rand(num_days, num_assets) * 100, columns=[f"Prediction_{i+1}" for i in range(num_assets)])

    # Sample sentiments
    sentiments_df = pd.DataFrame(np.random.rand(num_days, num_assets) * 100, columns=[f"Sentiment{i+1}" for i in range(num_assets)])

    # Now, let's interact with the environment and agent

    # Instantiate the environment and build the model
    rl_env = StockRLNNTrainingEnv(0.01)

    # Compile the environment with necessary parameters
    agent = rl_env.compile(capital=10000, num_assets=num_assets, hold_threshold=0.1,
                num_indicators=5, num_price_predictions=num_assets, num_sentiments=5,
                X_closing_prices=closing_prices_df,
                X_indicators=indicators_df,
                X_price_predictions=price_predictions_df,
                X_sentiments=sentiments_df)

    # Train the model
    agent.learn(total_timesteps=10000, log_interval=10)

    # Make predictions
    obs = rl_env.reset()
    # action, _ = agent.predict(obs)

    # # Perform a step in the environment
    # next_obs, reward, done, info = env.step(action)

    # # Reset the environment
    # obs = env.reset()

main()
