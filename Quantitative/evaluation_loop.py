from alpha_vantage.timeseries import TimeSeries
import numpy as np
from queue import Queue
from typing import List, Dict, Tuple


API_KEY = 'YOUR API KEY HERE'

class Stock: 

    """
    Class to represent a stock object.
    Attributes: symbol (string): Stock symbol, price (float): Stock price
    Methods: init, __str__, getReturn
    """
    def __init__(self, symbol, price):   #add data types
        self.symbol = symbol
        self.price = price

    def __str__(self) -> str:
        return f"(Symbol: {self.symbol}, Price: {self.price})"
    
    def ret(self, current_price) -> float:
        return (current_price - self.price) / self.price
    


def get_sharpe_ratio(sell_stock_list: List[Stock], recent_price: float, RFR: float = 0.05, N: int = 252) -> float:
    """
    Gets the sharpe ratio of a stocks 
    Input: stock object (Stock), recent price (FLoat), Risk Free Rate (float), N (int) (for annualization of sharpe ratio)
    Output: Sharpe Ratio (float)

    Note(s): 
    free to modify risk free rate, just took 0.05 a a basis from website below (3 month t-bill)
    https://ycharts.com/indicators/3_month_t_bill#:~:text=Basic%20Info,long%20term%20average%20of%204.19%25.
    """

    # store percentage 
    list_returns = np.array([(recent_price - stock.price)/stock.price for stock in sell_stock_list])

    # Calculate the Sharpe Ratio
    excess_returns = (np.mean(list_returns) - RFR) # trading days in a year 
    std_dev = np.std(list_returns)  #to annualize sharpe ratio
    print(std_dev)

    return excess_returns / std_dev


def get_total_percentage_return(sell_stock_list: List[Stock], recent_price: float, num_stocks: int) -> float:
    """
    Fetches the daily percentage returns of a stock.
    Input: Stock object (Stock), most recent stock price (float)
    Output: daily percentage returns for the stock (float)
    
    """
    # Calculate daily percentage returns
    returns_percentage = 0
    #add stock 
    for stock in sell_stock_list:
        print( (recent_price - stock.price) / stock.price * 100)
        returns_percentage += (recent_price - stock.price) / stock.price * 100
        

    return returns_percentage / num_stocks #average return for all stocks

def get_recent_stock_price(symbol: str) -> float:
    """
    Fetches most recent stock price for a given stock symbol.
    Input: Symbol of the stock (string), TimeSeries object, API key (string)
    Output: Most recent stock price (float)
    """

    ts = TimeSeries(key=API_KEY, output_format='pandas')

    # Fetch most recent stock price
    data, _ = ts.get_quote_endpoint(symbol=symbol)
    recent_price = data['05. price'][0]

    return recent_price

def get_win_loss_ratio(wl_arr: List[int]) -> float:
    """
    Calculates the win loss ratio for a given list of win/loss values.
    Input: list of win/loss values (list)
    Output: win/loss ratio (float)
    """
    wins = sum(wl_arr)
    print(len(wl_arr))
    losses = len(wl_arr) - wins

    ## No losses, return 100% win ratio (1)
    if losses == 0:
        return 1
    
    return wins / losses

#assuminh curr stock is an object htat contains the price and the ticket
def evaluate_stock(portfolio_data: Dict[str, Queue], symbol: str, result: str, num_stocks: int) -> Tuple[float, float, float]:
    """
    Evaluates a stock using the Sharpe Ratio and Daily Percentage Returns.
    Input: stock_prediction_data (dict, key - ticker: val - queue of stock obj) symbol (str),
           result (buy sell or hold) (string - for now), num_stocks to sell/buy (int)
    Output: Sharpe Ratio (float), Daily Percentage Returns (float)
    """\
    

    stock_queue = portfolio_data[symbol]
    #if we decide to hold or we have no stocks do nothing 
    if not portfolio_data or result == "hold":
        return None

    #get the most recent price of the stock (we need for by and sell)
    recent_price = get_recent_stock_price(symbol)

    #if we decide to sell calcualte evaluation metrics
    if result == "sell" and stock_queue.qsize() > 0: 
       
        #get the desired stock queue based on the ticker 
        sell_stock_list = [] #list to hold the stocks in the queue

        #get the stocks in the queue
        for _ in range(num_stocks):
            if not stock_queue.empty():
                sell_stock_list.append(stock_queue.get())

        total_percent_return = get_total_percentage_return(sell_stock_list, recent_price, len(sell_stock_list))
        sharpe_ratio = get_sharpe_ratio(sell_stock_list, recent_price)
       


        #calculate win loss ratio 
        wl_arr = []
        for stock in sell_stock_list:
            ret = stock.ret(recent_price)
            if ret > 0: 
                wl_arr.append(1)
            else: 
                wl_arr.append(0)

    
        win_loss_ratio = get_win_loss_ratio(wl_arr)

        return sharpe_ratio, total_percent_return, win_loss_ratio
    

    #if we decide to buy add to queue
    elif result == "buy":
        for _ in range(num_stocks):
            curr_stock = Stock(symbol, recent_price)
            stock_queue.put(curr_stock)

        return None, None, None

#EXAMPLE
# Mocking the TimeSeries class and get_quote_endpoint function for testing
class TimeSeries:
    def __init__(self, key, output_format):
        pass

    def get_quote_endpoint(self, symbol):
        # Mocking data for testing
        data = {'05. price': [150.0]}  # Mocking recent price
        return data, None


# # Implementing the Stock class and required functions here

# # Mock data for testing
portfolio_data = {'AAPL': Queue(), 'GOOG': Queue()}
# portfolio_data['AAPL'].put(Stock('AAPL', 154.0))
# portfolio_data['AAPL'].put(Stock('AAPL', 151.0))
# portfolio_data['AAPL'].put(Stock('AAPL', 145.0))
# portfolio_data['GOOG'].put(Stock('GOOG', 200.0))

# # Mock result for testing
result = 'buy'

# # Number of stocks to buy/sell
num_stocks = 2
# # Testing the evaluate_stock function
sharpe_ratio, total_percent_return, win_loss_ratio = evaluate_stock(portfolio_data, 'AAPL', result, num_stocks)
print("Sharpe Ratio:", sharpe_ratio)
print("Total Percentage Return:", total_percent_return)
print("Win Loss Ratio:", win_loss_ratio)
print(portfolio_data['AAPL'].qsize())  # Expected output: 2
