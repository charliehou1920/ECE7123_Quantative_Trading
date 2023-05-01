import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf
import tushare as ts
import math


# We use this method to fetch the stock price data from US Stock Market
def get_data_yfinance(ticker, num_days, start_date, end_date):
    """
    Reads data out of the API of yfinance.
    Then formats data into training and testing input/output sets.

    :param ticker: ticker of the stock we want to build strategy on
    :param num_days: number of days used for back test since we are in a scale of day trading
    :param start_date: start date for the back test (it should be in type of datetime)
    :param end_date: end date for the back test (it should be in type of datetime)
    :return: Training and testing inputs and expected outputs.
            DataFrame of test data to be used for back test.
    """
    yf.pdr_override()
    df = pdr.get_data_yahoo(ticker, start_date, end_date)
    close_prices = df[['Adj Close']]
    # Now the dataframe has two columns: date and close price
    close_prices.columns = ['Price']
    training_data_len = math.ceil(len(close_prices) * 0.8)
    train_prices = close_prices[:training_data_len]
    test_prices = close_prices[training_data_len:]

    # Dealing with training dataset
    x_train_prices = []
    y_train_prices = []
    for i in range(num_days, len(train_prices)):
        x_train_prices.append(train_prices[i - num_days:i])
        y_train_prices.append(train_prices.iloc[i])
    x_train_prices, y_train_prices = np.array(x_train_prices, dtype=float), np.array(y_train_prices, dtype=float)
    # reshape matrices to fit the model: number of training examples * num_days * 1
    # matrices are 3D as LSTM model in models.py requires 3D inputs
    x_train_prices = np.reshape(x_train_prices, (x_train_prices.shape[0], x_train_prices.shape[1], 1))

    # Dealing with test dataset
    x_test_prices = []
    y_test_prices = []
    for i in range(num_days, len(test_prices)):
        x_test_prices.append(test_prices[i - num_days:i])
        y_test_prices.append(test_prices.iloc[i])
    x_test_prices, y_test_prices = np.array(x_test_prices, dtype=float), np.array(y_test_prices, dtype=float)
    # reshape matrices to fit the model: number of testing examples x num_days x 1
    x_test_prices = np.reshape(x_test_prices, (x_test_prices.shape[0], x_test_prices.shape[1], 1))
    test_data_frames = df.iloc[0:x_test_prices.shape[0], :]

    return x_train_prices, y_train_prices, x_test_prices, y_test_prices, test_data_frames


# We use this method to fetch the stock price data from Chinese Stock Market
def get_data_tushare(ticker, num_days, start_date, end_date):
    """
        Reads data out of the API of Tushare.
        Then formats data into training and testing input/output sets.

        :param ticker: ticker of the stock we want to build strategy on
        :param num_days: number of days used for back test since we are in a scale of day trading
        :param start_date: start date for the back test (it should be in type of string)
        :param end_date: end date for the back test (it should be in type of string)
        :return: Training and testing inputs and expected outputs.
                DataFrame of test data to be used for back test.
        """
    pro = ts.pro_api('73c1bd181e417b067a37d836c714f2b683feeaa47a0da1f4f27c9dca')
    df = pro.daily(ts_code=ticker, start_date=start_date, end_date=end_date)
    df = df.iloc[::-1]
    df = df.set_index('trade_date')
    close_prices = df[['close']]
    close_prices.columns = ['Price']
    training_data_len = math.ceil(len(close_prices) * 0.8)
    train_prices = close_prices[:training_data_len]
    test_prices = close_prices[training_data_len:]

    # Dealing with training dataset
    x_train_prices = []
    y_train_prices = []
    for i in range(num_days, len(train_prices)):
        x_train_prices.append(train_prices[i - num_days:i])
        y_train_prices.append(train_prices.iloc[i])
    x_train_prices, y_train_prices = np.array(x_train_prices, dtype=float), np.array(y_train_prices, dtype=float)
    # reshape matrices to fit the model: number of training examples * num_days * 1
    # matrices are 3D as LSTM model in models.py requires 3D inputs
    x_train_prices = np.reshape(x_train_prices, (x_train_prices.shape[0], x_train_prices.shape[1], 1))

    # Dealing with test dataset
    x_test_prices = []
    y_test_prices = []
    for i in range(num_days, len(test_prices)):
        x_test_prices.append(test_prices[i - num_days:i])
        y_test_prices.append(test_prices.iloc[i])
    x_test_prices, y_test_prices = np.array(x_test_prices, dtype=float), np.array(y_test_prices, dtype=float)
    # reshape matrices to fit the model: number of testing examples x num_days x 1
    x_test_prices = np.reshape(x_test_prices, (x_test_prices.shape[0], x_test_prices.shape[1], 1))
    test_data_frames = df.iloc[0:x_test_prices.shape[0], :]

    return x_train_prices, y_train_prices, x_test_prices, y_test_prices, test_data_frames




