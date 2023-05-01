import numpy as np
from backtesting import Backtest, Strategy
import datetime as dt
from models import nn_trader_decision, dense_model, lstm_model
from preprocess import get_data_yfinance, get_data_tushare

NUM_DAYS = 10
TICKER = "GOOG"
START_DATE = dt.datetime(2008, 1, 1)
END_DATE = dt.datetime(2023, 4, 26)


class LSTMNeuralNetTrader(Strategy):

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.num_days = None
        self.ticker = None
        self.start_date = None
        self.end_date = None
        self.lstm = None
        self.dense = None
        self.y_hats_lstm = None
        self.y_hats_dense = None

    def init(self):
        self.num_days = NUM_DAYS
        self.ticker = TICKER
        self.start_date = START_DATE
        self.end_date = END_DATE
        x_train, y_train, x_test, y_test, test_data_frames = get_data_yfinance(self.ticker, self.num_days,
                                                                               self.start_date, self.end_date)

        # LSTM Model
        self.lstm, self.y_hats_lstm = lstm_model(
            x_train_lstm=x_train,
            x_test_lstm=x_test,
            y_train_lstm=y_train,
            y_test_lstm=y_test
        )
        self.lstm._name = 'lstm'

    def next(self):
        if self.data.Close.shape[0] >= self.num_days:
            latest_close_prices = np.array(self.data.Close[-1 * self.num_days:], dtype=float)
            # Reshape the input to fit the model and make matrix multiplication possible. When training, shape was:
            #   number of training examples x num_days x 1 (since LSTMs require 3D input).
            # So, we use 1 x num_days x 1 here to make a single prediction.
            latest_close_prices = np.reshape(latest_close_prices, (1, self.num_days, 1))
            if nn_trader_decision(
                    model=self.lstm,
                    latest_close=self.data.Close[-1],
                    last_n_close_prices=latest_close_prices
            ):
                self.buy(size=2)
            elif self.position.size > 0:
                self.sell(size=self.position.size)
        else:
            pass


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, test_data_frames = get_data_yfinance(TICKER, NUM_DAYS,
                                                                           START_DATE, END_DATE)
    bt = Backtest(
        data=test_data_frames,
        strategy=LSTMNeuralNetTrader,
        cash=100000,
        commission=0,
    )
    lstm_output = bt.run()
    print(lstm_output)
    bt.plot()
